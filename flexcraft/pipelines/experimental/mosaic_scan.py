# Multi-target binder hallucination from salad-initialized sequences.
#
# This is `flexcraft.pipelines.bind.mosaic` swept over a fasta of targets, with the
# blueprint-comparison metrics of `flexcraft.pipelines.experimental.landscape` bolted onto
# every re-prediction. For each target we run a fixed number of attempts; each attempt
# starts from a *fresh* salad backbone whose ProteinMPNN distribution seeds the two-stage
# simplex-APGM hallucination. Everything lands in one combined results.csv.
#
# The structure predictor is selectable with `--structure_model`:
#   joltz        Boltz-2 (default; behaviour unchanged)
#   protenix_v1  Protenix base v1.0.0   (checkpoint "protenix_base_default_v1.0.0")
#   protenix_v2  Protenix v2            (checkpoint "protenix-v2")
# Everything downstream of the predictor (loss, metrics, CSV/PDB layout, --resume) is shared;
# the two backends differ only in how inputs are built (see _complex_spec / _to_input below).
#
# Optional blueprint-distogram loss (--distogram_loss True, requires --dock_blueprint True): adds
# `distogram_loss_weight * distogram_nll` to the hallucination loss, where the target is the
# distogram of the docked salad blueprint over the WHOLE complex (binder + target). Boltz never
# sees the blueprint's coordinates otherwise, so this is a direct steer of the binding site.
# --distogram_loss_weights "0.0,0.1,1.0" sweeps the weight: each attempt builds ONE blueprint and
# ProteinMPNN init, then runs one full hallucination per weight from that same starting point
# (include 0.0 for an in-process control that also logs distogram_nll).
#
# Note on cost: `params["structure_input"]` has shape binder_length + len(target), so the jitted
# loss retraces once per target (targets are outer, attempts inner, so each compile is
# amortized over `--attempts_per_target` designs). For Boltz, `pad=True` does not help --
# pad_boltz_atom_features_for_compilation pads only the atom axis, never the token axis.

import os
import re
import time
import shutil
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import optax

from flexcraft.utils import Keygen, parse_options, data_from_salad
from flexcraft.sequence.sample import *
from flexcraft.data.data import DesignData
from flexcraft.sequence.mpnn import make_pmpnn
import flexcraft.sequence.aa_codes as aas
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.hallucination.losses.distogram import blueprint_distogram, distogram_nll
from flexcraft.structure.boltz._model import Joltz2
from flexcraft.structure.boltz._data import JoltzSpec
from flexcraft.structure.common._result import AF3LikeResult
from flexcraft.structure.metrics import RMSD
from flexcraft.files import FastaFile, ScoreCSV

opt = parse_options(
    "mosaic binder hallucination swept over a fasta of targets",
    # inputs
    targets="runs/targets.fasta",
    only_targets="none",
    attempts_per_target=10,
    length=80,
    # parameters
    # Which structure predictor to hallucinate against: joltz | protenix_v1 | protenix_v2.
    # Use a distinct --out_path per model: results.csv gains a `structure_model` column for the
    # Protenix models, and --resume refuses to continue a run made with a different model.
    structure_model="joltz",
    pmpnn_path="params/solmpnn/v_48_030.pkl",
    salad_path="params/salad/default_vp_scaled-200k.jax",
    boltz_path="params/boltz/",
    # outputs
    out_path="out",
    msa_dir="none",
    msa_only="False",
    msa_sleep=5.0,
    # protocol
    sample_structure="True",
    stage1_steps=100,
    stage2_steps=50,
    init_temperature=1.0,
    std_buffer=3.0,
    clear_jit_cache="False",
    seed=42,
    # Resume an interrupted run at the same --out_path: append to the existing CSVs instead
    # of truncating them, and skip any (target, attempt) already fully recorded in
    # results.csv (both stages present). Requires identical options to the original run --
    # ScoreCSV(append=True) checks the column schema and refuses to append on a mismatch, but
    # a changed --length/--dock_blueprint/etc. can still change what a "resumed" attempt
    # would compute even if the schema happens to match, so keep the invocation the same.
    resume="False",
    # --- docked blueprint mode -----------------------------------------------------------
    dock_blueprint="False",       # generate the blueprint AGAINST the target at a hotspot
    salad_ve_path="params/salad/default_ve_scaled-200k.jax",
    # CSV: target,hotspots  (e.g. "proteinG,A20-30"). Optional: "none" makes every target residue
    # eligible for the docking site. Hotspot and coldspot files are independent -- use either,
    # both, or neither (neither = a random surface patch per attempt).
    hotspot_file="none",
    # CSV: target,coldspots -- same format and 1-based fasta numbering as the hotspot file, but
    # OPTIONAL per target (a missing or blank row means no coldspots for that target). Docking
    # centres are kept --coldspot_radius Angstrom away from every coldspot CA, so the blueprint
    # is placed away from those residues. Needs --dock_blueprint True.
    coldspot_file="none",
    target_pdb_dir="none",        # default {out_path}/target_structures
    # Use the target's cached structure (single-sequence prediction) as a TEMPLATE (Boltz or Protenix) for
    # every complex prediction, instead of an MSA. Independent of --dock_blueprint: predicting
    # and caching the target structure happens whenever either flag is set.
    target_use_template="False",
    epitope_cutoff=8.0,
    epitope_cutoff_2=10.0,        # second cutoff, to check the metric isn't cutoff-driven
    # Blueprint-distogram loss (needs --dock_blueprint True): cross-entropy between the predicted
    # complex distogram and the distogram of the docked blueprint complex (binder + target),
    # weighted by --distogram_loss_weight and added to the hallucination loss in both stages.
    distogram_loss="False",
    distogram_loss_weight=0.1,
    # Comma-separated list, e.g. "0.0,0.1,1.0": run one hallucination per weight against the SAME
    # blueprint and initial sequence of every attempt. Needs --distogram_loss True and overrides
    # --distogram_loss_weight. "none" = the single --distogram_loss_weight. Adds a `distogram_weight`
    # column to the CSVs and a `_w{weight}` tag to the attempt/success PDB names.
    distogram_loss_weights="none",
    # salad binder potentials (defaults from bind/config.py BINDER_OPT)
    radius=10.0,
    clash_radius=10.0,
    # Clearance of the docking CENTRE from every coldspot CA (see --coldspot_file). This is a
    # centroid clearance, not a surface clearance: on salad's 80-residue blueprints the radius of
    # gyration is ~12 A and the farthest residues sit 16-28 A from the centroid (mean ~22 A), so
    # the 15 A default can still leave the binder touching a coldspot. Check
    # coldspot_hit_blueprint in results.csv and raise this if it does. The trade-off is fewer
    # viable docking centres; the sweep prints how many survive vs. how many exist without the
    # filter, and skips a target (skipped.csv) if the coldspots leave none.
    coldspot_radius=20.0,
    min_center_distance=2.0,
    clash_lr=5e-3,
    compact_lr=1e-4,
    contact_lr=1e-2,
    relax_cutoff=3.0,
    fix_center_threshold=2.0,
    prev_threshold=0.9,
    l_bias=0.0,
    h_bias=0.0,
    e_bias=0.0,
)

# Checkpoint names are the ones protenix.backend.KNOWN_MODELS resolves (downloaded on first
# use from the nickrb/protenij HuggingFace repo into ~/.protenix).
PROTENIX_CHECKPOINTS = {
    "protenix_v1": "protenix_base_default_v1.0.0",
    "protenix_v2": "protenix-v2",
}
assert opt.structure_model == "joltz" or opt.structure_model in PROTENIX_CHECKPOINTS, (
    f"--structure_model must be one of {['joltz', *PROTENIX_CHECKPOINTS]}, "
    f"got '{opt.structure_model}'")
use_protenix = opt.structure_model in PROTENIX_CHECKPOINTS
if use_protenix:
    # Imported only when selected, so a Joltz run never depends on the protenix package.
    import equinox as eqx
    from protenix.backend import download_data, _resolve_model_path
    from flexcraft.structure.protenix._model import Protenix, ProtenixEvaluator
    from flexcraft.structure.protenix._data import ProtenixSpec, _run_msa_search
    from flexcraft.structure.protenix._result import ProtenixPrediction

binder_length = opt.length
resume = opt.resume == "True"
dock_blueprint = opt.dock_blueprint == "True"
use_target_template = opt.target_use_template == "True"
# In docked mode the blueprint comes from the target-conditioned ve model, so the free-monomer
# vp path is switched off entirely (different model, different length, dead cloud_std).
sample_structure = (opt.sample_structure == "True") and not dock_blueprint
use_distogram_loss = opt.distogram_loss == "True"
assert not use_distogram_loss or dock_blueprint, (
    "--distogram_loss True requires --dock_blueprint True: the loss target is the distogram of "
    "the docked blueprint COMPLEX, which only exists when the blueprint is generated against "
    "the target.")
sweeping_weights = opt.distogram_loss_weights != "none"
assert not sweeping_weights or use_distogram_loss, (
    "--distogram_loss_weights needs --distogram_loss True")

def _weight_tag(w: float) -> str:
    """Canonical text form of a weight: the CSV value, the PDB-name tag and the resume key."""
    return repr(float(w))

# One (weight, tag) entry per hallucination run made from each blueprint. tag is None outside sweep
# mode, so the CSV schema, file names and resume keys of single-weight runs are unchanged.
if sweeping_weights:
    _weights = [float(v) for v in opt.distogram_loss_weights.split(",") if v.strip() != ""]
    assert _weights and len(set(_weights)) == len(_weights) and min(_weights) >= 0.0, (
        f"--distogram_loss_weights must be a non-empty list of unique non-negative numbers, got "
        f"'{opt.distogram_loss_weights}'")
    run_weights = [(w, _weight_tag(w)) for w in _weights]
elif use_distogram_loss:
    run_weights = [(opt.distogram_loss_weight, None)]
else:
    run_weights = [(None, None)]
if use_distogram_loss:
    print(f"distogram loss: {len(run_weights)} run(s) per blueprint, weights "
          f"{[w for w, _ in run_weights]}")
use_hotspots = opt.hotspot_file != "none"
use_coldspots = opt.coldspot_file != "none"
assert not use_coldspots or dock_blueprint, (
    "--coldspot_file requires --dock_blueprint True: coldspots steer where the docked "
    "blueprint is placed, which only exists in docked mode. Without the flag they would be "
    "silently ignored.")
# No per-target hotspot spec reaches the Boltz loss: the hotspot biases the BLUEPRINT only, so
# that the ProteinMPNN sequence is the single causal channel from pose to hallucination. With
# hotspot_index=None, metrics()'s hotspot_selector falls back to `chain_index != 0`, so
# `hotspot_contacts` degrades to the symmetrized binder<->whole-target contact score.
hotspot_index = None

# ---------------------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------------------

def _safe_name(raw: str) -> str:
    """Sanitize a fasta header into something safe for filenames and for ScoreCSV.

    ScoreCSV does no quoting (files/csv.py:31), so a comma anywhere in a row corrupts the
    whole CSV. Keep only filename- and CSV-safe characters.
    """
    token = raw.split()[0] if raw.split() else "unnamed"
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", token)

selected = None if opt.only_targets == "none" else set(opt.only_targets.split(","))
targets = []
seen = set()
# Iterate .items() rather than .line_dict: the dict silently collapses duplicate headers.
for raw_name, raw_sequence in FastaFile(opt.targets).items():
    name = _safe_name(raw_name)
    assert name not in seen, f"duplicate target name after sanitization: {name}"
    seen.add(name)
    if selected is not None and name not in selected:
        continue
    targets.append((name, raw_sequence.strip().upper()))
if selected is not None:
    missing = selected - {n for n, _ in targets}
    assert not missing, (
        f"--only_targets names not in {opt.targets}: {sorted(missing)}. "
        f"available: {sorted(seen)}")
assert targets, f"no targets selected from {opt.targets}"
print(f"{len(targets)} targets: " + ", ".join(f"{n}({len(s)})" for n, s in targets))

# ---------------------------------------------------------------------------------------
# hotspot file and optional coldspot file (docked mode only)
#
# Numbering contract: the target structure is a Boltz prediction OF THE FASTA SEQUENCE, which
# Boltz writes as chain A, residues 1..N in sequence order. So a hotspot is "A<i>" with i the
# 1-based position in the fasta sequence -- NOT a crystal structure's author numbering. The
# coldspot file uses exactly the same format and numbering.
# ---------------------------------------------------------------------------------------

def _expand_hotspot_spec(spec: str, target_name: str, target_length: int,
                         kind: str = "hotspot") -> str:
    """Expand "A20-30,A45" into the flat "A20,A21,...,A30,A45" that parse_hotspots wants
    (it has no range support: it does int(h[1:]) on each comma-separated token).

    `kind` ("hotspot" or "coldspot") only labels the error messages: both go through the same
    parse_hotspots string format."""
    out = []
    for chunk in spec.strip().split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        chain, rest = chunk[0], chunk[1:]
        if chain not in "ABCDEFGHIJKLMNOPQRSTUVW":
            raise ValueError(
                f"{target_name}: {kind} chain '{chain}' in '{chunk}' is not in "
                f"'ABCDEFGHIJKLMNOPQRSTUVW' (parse_hotspots maps the letter through that "
                f"fixed alphabet). Predicted targets are single-chain 'A'.")
        if "-" in rest:
            start, end = [int(v) for v in rest.split("-", 1)]
        else:
            start = end = int(rest)
        if not (1 <= start <= end <= target_length):
            raise ValueError(
                f"{target_name}: {kind} '{chunk}' is out of range for a {target_length}-residue "
                f"target (valid: {chain}1-{chain}{target_length}). {kind.capitalize()} numbering "
                f"is 1-based over the fasta sequence.")
        out.extend(f"{chain}{i}" for i in range(start, end + 1))
    if not out:
        raise ValueError(f"{target_name}: empty {kind} spec")
    return ",".join(out)

def _read_spec_file(path: str, lengths: dict, kind: str, allow_blank: bool) -> dict:
    """Read a `target,<spec>` CSV into {target: expanded spec}. Used for both the hotspot and
    the coldspot file, so they share one format: '#' comments, an optional `target,` header,
    ranges and single residues mixed with commas, entries for targets outside this run ignored.

    allow_blank: a listed target with an empty spec is skipped (coldspots are optional per
    target) instead of raising (a hotspot spec is mandatory)."""
    specs = {}
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("target,"):
                continue
            name, _, spec = line.partition(",")
            name, spec = name.strip(), spec.strip()
            if name not in lengths:
                continue  # an entry for a target not in this run
            if allow_blank and not spec:
                continue
            specs[name] = _expand_hotspot_spec(spec, name, lengths[name], kind=kind)
    return specs

def _compact_spec(spec: str) -> str:
    """Compress an expanded spec back to ranges joined by '+'.

    ScoreCSV does no quoting (files/csv.py:31), so the comma-joined form would corrupt the
    row. "A20,A21,...,A30,A45" -> "A20-30+A45".
    """
    if not spec:
        return "none"
    items = []
    for tok in spec.split(","):
        items.append((tok[0], int(tok[1:])))
    out, i = [], 0
    while i < len(items):
        chain, start = items[i]
        end = start
        while (i + 1 < len(items) and items[i + 1][0] == chain
               and items[i + 1][1] == end + 1):
            i += 1
            end = items[i][1]
        out.append(f"{chain}{start}" if start == end else f"{chain}{start}-{end}")
        i += 1
    return "+".join(out)

hotspot_specs = {}
coldspot_specs = {}
if dock_blueprint:
    lengths = {n: len(s.replace(":", "")) for n, s in targets}
    # Hotspots and coldspots are independent: any of the four combinations is allowed. Without a
    # hotspot file EVERY target residue is eligible (parse_hotspots' "none"), so the docking
    # centre is drawn from the whole surface -- minus the coldspot exclusion zone if there is one.
    # Measured on a 126-residue target this costs ~4 s and ~90 MB, so it is not a resource risk.
    if use_hotspots:
        hotspot_specs = _read_spec_file(opt.hotspot_file, lengths, "hotspot", allow_blank=False)
        missing = [n for n, _ in targets if n not in hotspot_specs]
        assert not missing, (
            f"--hotspot_file {opt.hotspot_file} has no entry for: {missing}. "
            f"Add them, or restrict the run with --only_targets.")
        print(f"hotspots loaded for {len(hotspot_specs)} targets")
    if use_coldspots:
        # Optional per target: unlike hotspots, a target with no row (or a blank one) simply
        # has no coldspots.
        coldspot_specs = _read_spec_file(opt.coldspot_file, lengths, "coldspot", allow_blank=True)
        if use_hotspots:
            for name, cold in coldspot_specs.items():
                # A residue that is both wanted and unwanted is a contradiction in the input, not
                # something to resolve silently -- and it is cheap to catch here, before any GPU
                # time. (Without a hotspot file there is nothing to contradict.)
                both = set(hotspot_specs[name].split(",")) & set(cold.split(","))
                if both:
                    both = sorted(both, key=lambda t: (t[0], int(t[1:])))
                    raise ValueError(
                        f"{name}: residues {_compact_spec(','.join(both))} are listed as both "
                        f"hotspots ({opt.hotspot_file}) and coldspots ({opt.coldspot_file}).")
        print(f"coldspots loaded for {len(coldspot_specs)} of {len(targets)} targets")
    mode = {(True, True): "hotspots + coldspots", (True, False): "hotspots only",
            (False, True): "coldspots only (every non-coldspot residue is eligible)",
            (False, False): "NO hotspots or coldspots: the docking site is a random surface "
                            "patch per attempt"}[(use_hotspots, use_coldspots)]
    print(f"[docking constraints] {mode}")

# ---------------------------------------------------------------------------------------
# per-target MSA cache
#
# mosaic.py rebuilds its re-prediction spec with use_msa=True on every attempt and every
# stage, and use_msa=True queries https://api.colabfold.com. Precomputing one MSA per
# target and attaching it with precomputed_msa() turns O(attempts * stages * targets)
# server calls into exactly one per target, cached on disk across runs.
#
# This block runs before any model weights are loaded so that `--msa_only True` warms the
# cache on a CPU/login node without touching a GPU.
# ---------------------------------------------------------------------------------------

def _seq_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]

def _pdb_to_template_cif(pdb_path: str, cif_path: str) -> None:
    """Convert a plain flexcraft-written PDB into an mmCIF that Boltz's add_template() can
    actually read.

    add_template() on a .pdb goes through gemmi -> a round-tripped mmCIF -> Boltz's own
    parse_mmcif, which requires _entity_poly.pdbx_seq_one_letter_code to be populated.
    DesignData.save_pdb() writes no SEQRES, and gemmi.Structure.setup_entities() alone leaves
    entity.full_sequence empty, so the round-tripped mmCIF has pdbx_seq_one_letter_code='?' and
    boltz/data/parse/mmcif.py:parse_polymer raises `IndexError: list index out of range`
    (verified empirically -- add_template() on a save_pdb() output crashes with this error).
    Setting full_sequence explicitly from the actual residues before writing works around it.

    Assumes a single-chain (or single-entity-per-chain) target, matching how
    _predict_target_pdb below always builds it.
    """
    import gemmi
    structure = gemmi.read_structure(pdb_path)
    structure.setup_entities()
    chain_residues = {ch.name: [r.name for r in ch] for m in structure for ch in m}
    for entity in structure.entities:
        entity.full_sequence = chain_residues[entity.subchains[0][0]]
    structure.assign_label_seq_id()
    structure.make_mmcif_document().write_file(cif_path)

msa_dir = f"{opt.out_path}/msa" if opt.msa_dir == "none" else opt.msa_dir
os.makedirs(msa_dir, exist_ok=True)
target_pdb_dir = (f"{opt.out_path}/target_structures"
                  if opt.target_pdb_dir == "none" else opt.target_pdb_dir)
if use_protenix:
    # The cached target structure is what gets templated / docked against, so Boltz- and
    # Protenix-predicted targets must never share a cache: a Protenix run would otherwise silently
    # pick up Boltz's prediction of the same sequence (same filename) and vice versa, confounding
    # the model comparison. The Joltz path stays flat, so existing caches are untouched. To force
    # both models onto the same target structure, copy the PDB into this subdirectory by hand.
    target_pdb_dir = f"{target_pdb_dir}/{opt.structure_model}"
if dock_blueprint or use_target_template:
    os.makedirs(target_pdb_dir, exist_ok=True)

def _protenix_chain_msa_dirs(path: str, num_chains: int) -> list:
    return [f"{path}/chain_{i}" for i in range(num_chains)]

msa_paths = {}
for name, target_sequence in targets:
    digest = _seq_hash(target_sequence)
    if use_protenix:
        # Protenix takes an MSA as a directory of a3m files per chain, not Boltz's npz feature
        # dump. One directory per target chain: {path}/chain_{i}/{non_pairing,pairing}.a3m. The
        # a3m files do not depend on the model, so v1 and v2 share this cache.
        path = f"{msa_dir}/protenix/{name}_{digest}"
        chains = target_sequence.split(":")
        chain_dirs = _protenix_chain_msa_dirs(path, len(chains))
        if all(os.path.isfile(f"{d}/non_pairing.a3m") for d in chain_dirs):
            print(f"[msa] cached: {path}")
        else:
            print(f"[msa] querying colabfold (protenix a3m): {name} ({len(target_sequence)} aa)")
            if len(chains) > 1:
                print(f"[msa] WARNING {name}: {len(chains)} chains -- the cached per-chain MSAs "
                      f"carry no cross-chain pairing, unlike Boltz's joint search.")
            tmp = f"{path}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            # _run_msa_search dedups AND sorts its input, so its result directories follow
            # sorted(set(chains)), not the chain order -- map back the way ProtenixSpec's own
            # to_features does. A failed search is only printed inside it (no exception), so
            # fewer directories than sequences means a chain has no MSA: fail loudly here rather
            # than let zip() silently misalign chains and directories.
            unique = sorted(set(chains))
            found = _run_msa_search(chains, tmp, None)
            assert len(found) == len(unique) and all(
                os.path.isfile(f"{d}/non_pairing.a3m") for d in found), (
                f"{name}: colabfold search returned {len(found)} usable MSA directories for "
                f"{len(unique)} unique chain sequences. See the MMSEQS2 error printed above.")
            by_sequence = dict(zip(unique, found))
            shutil.rmtree(path, ignore_errors=True)
            for chain, dst in zip(chains, chain_dirs):
                shutil.copytree(by_sequence[chain], dst)
            shutil.rmtree(tmp, ignore_errors=True)
            time.sleep(opt.msa_sleep)
        msa_paths[name] = path
        continue
    path = f"{msa_dir}/{name}_{digest}.npz"
    if not os.path.isfile(path):
        print(f"[msa] querying colabfold: {name} ({len(target_sequence)} aa)")
        # Target chains only -- no binder chain. precomputed_msa(..., start_chain=1)
        # below re-attaches this block at token offset binder_length.
        (JoltzSpec()
            .add_protein(*target_sequence.split(":"), use_msa=True)
            .save_msa(path, cache=opt.boltz_path))
        time.sleep(opt.msa_sleep)
    else:
        print(f"[msa] cached: {path}")
    msa_paths[name] = path

if opt.msa_only == "True":
    if use_protenix:
        # Keep ALL network use in this stage: fetch the checkpoint and the shared CCD data now
        # (download only, no model load), so a firewalled compute node fails here, before any
        # GPU time is spent, rather than at first use.
        print(f"[protenix] prefetching {PROTENIX_CHECKPOINTS[opt.structure_model]} + CCD data")
        download_data()
        _resolve_model_path(PROTENIX_CHECKPOINTS[opt.structure_model])
    print("--msa_only True: MSAs cached, exiting before model load.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------------------
# models (all target-independent: binder length is fixed across the sweep)
# ---------------------------------------------------------------------------------------

pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_path, eps=0.05, split_params=True)
jit_pmpnn = jax.jit(pmpnn)
params = dict(mpnn=pmpnn_params)

sampler = None
if sample_structure:
    import salad.inference as si
    from copy import deepcopy

    def cloud_std_default(num_aa):
        minval = num_aa ** 0.4
        return minval + np.random.rand() * opt.std_buffer

    def model_step(config):
        config = deepcopy(config)
        config.eval = True
        @hk.transform
        def step(data, prev):
            noise = si.StructureDiffusionNoise(config)
            predict = si.StructureDiffusionPredict(config)
            data.update(noise(data))
            out, prev = predict(data, prev)
            return out, prev
        return step.apply

    salad_config, salad_params = si.make_salad_model("default_vp_scaled", opt.salad_path)
    num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(f"{opt.length}")
    salad_data, init_prev = si.data.from_config(
        salad_config,
        num_aa=num_aa,
        residue_index=resi,
        chain_index=chain,
        cyclic_mask=cyclic_mask)
    salad_step = model_step(salad_config)
    sampler = si.Sampler(salad_step, out_steps=400)

salad_ve_config = salad_ve_params = None
if dock_blueprint:
    import salad.inference as si
    from copy import deepcopy
    from salad.modules.utils.geometry import index_align
    from flexcraft.pipelines.utils import bias_dssp
    from flexcraft.pipelines.experimental._docked_salad import (
        binder_step, parse_target, parse_hotspots, sample_centers_spaced, config_from_opt)

    # NOTE: make_salad_model returns a SHARED, MUTABLE module-level config object (no copy).
    # parse_target writes c.target_size onto it and si.data.from_config writes c.cyclic, so
    # every target must get its own deepcopy before either is called.
    salad_ve_config, salad_ve_params = si.make_salad_model(
        "default_ve_scaled", opt.salad_ve_path)

def _make_protenix_predictor(protenix_model, num_recycle=4, sampling_steps=25):
    """Full-atom predictor for Protenix: spec -> prediction, the counterpart of Joltz2.predictor().

    Built here rather than taken from Protenix.predictor(), which cannot currently be called: it
    reads self.cache (never set on Protenix) and passes num_sampling_steps / deterministic to
    ProtenixEvaluator, which has neither field. Only pieces that work are used: the evaluator's
    _predict, ProtenixSpec.to_input() (which takes no pad/cache arguments) and
    ProtenixPrediction. Unlike Protenix.evaluator(), the sampler is NOT overridden, so it runs
    the checkpoint's own diffusion settings (gamma0 0.8, eta 1.5), as Joltz2's predictor does.
    """
    evaluator = ProtenixEvaluator(
        model=protenix_model.model, num_recycle=num_recycle, sampling_steps=sampling_steps)
    jit_predict = eqx.filter_jit(evaluator._predict)
    def _predict(key, spec):
        protenix_input, writer = spec.to_input()
        prediction = jit_predict(key, protenix_input.features, num_samples=1)
        return ProtenixPrediction(data=prediction.data, writer=writer)
    return _predict

# Backend-neutral handles used everywhere below:
#   structure_pred(key, spec) -> prediction   (.result, .save_pdb)   full-atom predictor
#   structure_eval(params)    -> evaluator    (.predict(key, input, num_samples) -> result)
#   params["structure"]                        evaluator parameters, differentiated through in loss()
# 25 sampling steps and 4 recycles for both backends, exactly the values Joltz always used.
if use_protenix:
    model = Protenix(model=PROTENIX_CHECKPOINTS[opt.structure_model])
    structure_pred = _make_protenix_predictor(model, num_recycle=4, sampling_steps=25)
    structure_eval, structure_eval_params = model.evaluator(sampling_steps=25, num_recycle=4)
else:
    model = Joltz2(cache=opt.boltz_path)
    structure_pred = model.predictor(num_recycle=4)
    structure_eval, structure_eval_params = model.evaluator(num_recycle=4)
params["structure"] = structure_eval_params

# ---------------------------------------------------------------------------------------
# metrics (mosaic.py:209-280, verbatim apart from dropping the off-target path)
# ---------------------------------------------------------------------------------------

def batch_mpnn(mpnn):
    def _mpnn_map(params, key, data):
        if len(data["atom_positions"].shape) == 4:
            num_samples = data["atom_positions"].shape[0]
            # replicate all un-batched inputs
            xx = {
                key: jnp.repeat(data[key][None, :], num_samples, axis=0)
                for key in ["atom_mask", "mask", "aa", "residue_index", "chain_index", "batch_index"]
            }
            # add in the batched atom positions
            xx["atom_positions"] = data["atom_positions"]
            # split keys across batch
            keys = jax.random.split(key, num_samples)
            # vmap the ProteinMPNN
            return jax.vmap(mpnn, (None, 0, 0), 0)(params, keys, xx)
        else:
            return mpnn(params, key, data)
    return _mpnn_map

def metrics(sequence, key, result: AF3LikeResult):
    res_data = result.to_data(return_samples=True)
    binder_data = res_data[:binder_length]
    dssp = binder_data.p_dssp
    pmpnn_input = res_data.update(aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    is_binder = jnp.zeros((pmpnn_input.aa.shape[0],), dtype=jnp.bool_).at[:binder_length].set(True)
    logits = batch_mpnn(pmpnn)(params["mpnn"], key, pmpnn_input.drop_aa(where=is_binder))["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not result.is_single_sample:
        logits = logits.mean(axis=0)
    logits = logits[:binder_length]
    center = logits.mean(axis=0)
    logits = logits - center
    prob = jax.lax.stop_gradient(jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    entropy = -(jax.nn.log_softmax(logits, axis=-1) * prob).sum(axis=-1).mean()
    # losses
    plddt = result.plddt
    pae = 32 * result.pae # unnormalized pAE
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)
    binder_selector = result.chain_index == 0
    hotspot_selector = result.chain_index != 0
    if hotspot_index is not None:
        hotspot_selector = (np.arange(result.residue_index.shape[0])[:, None] == hotspot_index).any(axis=-1)
    hotspot_contacts = (
        result.index_contact_score(
            binder_selector, hotspot_selector, # NOTE: flip target and binder here
            num_contacts=3, contact_distance=20.0)
      + result.index_contact_score(
            hotspot_selector, binder_selector, # NOTE: flip target and binder here
            num_contacts=3, contact_distance=20.0)
    ) / 2
    binder_target_contacts = result.chain_contact_score(1, 0, num_contacts=3, contact_distance=20.0)
    if hotspot_index is not None:
        binder_target_contacts = hotspot_contacts
    return dict(
        logits = logits,
        entropy = entropy,
        plddt = plddt[:binder_length].mean(),
        recovery = (sequence * prob).sum(axis=-1).mean(),
        binder_target_contacts = binder_target_contacts,
        within_binder_contacts = result.chain_contact_score(0, 0, num_contacts=25, min_resi_distance=10),
        hotspot_contacts = hotspot_contacts,
        within_binder_pae = pae[:binder_length, :binder_length].mean(),
        binder_target_pae = pae[:binder_length, binder_length:].mean(),
        target_binder_pae = pae[binder_length:, :binder_length].mean(),
        iptm = result.index_iptm(chain_index=is_binder),
        eptm = result.index_ptm_score(chain_index=is_binder),
        ipsae = result.ipsae(chain_index=is_binder),
        average_num_nonzero = (sequence > 0.01).sum(axis=-1).mean(),
        L = dssp["L"], H = dssp["H"], E = dssp["E"]
    )

def _passing(aux):
    return (
        aux["ipsae"] > 0.75 and aux["plddt"] > 0.8
        and aux["binder_target_pae"] < 5.0 and aux["target_binder_pae"] < 5.0
        and aux["recovery"] > 0.6
    )

# ---------------------------------------------------------------------------------------
# blueprint comparison (landscape.py:246-262, adapted for a binder inside a complex)
# ---------------------------------------------------------------------------------------

def dssp_to_string(dssp_array) -> str:
    """Convert an int DSSP array (0=loop, 1=helix, 2=strand) to an 'LHE' string."""
    code = np.array(list("LHE"))
    return "".join(code[np.asarray(dssp_array)])

def _structure_comparison(design: DesignData, predicted_complex: DesignData) -> dict:
    """Compare the salad blueprint against the binder as predicted inside the complex.

    Two ordering hazards, both handled by slicing the binder out FIRST:

    - RMSD requires equal residue counts. It derives every mask from `x.shape[:1]`
      (structure/metrics/rmsd.py:53-63), so an 80-residue blueprint against a
      binder+target complex would align garbage. Atom width may differ freely -- the
      blueprint is atom14 from salad, the prediction atom24, and RMSD indexes column 1
      (CA) out of the fixed ["N","CA","C","O","CB"] order in both.
    - assign_dssp() writes into self.data["dssp"] and the .dssp property returns that
      cache (data/data.py:158-181), while index() copies every field including a cached
      "dssp" (data.py:267-289). AF3LikeResult.to_data() (Boltz and Protenix) also zeroes batch_index across both
      chains (boltz/_result.py:357), so DSSP assigned on the complex would see
      binder<->target backbone H-bonds and report target-induced strands.

    Note: `dssp_predicted` comes from to_data(), which picks the single best-pAE sample,
    while metrics()'s L/H/E are averaged over samples. They will not agree exactly.
    """
    assert "dssp" not in predicted_complex.data, \
        "DSSP already cached on the complex -- it would be sliced into the binder"
    binder = predicted_complex[:binder_length]
    binder_dssp = np.asarray(binder.dssp)
    salad_dssp = np.asarray(design.dssp)
    assert binder_dssp.shape == salad_dssp.shape, \
        f"blueprint/binder length mismatch: {salad_dssp.shape} vs {binder_dssp.shape}"
    return dict(
        rmsd_to_blueprint=RMSD(atoms=["CA"])(binder, design),
        dssp_salad=dssp_to_string(salad_dssp),
        dssp_predicted=dssp_to_string(binder_dssp),
        dssp_identity=np.float32((salad_dssp == binder_dssp).mean()),
    )

# ---------------------------------------------------------------------------------------
# placement comparison (docked mode): did the hallucinated binder land on the epitope the
# blueprint was docked to?
#
# Boltz never sees the blueprint's coordinates. The only channel from pose to hallucination is
# the ProteinMPNN sequence, so these columns measure exactly "does a sequence designed for a
# specific docked pose reproduce that pose?".
# ---------------------------------------------------------------------------------------

def epitope_mask(complex_data: DesignData, cutoff: float) -> np.ndarray:
    """Per-target-residue bool: is this residue within `cutoff` of any binder residue?

    Uses pseudo-CB (DesignData.cb -> positions_to_ncacocb, reconstructed from N/CA/C) rather
    than real CB, because salad's and Boltz's sidechains are independently generated and would
    inject noise. Boltz's own contact atom (boltz/_utils.py:74) is the same pseudo-CB.
    Expects a BINDER-FIRST complex. Frame-invariant: computed within one structure, so it
    needs no superposition.
    """
    cb = np.asarray(complex_data.cb)
    d = np.linalg.norm(cb[binder_length:, None] - cb[None, :binder_length], axis=-1)
    return d.min(axis=1) < cutoff

def _epitope_agreement(bp_mask: np.ndarray, pred_mask: np.ndarray, suffix="") -> dict:
    """Set agreement between two target-residue epitope masks.

    Directions are pinned down in the names: recall is the fraction of the BLUEPRINT epitope
    that the prediction recovered; precision is the fraction of the PREDICTED epitope that was
    in the blueprint. Empty masks give 0.0 rather than NaN -- `_report_result` would otherwise
    write "nan" -- and the n_* columns tell you which kind of zero it is.
    """
    inter = float((bp_mask & pred_mask).sum())
    union = float((bp_mask | pred_mask).sum())
    n_bp, n_pred = float(bp_mask.sum()), float(pred_mask.sum())
    return {
        f"epitope_jaccard{suffix}": np.float32(inter / union if union > 0 else 0.0),
        f"epitope_recall_vs_blueprint{suffix}": np.float32(inter / n_bp if n_bp > 0 else 0.0),
        f"epitope_precision_vs_blueprint{suffix}": np.float32(inter / n_pred if n_pred > 0 else 0.0),
        f"n_epitope_blueprint{suffix}": np.float32(n_bp),
        f"n_epitope_predicted{suffix}": np.float32(n_pred),
    }

def _placement_comparison(bp_complex: DesignData, predicted_complex: DesignData,
                          hotspot_mask: np.ndarray, pos_center, target_size: int,
                          coldspot_mask: np.ndarray = None) -> dict:
    """Compare where the blueprint put the binder against where the hallucination put it.

    `bp_complex` must already be BINDER-FIRST and in the blueprint's target frame; see the
    caller for the two fixes that guarantee that (chain reindexing and the atom_pos re-align).
    `coldspot_mask` (per target residue, or None) adds the coldspot_hit_* columns.
    """
    n = bp_complex.size
    assert predicted_complex.size == n, \
        f"blueprint/prediction size mismatch: {n} vs {predicted_complex.size}"
    tgt = np.arange(n) >= binder_length
    w = tgt.astype(np.float32)

    out = {}
    for cutoff, suffix in ((opt.epitope_cutoff, ""), (opt.epitope_cutoff_2, "_10a")):
        out.update(_epitope_agreement(
            epitope_mask(bp_complex, cutoff),
            epitope_mask(predicted_complex, cutoff), suffix=suffix))

    # fraction of the REQUESTED hotspot residues the hallucinated binder actually contacts.
    # hotspot_mask is positional over target residues, so it maps to complex rows by offset.
    pred_ep = epitope_mask(predicted_complex, opt.epitope_cutoff)
    if hotspot_mask is not None and hotspot_mask.any():
        out["hotspot_recall"] = np.float32((pred_ep & hotspot_mask).sum() / hotspot_mask.sum())

    # fraction of the requested COLDSPOT residues the binder touches -- lower is better. Computed
    # for the blueprint as well, because the coldspot logic only steers where salad's docking
    # centre is drawn, so the blueprint can still reach a coldspot and this is what shows it.
    if coldspot_mask is not None and coldspot_mask.any():
        bp_ep = epitope_mask(bp_complex, opt.epitope_cutoff)
        n_cold = coldspot_mask.sum()
        out["coldspot_hit_blueprint"] = np.float32((bp_ep & coldspot_mask).sum() / n_cold)
        out["coldspot_hit_predicted"] = np.float32((pred_ep & coldspot_mask).sum() / n_cold)

    # Target-frame superposition: weight=w makes index_kabsch use ONLY target rows for both
    # the centroid and the covariance, so this is an exact target-only fit that still moves
    # every atom. weight is NOT broadcast over atoms in RMSD.__call__ (rmsd.py:59-61), so
    # atoms=["CA"] is mandatory here.
    out["binder_rmsd_target_frame"] = RMSD(atoms=["CA"])(
        predicted_complex, bp_complex, weight=w, eval_mask=~tgt)
    # sanity: how well the two targets agree at all. A large value means the correspondence
    # is broken, not that the binder moved.
    out["target_fit_rmsd"] = RMSD(atoms=["CA"])(
        predicted_complex, bp_complex, weight=w, eval_mask=tgt)

    # binder centroid displacement, after putting the prediction in the blueprint's frame
    aligned = predicted_complex.align_to(bp_complex, weight=w)
    pred_c = np.asarray(aligned.ca)[:binder_length].mean(axis=0)
    bp_c = np.asarray(bp_complex.ca)[:binder_length].mean(axis=0)
    out["binder_centroid_drift"] = np.float32(np.linalg.norm(pred_c - bp_c))
    # did salad honour the requested docking centre? Both are in the blueprint's target frame.
    out["blueprint_center_offset"] = np.float32(
        np.linalg.norm(bp_c - np.asarray(pos_center)))
    return out

# ---------------------------------------------------------------------------------------
# reporting (landscape.py:303-325, plus hunter.py's leading `target` column)
# ---------------------------------------------------------------------------------------

def _report_trajectory(target: str, index: int, writer: ScoreCSV, start_step=0, weight_tag=None):
    def _report(step: int, loss: float, aux: dict):
        line_info = dict(
            target=target, attempt=index, step=start_step + step,
            total=float(loss.mean()))
        if weight_tag is not None:
            line_info["distogram_weight"] = weight_tag
        for k in writer.keys:
            if k in aux and not isinstance(aux[k], str):
                line_info[k] = float(aux[k].mean())
        writer.write_line(line_info)
    return _report

def _report_result(target: str, attempt: int, stage: str, key,
                   writer: ScoreCSV, sequence, result: AF3LikeResult, aux=None):
    """Write one row. Must tolerate string values (the DSSP columns) -- mosaic's version
    calls float(aux[k].mean()) unconditionally and would raise on them."""
    if aux is None:
        aux = metrics(sequence, key, result)
    line_info = dict(
        target=target, attempt=attempt, stage=stage,
        sequence=aas.decode(jnp.argmax(sequence, axis=-1), aas.AF2_CODE))
    for k in writer.keys:
        if k in aux:
            value = aux[k]
            if isinstance(value, str):
                line_info[k] = value
            else:
                line_info[k] = float(value.mean())
    writer.write_line(line_info)
    return aux

# ---------------------------------------------------------------------------------------
# hallucination loss (mosaic.py:344-393, no off-target term). Backend-neutral: the same body
# serves Boltz and Protenix, whose evaluators share predict(key, input, num_samples=...) and
# whose results share the AF3LikeResult metric surface.
# ---------------------------------------------------------------------------------------

def loss(sequence, key=None, context=None, params=None):
    keys = jax.random.split(key, 4)
    # forbid cysteines
    sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
    sequence = sequence / sequence.sum(axis=1, keepdims=True)
    # set sequence
    structure_input = params["structure_input"].set_aa(sequence)
    # predict structure
    result: AF3LikeResult = structure_eval(params["structure"]).predict(
        keys[0], structure_input, num_samples=4)
    # compute losses
    out = metrics(sequence, keys[1], result)
    out["result"] = result

    # combination
    value = (
          out["binder_target_contacts"]
        + out["within_binder_contacts"]
        - 10.0 * out["recovery"]
        + 0.05 * out["target_binder_pae"]
        + 0.05 * out["binder_target_pae"]
        + 0.4 * out["within_binder_pae"]
        - 0.1 * out["plddt"]
        - 0.025 * out["iptm"]
        - 0.025 * out["eptm"]
    )
    if use_distogram_loss:
        # Read the target from the TRACED `params` argument, never from the module-level dict:
        # metrics() and friends close over the global `params`, and anything read that way is
        # frozen into the jitted loss at first trace, so every later attempt would silently
        # keep optimizing against the FIRST blueprint's distogram.
        assert result.log_distogram.shape == params["target_distogram"].shape, (
            f"predicted distogram {result.log_distogram.shape} vs blueprint distogram "
            f"{params['target_distogram'].shape}: token order/count differs from the blueprint "
            f"complex")
        out["distogram_nll"] = distogram_nll(result, params["target_distogram"])
        # The weight is a traced scalar in `params` too (set per run in _hallucinate_once), not
        # opt.distogram_loss_weight: a Python float closed over here would be frozen into the
        # jitted loss, and changing it per run would silently do nothing (or force a recompile).
        value = value + params["distogram_loss_weight"] * out["distogram_nll"]
    out["total"] = value
    return value, out
loss_update = jit_loss_update(loss)

# ---------------------------------------------------------------------------------------
# output files -- created ONCE, before both loops.
# ScoreCSV.__init__ opens "wt" and rewrites the header (files/csv.py:15) unless append=True
# and the file already exists, so constructing one inside the target loop would (still)
# discard every previous target's rows -- append mode only changes the ONE construction here.
# ---------------------------------------------------------------------------------------

for subdir in ("blueprints", "attempts", "success"):
    os.makedirs(f"{opt.out_path}/{subdir}/", exist_ok=True)

# --resume: an attempt is complete once both its stage_1 and stage_2 rows are in
# results.csv. Read that BEFORE constructing the (possibly append=True) writers below, since
# ScoreCSV(append=True) leaves an existing file untouched -- we need its old contents, not
# what we're about to (not) write to it.
completed_attempts = set()
if resume:
    results_path = f"{opt.out_path}/results.csv"
    if os.path.isfile(results_path):
        import csv as _csv
        stages_seen = {}
        with open(results_path, "rt", newline="") as f:
            for row in _csv.DictReader(f):
                # Refuse to continue a run made with a different structure model. The header
                # check in ScoreCSV(append=True) catches joltz <-> protenix (different schema)
                # but not protenix_v1 <-> protenix_v2, whose columns are identical -- that would
                # silently mix two models' rows in one CSV. Files without the column are Joltz.
                previous_model = row.get("structure_model", "joltz")
                if previous_model != opt.structure_model:
                    raise ValueError(
                        f"--resume: {results_path} was written with structure_model="
                        f"'{previous_model}' but this run uses '{opt.structure_model}'. Use the "
                        f"same --structure_model, or a fresh --out_path.")
                # `distogram_weight` only exists in sweep runs (None otherwise): a run is one
                # (target, attempt, weight), so a partly finished sweep resumes at the missing weights.
                stages_seen.setdefault(
                    (row["target"], int(row["attempt"]), row.get("distogram_weight")),
                    set()).add(row["stage"])
        completed_attempts = {k for k, stages in stages_seen.items()
                              if {"stage_1", "stage_2"} <= stages}
        print(f"[resume] {results_path}: {len(completed_attempts)} runs already complete")
    else:
        print(f"[resume] no existing {results_path} -- starting fresh")

common_keys = [
    "plddt",
    "recovery",
    "binder_target_contacts",
    "hotspot_contacts",
    "within_binder_contacts",
    "within_binder_pae",
    "binder_target_pae",
    "target_binder_pae",
    "iptm",
    "eptm",
    "ipsae",
    "average_num_nonzero",
    "entropy",
    "L", "H", "E",
]
if use_distogram_loss:
    # Conditional schema, like the docked columns below: runs without the loss keep the
    # original columns, so existing runs and --resume on them are unaffected.
    common_keys.append("distogram_nll")
blueprint_keys = ["rmsd_to_blueprint", "dssp_identity"]
# Docked-mode placement columns. `rmsd_to_blueprint` (above) is a BINDER-ONLY superposition and
# measures fold agreement; `binder_rmsd_target_frame` is a TARGET-frame superposition and
# measures docking-site agreement. They are different quantities -- do not conflate them.
placement_keys = ([
    "epitope_jaccard", "epitope_recall_vs_blueprint", "epitope_precision_vs_blueprint",
    "n_epitope_blueprint", "n_epitope_predicted",
    "epitope_jaccard_10a", "epitope_recall_vs_blueprint_10a",
    "epitope_precision_vs_blueprint_10a",
    "n_epitope_blueprint_10a", "n_epitope_predicted_10a",
    "hotspot_recall", "binder_rmsd_target_frame", "target_fit_rmsd",
    "binder_centroid_drift", "blueprint_center_offset",
] if dock_blueprint else [])
# Coldspot columns, only with --coldspot_file (conditional schema, like the docked columns above:
# existing docked runs keep their columns and --resume on them is unaffected). Each is the
# FRACTION OF THE TARGET'S COLDSPOT RESIDUES the binder contacts (within --epitope_cutoff), so
# LOWER is better and 0 means fully avoided -- the opposite direction to hotspot_recall. The
# blueprint one tells you whether salad obeyed (the coldspot logic only steers the docking
# centre); the predicted one whether the hallucinated binder stayed away.
coldspot_keys = (["coldspot_hit_blueprint", "coldspot_hit_predicted"] if use_coldspots else [])
string_keys = ["dssp_requested", "dssp_salad", "dssp_predicted"]
if dock_blueprint:
    string_keys = (["blueprint_mode", "hotspot_spec"]
                   + (["coldspot_spec"] if use_coldspots else []) + string_keys)
if use_protenix:
    # Provenance for merged / copied results. Joltz runs deliberately keep the original 27-column
    # schema (same conditional-schema precedent as the docked columns above), so existing runs
    # and --resume on them are unaffected.
    string_keys = ["structure_model"] + string_keys

weight_keys = ["distogram_weight"] if sweeping_weights else []
trajectory = ScoreCSV(
    f"{opt.out_path}/trajectory.csv",
    keys=["target", "attempt"] + weight_keys + ["step", "total"] + common_keys, append=resume)
results_keys = (
    ["target", "attempt", "stage"] + weight_keys + ["target_length", "passed"]
    + common_keys + blueprint_keys + placement_keys + coldspot_keys
    + ["sequence"] + string_keys)
results = ScoreCSV(f"{opt.out_path}/results.csv", keys=results_keys, append=resume)
success = ScoreCSV(f"{opt.out_path}/success.csv", keys=results_keys, append=resume)

# The shuffled-pair null: epitope agreement between the blueprint of attempt i and the
# prediction of attempt j (i != j) for the same target. Without it a Jaccard of 0.3 is
# meaningless. Written once per target, after all its attempts finish. `center_distance` is
# recorded so the null can be conditioned on how far apart the two docking sites were --
# sample_centers_spaced is hotspot-biased, so different attempts often land on the same patch
# and an unconditioned null would be inflated.
placement_null = None
if dock_blueprint:
    placement_null = ScoreCSV(
        f"{opt.out_path}/placement_null.csv",
        keys=["target"] + weight_keys + ["blueprint_attempt", "prediction_attempt", "stage",
              "center_distance", "epitope_jaccard", "epitope_recall_vs_blueprint",
              "n_epitope_blueprint", "n_epitope_predicted"], append=resume)
skipped = ScoreCSV(f"{opt.out_path}/skipped.csv", keys=["target", "reason"], append=resume)

# ---------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------

key = None  # rebound per target

# Populated per target when needed (dock_blueprint and/or target_use_template): name -> path
# to that target's cached standalone-prediction PDB. Shared by the docked-blueprint setup
# (parse_target reads it) and by --target_use_template (converted to a template CIF below).
target_pdbs = {}
# name -> template CIF path (Joltz only), only populated when --target_use_template True.
# Protenix reads the PDB directly and never fills this. Whether the target enters the complex
# as a template or as an MSA is decided by use_target_template in _complex_spec, so a target
# predicted only for docking (template off) still uses MSA as before.
target_template_paths = {}

def _protenix_attach_target_msa(spec, name: str, first_chain: int, num_chains: int):
    """Point the target chains of a ProtenixSpec at their cached a3m directories.

    ProtenixSpec ignores precomputed_msa(); its to_features honours a per-chain "msa_dir".
    use_msa=False keeps it from launching its own colabfold search. Chains without an msa_dir
    (the binder) get ProtenixSpec's dummy single-sequence MSA, which is what we want.
    """
    dirs = _protenix_chain_msa_dirs(msa_paths[name], num_chains)
    for chain, msa in zip(spec.chains[first_chain:first_chain + num_chains], dirs):
        chain["msa_dir"] = msa
        chain["use_msa"] = False
    return spec

def _predict_target_pdb(name: str, target_sequence: str) -> str:
    """Predict the target's standalone structure once, cached by content hash. Needed by
    --dock_blueprint (parse_target reads it) and/or --target_use_template (converted to a
    template CIF below) -- called at most once per target regardless of which flags are set.
    """
    if name in target_pdbs:
        return target_pdbs[name]
    path = f"{target_pdb_dir}/{name}_{_seq_hash(target_sequence)}.pdb"
    if not os.path.isfile(path):
        print(f"[target] predicting structure for {name}")
        target_chains = target_sequence.split(":")
        if use_protenix:
            spec = _protenix_attach_target_msa(
                ProtenixSpec().add_protein(*target_chains), name, 0, len(target_chains))
        else:
            spec = (JoltzSpec()
                    .add_protein(*target_chains)
                    .precomputed_msa(msa_paths[name], start_chain=0))
        pred = structure_pred(key(), spec)
        # save_pdb, NOT save_cif: save_cif (boltz/_result.py:378) indexes
        # samples[sample_index] unconditionally, but predictor output is single-sample
        # (ndim 3), so it would slice the wrong axis.
        pred.save_pdb(path)
        del pred          # release the writer's TemporaryDirectory handle
    else:
        print(f"[target] cached structure: {path}")
    target_pdbs[name] = path
    return path

def _target_template_cif(name: str, target_sequence: str) -> str:
    """Cached template CIF for a target, built from its standalone-prediction PDB."""
    if name in target_template_paths:
        return target_template_paths[name]
    pdb_path = _predict_target_pdb(name, target_sequence)
    cif_path = f"{target_pdb_dir}/{name}_{_seq_hash(target_sequence)}_template.cif"
    if not os.path.isfile(cif_path):
        _pdb_to_template_cif(pdb_path, cif_path)
    target_template_paths[name] = cif_path
    return cif_path

def _complex_spec(binder_sequence: str, name: str, target_sequence: str):
    """Binder chain first, then the target chain(s), with the target supplied as a template
    (--target_use_template) or as its cached MSA. Used both for the differentiable input
    (binder_sequence = "X" * binder_length) and for the full-atom re-predictions (the real
    hardened binder), so the two can never drift apart.

    A fresh spec every call -- specs mutate themselves (add_* append to self.chains, and
    precomputed_msa / to_input flip chain["use_msa"] in place).
    """
    target_chains = target_sequence.split(":")
    if use_protenix:
        spec = ProtenixSpec().add_protein(binder_sequence).add_protein(*target_chains)
        if use_target_template:
            # UNLIKE Boltz, to_chains names chains of the COMPLEX, not of the template file:
            # ProtenixSpec's _load_template writes the template residues, positionally, onto
            # exactly these complex chains and needs their lengths to match. The binder is
            # chain A, so the target chain(s) are B, C, ... Passing Boltz's ["A"] here would
            # template the binder and fail on the length mismatch.
            complex_letters = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(target_chains)]
            return spec.add_template(
                _predict_target_pdb(name, target_sequence), to_chains=list(complex_letters))
        return _protenix_attach_target_msa(spec, name, 1, len(target_chains))

    spec = JoltzSpec().add_protein(binder_sequence).add_protein(*target_chains)
    if use_target_template:
        # to_chains names chains WITHIN the template file, not the complex -- it is currently
        # a no-op in flexcraft/structure/boltz/_data.py (build_template_yaml writes the YAML
        # key "template_chain_id", but boltz/data/parse/schema.py reads "template_id"), but is
        # harmless to pass and documents intent. Correctness does not depend on it here: the
        # template file (_predict_target_pdb's output) contains ONLY the target chain, and
        # Boltz assigns templates to complex chains via a global sequence-alignment matching
        # (linear_sum_assignment over all complex chains x all template chains), which always
        # picks the target -- its sequence is literally identical to the template's, so it
        # dominates the assignment regardless of what the binder chain looks like. Verified
        # empirically: with a target-only template, 76/76 target tokens get template
        # coordinates and 0/80 binder tokens do, for both a random and an all-X binder chain.
        return spec.add_template(
            _target_template_cif(name, target_sequence), to_chains=["A"])
    # No use_msa=True here: the precomputed MSA covers the target, so this makes no network call.
    return spec.precomputed_msa(msa_paths[name], start_chain=1)

def _to_input(spec):
    """Differentiable model input for a spec. Boltz pads/caches through to_input's arguments
    (pad=False keeps the binder/target slices in metrics() exact); ProtenixSpec.to_input takes
    none and does not pad."""
    if use_protenix:
        return spec.to_input()[0]
    return spec.to_input(pad=False, cache=opt.boltz_path)[0]

def _repredict(result: AF3LikeResult, name: str, target_sequence: str):
    """Full-atom re-prediction of the hardened binder against its target."""
    binder_sequence = result.to_data().to_sequence_string()[:binder_length]
    return structure_pred(key(), _complex_spec(binder_sequence, name, target_sequence))

def _report_stage(name, attempt, stage, sequence, prediction, design,
                  dssp_requested, target_length,
                  bp_complex=None, hotspot_mask=None, pos_center=None,
                  target_size=None, hotspot_spec=None,
                  coldspot_mask=None, coldspot_spec=None, weight_tag=None):
    """metrics() + blueprint comparison (+ placement comparison in docked mode)
    -> one results.csv row. Returns the aux dict."""
    result = prediction.result
    aux = metrics(sequence, key(), result)
    if design is not None:
        aux.update(_structure_comparison(design, result.to_data()))
        aux["dssp_requested"] = dssp_requested
    if bp_complex is not None:
        # a fresh to_data() each time: _structure_comparison's guard asserts no cached DSSP
        aux.update(_placement_comparison(
            bp_complex, result.to_data(), hotspot_mask, pos_center, target_size,
            coldspot_mask=coldspot_mask))
        aux["blueprint_mode"] = "docked"
        aux["hotspot_spec"] = _compact_spec(hotspot_spec)
        if use_coldspots:
            # "none" for a target without coldspots (its coldspot_hit_* cells stay "none" too)
            aux["coldspot_spec"] = _compact_spec(coldspot_spec)
    elif dock_blueprint:
        aux["blueprint_mode"] = "free"
    if use_distogram_loss:
        aux["distogram_nll"] = np.float32(
            distogram_nll(result, params["target_distogram"]))
    if weight_tag is not None:
        aux["distogram_weight"] = weight_tag
    if use_protenix:
        aux["structure_model"] = opt.structure_model
    aux["target_length"] = str(target_length)
    aux["passed"] = "True" if _passing(aux) else "False"
    return _report_result(name, attempt, stage, key(), results,
                          sequence, result, aux=aux)

def _attempt_seed(name: str, attempt: int, tag: str = "") -> int:
    """Deterministic seed for one (target, attempt), independent of run order.

    Used to reseed both the JAX Keygen and the global numpy RNG at the top of every attempt
    (see the attempt loop below) -- not just once per target. A single continuously-advancing
    per-target stream would make attempt N's draws depend on every attempt 0..N-1 having
    actually run first in this process: skip attempts 0..11 under --resume and attempt 12
    would silently replay attempt 0's exact draws (same Keygen state, same numpy state),
    duplicating an already-completed design under a new attempt index. Reseeding from
    (opt.seed, name, attempt) alone makes every attempt reproducible in isolation, so skipping
    already-completed ones changes nothing about what the remaining ones produce.

    `tag` derives a further independent stream for the same (target, attempt), used to start every
    weight of a --distogram_loss_weights sweep from the same key. The default "" leaves the
    original digest, and so every existing seed, unchanged.
    """
    digest = hashlib.sha256(
        (f"{opt.seed}:{name}:{attempt}" + (f":{tag}" if tag else "")).encode()).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)

def _hallucinate_once(name, attempt, target_sequence, target_length, sequence, design,
                      dssp_requested, bp_complex, pos_center, docked, null_rows,
                      weight=None, weight_tag=None):
    """One two-stage hallucination (stage 1 soft APGM, stage 2 logit hardening) from the initial
    `sequence`, with all reporting. Called once per weight of a --distogram_loss_weights sweep,
    always with the same blueprint (`design`, `bp_complex`, params["target_distogram"]) and the
    same `sequence`: the arrays are immutable and the stages only rebind local names, so runs
    cannot leak into each other. Reads the module-level `key`, which the caller rebinds per run.
    """
    file_tag = "" if weight_tag is None else f"_w{weight_tag}"
    context = dict(recovery=10.0, ipae=0.05)
    if use_distogram_loss:
        params["distogram_loss_weight"] = jnp.float32(weight)

    # ---- stage 1: soft simplex APGM ----------------------------------------------
    # simplex_agpm takes the Keygen OBJECT (it calls key() per step); _report_result
    # and metrics take a DRAWN key.
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=key, params=params, context=context,
        num_steps=opt.stage1_steps,
        lr=0.1 * jnp.sqrt(sequence.shape[0]), momentum=0.3,
        scale=1.0,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(0.0),
        verbose=True, return_last=False,
        early_stop=None,
        report=_report_trajectory(name, attempt, trajectory, start_step=0,
                                  weight_tag=weight_tag))
    sequence_1 = sequence
    prediction_1 = _repredict(aux["result"], name, target_sequence)
    placement_kw = dict(
        bp_complex=bp_complex,
        hotspot_mask=None if docked is None else docked["hotspot_mask"],
        pos_center=pos_center,
        target_size=None if docked is None else docked["target_size"],
        hotspot_spec=None if docked is None else docked["spec"],
        coldspot_mask=None if docked is None else docked["coldspot_mask"],
        coldspot_spec=None if docked is None else docked["coldspot_spec"],
        weight_tag=weight_tag)
    stage_1 = _report_stage(name, attempt, "stage_1", sequence_1, prediction_1,
                            design, dssp_requested, target_length, **placement_kw)
    prediction_1.save_pdb(f"{opt.out_path}/attempts/{name}_{attempt}{file_tag}_stage_1.pdb")

    # ---- stage 2: logit-space hardening ------------------------------------------
    sequence = jnp.log(sequence + 1e-5)
    context["recovery"] = 10.0
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=key, params=params, context=context,
        num_steps=opt.stage2_steps,
        lr=0.5 * jnp.sqrt(sequence.shape[0]),
        scale=1.25,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(-1e6),
        verbose=True, return_last=True, opt_logits=True,
        report=_report_trajectory(name, attempt, trajectory,
                                  start_step=opt.stage1_steps, weight_tag=weight_tag))
    sequence_2 = sequence
    prediction_2 = _repredict(aux["result"], name, target_sequence)
    stage_2 = _report_stage(name, attempt, "stage_2", sequence_2, prediction_2,
                            design, dssp_requested, target_length, **placement_kw)
    prediction_2.save_pdb(f"{opt.out_path}/attempts/{name}_{attempt}{file_tag}_stage_2.pdb")

    # keep this attempt's epitope masks for the shuffled-pair null, computed once all
    # attempts for this target are done
    if docked is not None:
        null_rows.append(dict(
            attempt=attempt, weight=weight_tag,
            center=np.asarray(pos_center),
            bp=epitope_mask(bp_complex, opt.epitope_cutoff),
            pred_1=epitope_mask(prediction_1.result.to_data(), opt.epitope_cutoff),
            pred_2=epitope_mask(prediction_2.result.to_data(), opt.epitope_cutoff)))

    # ---- stage selection ----------------------------------------------------------
    stage, chosen, prediction, sequence = "stage_2", stage_2, prediction_2, sequence_2
    if stage_2["ipsae"] < stage_1["ipsae"] and _passing(stage_1):
        stage, chosen, prediction, sequence = "stage_1", stage_1, prediction_1, sequence_1
    if _passing(chosen):
        _report_result(name, attempt, stage, key(), success,
                       sequence, prediction.result, aux=chosen)
        prediction.save_pdb(f"{opt.out_path}/success/{name}_{attempt}{file_tag}_{stage}.pdb")

for name, target_sequence in targets:
    if resume and all((name, a, t) in completed_attempts
                      for a in range(opt.attempts_per_target) for _, t in run_weights):
        print(f"[resume] {name}: all {opt.attempts_per_target} attempts already complete, skipping")
        continue

    if opt.clear_jit_cache == "True":
        jax.clear_caches()

    # One-time per-target seed, used only for the one-time setup below this point: the target
    # structure prediction in _predict_target_pdb (Keygen), and sample_centers_spaced in the
    # docked-blueprint setup a little further down (numpy). Per-attempt work is reseeded
    # separately, inside the attempt loop -- see _attempt_seed above.
    target_seed = opt.seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100000
    key = Keygen(target_seed)
    np.random.seed(target_seed % (2 ** 32))

    # Chain A: the designable binder (all X); chain B..: the target. The template / MSA is static
    # conditioning: to_input() bakes its features into params["structure_input"] once, and
    # nothing about them is differentiable, so it is safe inside the hallucination loss too.
    params["structure_input"] = _to_input(
        _complex_spec("X" * binder_length, name, target_sequence))
    target_length = len(target_sequence.replace(":", ""))
    print(f"=== target {name} ({target_length} aa), "
          f"complex tokens = {binder_length + target_length} "
          f"[{opt.structure_model}, {'template' if use_target_template else 'msa'}] ===")

    # ---- docked-blueprint setup, once per target -------------------------------------
    docked = None
    if dock_blueprint:
        flat_target = target_sequence.replace(":", "")
        # 1. target structure, predicted once and cached by content hash. Shared with
        #    --target_use_template above via _predict_target_pdb's cache, so this is a no-op
        #    (dict lookup) when that flag already triggered the prediction for this target.
        tgt_pdb = _predict_target_pdb(name, target_sequence)

        # 2. per-target config deepcopy -- make_salad_model returns a SHARED mutable object,
        #    and parse_target writes c.target_size onto it. config_from_opt must run BEFORE
        #    binder_step, which deepcopies the config at closure-creation time.
        config_t = deepcopy(salad_ve_config)
        init_data, init_prev_t, target_center, is_target, target_size = parse_target(
            config_t, tgt_pdb, binder_length)
        config_t = config_from_opt(config_t, opt)

        # 3. correspondence assert. The epitope mask and the target-frame RMSD are POSITIONAL
        #    (neither reads residue_index), so what matters is that the blueprint target rows
        #    are the same residues, in the same order, as the Boltz complex's target rows.
        #    from_pdb_string silently drops residues with no resolvable atoms and maps
        #    non-standard residues to aatype 20, either of which would shift every row.
        bp_target_seq = aas.decode(
            np.asarray(init_data["aa_condition"][:target_size]), aas.AF2_CODE)
        assert target_size == len(flat_target) and bp_target_seq == flat_target, (
            f"{name}: blueprint target does not match the fasta target.\n"
            f"  parsed  ({target_size}): {bp_target_seq}\n"
            f"  fasta   ({len(flat_target)}): {flat_target}\n"
            f"Residues may have been dropped by from_pdb_string, or the PDB is stale.")

        # 4. hotspots (and coldspots) -> candidate binder-centroid positions
        spec_str = hotspot_specs.get(name)    # None: no hotspot file -> every residue is eligible
        cold_str = coldspot_specs.get(name)   # None: this target has no coldspots
        hotpos, coldpos, hotspot_mask, coldspot_mask = parse_hotspots(
            init_data, spec_str or "none", cold_str or "none")
        if spec_str is not None:
            n_requested = len(spec_str.split(","))
            assert hotspot_mask.any(), (
                f"{name}: none of the {n_requested} requested hotspots matched a target residue. "
                f"Hotspot numbering is 1-based over the fasta sequence and the predicted target "
                f"is chain A. Without this check an unmatched spec is indistinguishable from an "
                f"undockable target (both end in pos_centers=None).")
            print(f"[hotspot] {name}: matched {int(hotspot_mask.sum())} of {n_requested} residues")
        else:
            # parse_hotspots("none") marked every target residue as a hotspot. That mask is not a
            # user request, so it must not feed hotspot_recall ("fraction of all residues
            # touched" would be meaningless): drop it once the centres are sampled from it.
            print(f"[hotspot] {name}: no hotspot file -> all {int(hotspot_mask.sum())} residues "
                  f"eligible for the docking site")
        if cold_str:
            n_cold_requested = len(cold_str.split(","))
            # An unmatched coldspot spec is WORSE than an unmatched hotspot one: it leaves an
            # empty coldpos, and sample_centers_basic's `(dist > r).all(axis=1)` over an empty
            # axis is True for every candidate -- the filter silently does nothing.
            assert coldspot_mask.any(), (
                f"{name}: none of the {n_cold_requested} requested coldspots matched a target "
                f"residue, so the coldspot filter would silently do nothing. Coldspot numbering "
                f"is 1-based over the fasta sequence and the predicted target is chain A.")
            print(f"[coldspot] {name}: matched {int(coldspot_mask.sum())} of "
                  f"{n_cold_requested} residues; docking centres kept "
                  f"> {opt.coldspot_radius} A from their CA")

        target_ca = init_data["pos"][is_target][:, 1]
        pos_centers = sample_centers_spaced(
            target_ca, hotpos,
            radius_min=opt.radius, clash_radius=opt.clash_radius,
            min_distance=opt.min_center_distance,
            coldspot_radius=opt.coldspot_radius, coldpos=coldpos)
        n_centers = 0 if pos_centers is None else len(pos_centers)
        n_without_coldspots = None
        if coldpos is not None:
            # What the coldspot filter cost, to tune --coldspot_radius by. Sampled AFTER the real
            # call and the result discarded, so it cannot change which centres are used (and
            # every attempt reseeds numpy anyway).
            without = sample_centers_spaced(
                target_ca, hotpos,
                radius_min=opt.radius, clash_radius=opt.clash_radius,
                min_distance=opt.min_center_distance,
                coldspot_radius=opt.coldspot_radius, coldpos=None)
            n_without_coldspots = 0 if without is None else len(without)
        if n_centers == 0:
            reason = "no_docking_centres"
            message = f"[skip] {name}: no viable docking centres for this hotspot"
            if n_without_coldspots:
                # The coldspots alone explain it: say so, since otherwise this is
                # indistinguishable from an undockable hotspot.
                reason = "no_docking_centres_after_coldspots"
                message += (f" -- {n_without_coldspots} exist without the coldspot filter "
                            f"({int(coldspot_mask.sum())} coldspot residues at --coldspot_radius "
                            f"{opt.coldspot_radius} A); lower --coldspot_radius or trim the "
                            f"coldspots")
            print(message)
            skipped.write_line(dict(target=name, reason=reason))
            continue
        suffix = ("" if n_without_coldspots is None
                  else f" ({n_without_coldspots} without the coldspot filter)")
        print(f"[hotspot] {name}: {n_centers} candidate docking centres{suffix}")

        # 5. secondary-structure bias: set ONCE per target, unconditionally. bias_dssp only
        #    writes its keys when a bias is > 0, so setting it per attempt would (a) let a
        #    previous attempt's bias persist and (b) change the pytree mid-loop, forcing a
        #    second salad compile.
        init_data = bias_dssp(init_data, L=opt.l_bias, H=opt.h_bias, E=opt.e_bias,
                              where=~is_target)
        # 6. sampler hoisted to the target loop: si.Sampler.__init__ jits its step and
        #    binder_step returns a fresh function object, so building either per attempt
        #    would recompile salad every attempt.
        sampler_t = si.Sampler(
            binder_step(config_t), prev_threshold=opt.prev_threshold,
            out_steps=400, timescale="ve(t, sigma_max=80.0)")
        docked = dict(
            init_data=init_data, init_prev=init_prev_t, is_target=is_target,
            target_size=target_size, pos_centers=pos_centers,
            # None without a hotspot file, so _placement_comparison skips hotspot_recall
            hotspot_mask=None if spec_str is None else np.asarray(hotspot_mask),
            sampler=sampler_t, spec=spec_str,
            coldspot_mask=None if coldspot_mask is None else np.asarray(coldspot_mask),
            coldspot_spec=cold_str)
        # per-attempt epitope masks, for the shuffled-pair null at the end of this target
        null_rows = []

    for attempt in range(opt.attempts_per_target):
        # runs still to do for this attempt (all of them, unless --resume finished some)
        todo = [(w, t) for w, t in run_weights
                if not (resume and (name, attempt, t) in completed_attempts)]
        if not todo:
            continue

        # Reseed per attempt, not just per target -- see _attempt_seed's docstring. Must
        # happen before anything below draws a key or a numpy random number.
        seed_i = _attempt_seed(name, attempt)
        key = Keygen(seed_i)
        np.random.seed(seed_i % (2 ** 32))

        # ---- a fresh salad backbone per attempt --------------------------------------
        design = None
        dssp_requested = "none"
        bp_complex = None
        pos_center = None
        if docked is not None:
            init_data = docked["init_data"]
            is_target = docked["is_target"]
            target_size = docked["target_size"]
            # pick a docking centre and collapse the binder onto it (bind/__main__.py:131-134).
            # Safe to mutate init_data across attempts: si.inference.sampling.sample
            # shallow-copies the dict on entry (sampling.py:60) before writing data["pos"],
            # so the sampler never touches the target rows we keep here.
            pos_center = docked["pos_centers"][
                np.random.randint(0, len(docked["pos_centers"]))]
            init_data["center"] = jnp.asarray(pos_center, dtype=jnp.float32)
            init_data["pos"] = jnp.where(
                is_target[:, None, None], init_data["pos"], jnp.asarray(pos_center))
            salad_out = docked["sampler"](
                salad_ve_params, key(), init_data, docked["init_prev"])
            bp_complex = data_from_salad(salad_out)

            # FRAME FIX. binder_step aligns only out["pos"] onto the target
            # (_docked_salad.py, index_align with weight=is_target), but salad builds
            # out["atom_pos"] independently from the PRE-alignment pos
            # (noise_schedule_benchmark.py:1115), and data_from_salad reads atom_pos. So the
            # blueprint comes back in a different frame from init_data["pos"] -- off by the
            # joint-centroid recentring (~5 A) plus a rotation. Put it back in the target
            # frame, which is the frame pos_center and hotpos live in. index_align computes
            # the Kabsch from CA only (x[:, 1]), so the atom14/atom-5+augment width
            # difference is fine.
            n_complex = bp_complex.size
            bp_complex = bp_complex.update(atom_positions=index_align(
                bp_complex["atom_positions"], init_data["pos"],
                index=jnp.zeros((n_complex,), dtype=jnp.int32),
                mask=jnp.ones((n_complex,), dtype=jnp.bool_),
                weight=jnp.asarray(is_target, dtype=jnp.float32)))

            # ProteinMPNN over the FULL complex with the target sequence fixed. This is the
            # step that makes the experiment meaningful: interface residues get designed for
            # THIS epitope, so the sequence is the channel carrying the pose. Use
            # aa_condition (native target + 20s), not bp_complex["aa"] -- salad predicts its
            # own sequence and we want the real target identity.
            mpnn_aa = jnp.where(
                jnp.asarray(is_target),
                aas.translate(jnp.asarray(init_data["aa_condition"]),
                              aas.AF2_CODE, aas.PMPNN_CODE),
                20)
            logits = jit_pmpnn(
                pmpnn_params, key(), bp_complex.update(aa=mpnn_aa))["logits"]
            logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
            # target-FIRST layout: the binder is the tail
            sequence = jax.nn.softmax(
                logits[~is_target] / opt.init_temperature, axis=-1)

            # binder-only blueprint for the existing fold comparison, sliced BEFORE any .dssp
            # access so it never inherits complex-context DSSP
            design = bp_complex[~jnp.asarray(is_target)]
            # reorder the blueprint to BINDER-FIRST to match Boltz. The binder must be
            # explicitly re-chained to 0: _sep_groups (data.py:516-523) leaves the FIRST
            # item's chain_index untouched, so `binder / target` with binder chain 1 would
            # give chains 1 and 2, and `chain_index == 0` would match nothing.
            bp_binder = design.update(
                chain_index=jnp.zeros_like(design.chain_index))
            bp_complex = bp_binder / bp_complex[jnp.asarray(is_target)]
            if use_distogram_loss:
                # Distogram of the WHOLE blueprint complex, binder-first like the Boltz tokens.
                # Residue identities are set explicitly because blueprint_distogram picks CA
                # over CB for glycine: target rows take their fasta identity (aa_condition; the
                # blueprint's own aa there is salad's prediction), binder rows are ALA, i.e.
                # CB everywhere, since the binder sequence does not exist yet.
                complex_aa = jnp.concatenate([
                    jnp.full((binder_length,), aas.AF2_CODE.index("A"), dtype=jnp.int32),
                    jnp.asarray(init_data["aa_condition"][:target_size], dtype=jnp.int32)])
                assert complex_aa.shape[0] == bp_complex.size == binder_length + target_length, (
                    f"{name}: blueprint complex has {bp_complex.size} residues, expected "
                    f"{binder_length} binder + {target_length} target")
                params["target_distogram"] = blueprint_distogram(
                    bp_complex.update(aa=complex_aa))
            # Save AFTER the reorder so the blueprint PDB has the same chain layout as the
            # attempt PDBs (A = binder, B = target). Saving the target-first form made the
            # two files disagree, which broke naive overlays in PyMOL and forced
            # scripts/epitope_report.py to detect chains by sequence.
            bp_complex.save_pdb(
                f"{opt.out_path}/blueprints/{name}_{attempt}_complex.pdb")
        elif sampler is not None:
            salad_data["dssp_condition"], dssp_requested = si.random_dssp(
                num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
            print("DSSP target:", dssp_requested)
            salad_data["cloud_std"] = cloud_std_default(num_aa)
            design = data_from_salad(sampler(salad_params, key(), salad_data, init_prev))
            design.save_pdb(f"{opt.out_path}/blueprints/{name}_{attempt}.pdb")
            sequence = aas.translate_onehot(
                jit_pmpnn(pmpnn_params, key(), design.drop_aa())["logits"],
                aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
            sequence = jax.nn.softmax(sequence / opt.init_temperature, axis=-1)
        else:
            sequence = jax.random.gumbel(key(), (opt.length, 20), dtype=jnp.float32)
            sequence *= np.random.uniform(low=0.75, high=5.0)
            sequence = jnp.array(sequence, dtype=jnp.float32)
            sequence = jax.nn.softmax(sequence, axis=-1)

        # ---- one hallucination per weight, all from THIS blueprint and initial sequence ------
        # In sweep mode each weight restarts from the same derived key, so Boltz's sampling noise
        # is shared and differences between weights come from the weight, not from noise.
        # Otherwise `key` simply keeps advancing, exactly as before.
        for weight, weight_tag in todo:
            if sweeping_weights:
                key = Keygen(_attempt_seed(name, attempt, tag="run"))
            _hallucinate_once(
                name, attempt, target_sequence, target_length, sequence, design,
                dssp_requested, bp_complex, pos_center, docked,
                None if docked is None else null_rows,
                weight=weight, weight_tag=weight_tag)

    # ---- shuffled-pair null for this target ------------------------------------------
    # Blueprint of attempt i vs prediction of attempt j (i != j). Without this the matched
    # epitope Jaccard has no baseline. center_distance is recorded because
    # sample_centers_spaced is hotspot-biased by construction: different attempts often land
    # on the same patch, which inflates the null. Condition on it when analysing.
    #
    # Under --resume, null_rows only holds attempts actually re-run THIS session -- already-
    # completed attempts from a prior session are skipped outright (see the attempt loop) and
    # never rebuilt, so they can't be re-paired here. A resumed target's null undercounts the
    # full N*(N-1) matrix across its whole history rather than being wrong; each session's
    # rows are appended, not overwritten, and pairs are never duplicated across sessions since
    # a skipped attempt is never on either side of a newly-written pair.
    if docked is not None and len(null_rows) > 1:
        for a in null_rows:
            for b in null_rows:
                if a["attempt"] == b["attempt"] or a["weight"] != b["weight"]:
                    continue
                dist = float(np.linalg.norm(a["center"] - b["center"]))
                for stage_name, pred_key in (("stage_1", "pred_1"), ("stage_2", "pred_2")):
                    agree = _epitope_agreement(a["bp"], b[pred_key])
                    placement_null.write_line(dict(
                        target=name,
                        distogram_weight=a["weight"],
                        blueprint_attempt=a["attempt"],
                        prediction_attempt=b["attempt"],
                        stage=stage_name,
                        center_distance=round(dist, 3),
                        **{k: float(v) for k, v in agree.items()}))

print(f"done. combined results: {opt.out_path}/results.csv")
