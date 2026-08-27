# Multi-target binder hallucination from salad-initialized sequences.
#
# This is `flexcraft.pipelines.bind.mosaic` swept over a fasta of targets, with the
# blueprint-comparison metrics of `flexcraft.pipelines.experimental.landscape` bolted onto
# every re-prediction. For each target we run a fixed number of attempts; each attempt
# starts from a *fresh* salad backbone whose ProteinMPNN distribution seeds the two-stage
# simplex-APGM hallucination. Everything lands in one combined results.csv.
#
# Note on cost: `params["joltz_input"]` has shape binder_length + len(target), so the jitted
# loss retraces once per target (targets are outer, attempts inner, so each compile is
# amortized over `--attempts_per_target` designs). `pad=True` does not help --
# pad_boltz_atom_features_for_compilation pads only the atom axis, never the token axis.

import os
import re
import time
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
from flexcraft.structure.boltz._model import Joltz2, JoltzResult
from flexcraft.structure.boltz._data import JoltzSpec, JoltzInput
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
)

binder_length = opt.length
sample_structure = opt.sample_structure == "True"
# No per-target hotspot spec is available from bare sequences. With hotspot_index=None,
# metrics()'s hotspot_selector falls back to `chain_index != 0`, so `hotspot_contacts`
# degrades to the symmetrized binder<->whole-target contact score.
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

msa_dir = f"{opt.out_path}/msa" if opt.msa_dir == "none" else opt.msa_dir
os.makedirs(msa_dir, exist_ok=True)
msa_paths = {}
for name, target_sequence in targets:
    digest = hashlib.sha256(target_sequence.encode("utf-8")).hexdigest()[:16]
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

model = Joltz2(cache=opt.boltz_path)
joltz_pred = model.predictor(num_recycle=4)
joltz, joltz_params = model.evaluator(num_recycle=4)
params["joltz"] = joltz_params

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

def metrics(sequence, key, result: JoltzResult):
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
      "dssp" (data.py:267-289). JoltzResult.to_data() also zeroes batch_index across both
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
# reporting (landscape.py:303-325, plus hunter.py's leading `target` column)
# ---------------------------------------------------------------------------------------

def _report_trajectory(target: str, index: int, writer: ScoreCSV, start_step=0):
    def _report(step: int, loss: float, aux: dict):
        line_info = dict(
            target=target, attempt=index, step=start_step + step,
            total=float(loss.mean()))
        for k in writer.keys:
            if k in aux and not isinstance(aux[k], str):
                line_info[k] = float(aux[k].mean())
        writer.write_line(line_info)
    return _report

def _report_result(target: str, attempt: int, stage: str, key,
                   writer: ScoreCSV, sequence, result: JoltzResult, aux=None):
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
# hallucination loss (mosaic.py:344-393, joltz-only, no off-target term)
# ---------------------------------------------------------------------------------------

def loss(sequence, key=None, context=None, params=None):
    keys = jax.random.split(key, 4)
    # forbid cysteines
    sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
    sequence = sequence / sequence.sum(axis=1, keepdims=True)
    # set sequence
    joltz_input: JoltzInput = params["joltz_input"].set_aa(sequence)
    # predict structure
    result: JoltzResult = joltz(params["joltz"]).predict(keys[0], joltz_input, num_samples=4)
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
    out["total"] = value
    return value, out
loss_update = jit_loss_update(loss)

# ---------------------------------------------------------------------------------------
# output files -- created ONCE, before both loops.
# ScoreCSV.__init__ opens "wt" and rewrites the header (files/csv.py:15), so constructing
# one inside the target loop would discard every previous target's rows.
# ---------------------------------------------------------------------------------------

for subdir in ("blueprints", "attempts", "success"):
    os.makedirs(f"{opt.out_path}/{subdir}/", exist_ok=True)

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
blueprint_keys = ["rmsd_to_blueprint", "dssp_identity"]
string_keys = ["dssp_requested", "dssp_salad", "dssp_predicted"]

trajectory = ScoreCSV(
    f"{opt.out_path}/trajectory.csv",
    keys=["target", "attempt", "step", "total"] + common_keys)
results_keys = (
    ["target", "attempt", "stage", "target_length", "passed"]
    + common_keys + blueprint_keys
    + ["sequence"] + string_keys)
results = ScoreCSV(f"{opt.out_path}/results.csv", keys=results_keys)
success = ScoreCSV(f"{opt.out_path}/success.csv", keys=results_keys)

# ---------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------

key = None  # rebound per target

def _repredict(result: JoltzResult, name: str, target_sequence: str):
    """Full-atom re-prediction of the hardened binder against its target.

    A fresh JoltzSpec every call -- JoltzSpec mutates itself (add_* append to self.chains,
    precomputed_msa/to_input flip chain["use_msa"] in place). No use_msa=True here: the
    precomputed MSA covers the target, so this makes no network call.
    """
    binder_sequence = result.to_data().to_sequence_string()[:binder_length]
    spec = (JoltzSpec()
            .add_protein(binder_sequence)
            .add_protein(*target_sequence.split(":"))
            .precomputed_msa(msa_paths[name], start_chain=1))
    return joltz_pred(key(), spec)

def _report_stage(name, attempt, stage, sequence, prediction, design,
                  dssp_requested, target_length):
    """metrics() + blueprint comparison -> one results.csv row. Returns the aux dict."""
    result = prediction.result
    aux = metrics(sequence, key(), result)
    if design is not None:
        aux.update(_structure_comparison(design, result.to_data()))
        aux["dssp_requested"] = dssp_requested
    aux["target_length"] = str(target_length)
    aux["passed"] = "True" if _passing(aux) else "False"
    return _report_result(name, attempt, stage, key(), results,
                          sequence, result, aux=aux)

for name, target_sequence in targets:
    if opt.clear_jit_cache == "True":
        jax.clear_caches()

    # Reseed per target so a single target's attempts are reproducible in isolation
    # (a shared global stream would make target k depend on how many keys 0..k-1 drew,
    # so --only_targets would not reproduce the full sweep).
    key = Keygen(opt.seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100000)

    joltz_spec = (JoltzSpec()
        .add_protein("X" * binder_length)            # chain A: designable binder
        .add_protein(*target_sequence.split(":"))    # chain B..: the target
        .precomputed_msa(msa_paths[name], start_chain=1))
    params["joltz_input"], _ = joltz_spec.to_input(pad=False, cache=opt.boltz_path)
    target_length = len(target_sequence.replace(":", ""))
    print(f"=== target {name} ({target_length} aa), "
          f"complex tokens = {binder_length + target_length} ===")

    for attempt in range(opt.attempts_per_target):
        context = dict(recovery=10.0, ipae=0.05)

        # ---- a fresh salad backbone per attempt --------------------------------------
        design = None
        dssp_requested = "none"
        if sampler is not None:
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
            report=_report_trajectory(name, attempt, trajectory, start_step=0))
        sequence_1 = sequence
        prediction_1 = _repredict(aux["result"], name, target_sequence)
        stage_1 = _report_stage(name, attempt, "stage_1", sequence_1, prediction_1,
                                design, dssp_requested, target_length)
        prediction_1.save_pdb(f"{opt.out_path}/attempts/{name}_{attempt}_stage_1.pdb")

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
                                      start_step=opt.stage1_steps))
        sequence_2 = sequence
        prediction_2 = _repredict(aux["result"], name, target_sequence)
        stage_2 = _report_stage(name, attempt, "stage_2", sequence_2, prediction_2,
                                design, dssp_requested, target_length)
        prediction_2.save_pdb(f"{opt.out_path}/attempts/{name}_{attempt}_stage_2.pdb")

        # ---- stage selection ----------------------------------------------------------
        stage, chosen, prediction, sequence = "stage_2", stage_2, prediction_2, sequence_2
        if stage_2["ipsae"] < stage_1["ipsae"] and _passing(stage_1):
            stage, chosen, prediction, sequence = "stage_1", stage_1, prediction_1, sequence_1
        if _passing(chosen):
            _report_result(name, attempt, stage, key(), success,
                           sequence, prediction.result, aux=chosen)
            prediction.save_pdb(f"{opt.out_path}/success/{name}_{attempt}_{stage}.pdb")

print(f"done. combined results: {opt.out_path}/results.csv")
