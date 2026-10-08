"""Small-molecule / DNA biasing analysis for Boltz2/Joltz2: hallucinated binders.

For each target (a small molecule or a DNA structure) a binder of `--num_aa` residues is
HALLUCINATED against it for `--hallucination_steps` steps (default 100, single-stage), using
the exact hallucination recipe and 9-term loss of mosaic_scan.py's stage 1 (soft simplex-APGM
over a (num_aa, 20) amino-acid distribution, Boltz2 forward+backward each step). The final
hardened sequence is re-predicted with a plain Boltz2 forward pass, and that structure is
analyzed for:

  1. which binder residues form the interface (interface residue bias)?
  2. where along the binder chain does the target dock (positional bias)?
  3. is the binder's fold changed by the target's presence, relative to an apo (no-target)
     hallucination (DSSP composition, CA-RMSD to apo)?
  4. which amino acids does the induced backbone favor, at the interface vs. elsewhere? The
     binder-only structure (target atoms excluded) is fed through ProteinMPNN with the
     sequence masked (titration.py's own diagnostic) -- this measures what the backbone SHAPE
     favors, not direct ligand-chemistry reasoning (ProteinMPNN never sees the target).

Two runs share this script and differ only in how the hallucination's starting sequence
distribution is built (`--init_mode`):

  random       (Run 1, default) Gumbel-noise sequence distribution -- mosaic_scan.py's own
               fallback when no blueprint sampler is active. No salad, no ProteinMPNN: nothing
               structure-informed enters the starting point, so any bias that emerges is
               attributable to Boltz2 itself.
  salad_pmpnn  (Run 2, "arbitrary fold") a fresh, target-unaware salad blueprint whose
               ProteinMPNN distribution seeds the sequence -- mosaic_scan.py's default
               free-mode init / landscape.py's single-blueprint path. Asks whether a ligand
               can be grafted onto a fold that was never chosen with it in mind.

Target categories (`--categories`, comma list of A, B, C, DNA):
  A    conventional PDB/CCD cofactors (real CCD codes, local params/boltz/mols/ cache).
  B    random real biological compounds from a local ChEBI SDF export that are NOT close to
       any Category-A compound and NOT close to each other (Tanimoto), added via SMILES.
  C    random synthetic compounds from a local Enamine REAL '.cxsmiles' export, likewise
       mutually dissimilar (Tanimoto), added via SMILES.
  DNA  three nucleic-acid targets: a B-DNA duplex, a hairpin, a G-quadruplex.

Each target runs `--attempts_per_target` independent hallucinations (default 5). The apo
(no-target) baseline is a single hallucination regardless of attempts_per_target: it only
serves as the fold-impact reference (RMSD-to-apo), not as a population of interest.

Only the best-of-`--num_samples` diffusion sample (lowest mean PAE, via JoltzResult.atom24's
own selection) is kept for every metric of the final structure, matching landscape.py /
titration.py's PDB-saving convention.

Output layout:
    <out_path>/results.csv            one row per (condition, attempt)
    <out_path>/ligands.json           manifest: rows/conditions/attempts, entries, options
    <out_path>/contact_frequency.npy  (N_rows, num_aa) float32, NaN = apo/failed
    <out_path>/aa_preference.npy      (N_rows, num_aa, 20) float32 AF2-ordered, NaN = failed
    <out_path>/predictions/*.pdb      final re-predicted complex per (condition, attempt)
    <out_path>/init/*_init.pdb        Boltz structure of the hardened INITIAL sequence per attempt
    <out_path>/blueprints/*.pdb       salad blueprint per attempt (salad_pmpnn mode only)
    <out_path>/success.csv            attempts passing mosaic_scan.py's success filter (MOSAIC_PASS)
    <out_path>/success/*.pdb          their single-sample re-predictions
"""

import os
import re
import sys
import gzip
import json
import random
import hashlib
import pickle
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn, batch_mpnn
from flexcraft.utils import parse_options, Keygen, data_from_salad
from flexcraft.files import ScoreCSV
from flexcraft.data.data import DesignData
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.structure.boltz import Joltz2, JoltzSpec
from flexcraft.structure.boltz._utils import get_contact_atom
from flexcraft.structure.metrics import RMSD


# ============================== curated targets ==============================
# Category A: all CCD codes confirmed present in params/boltz/mols/ (Boltz-2's local per-code
# cache) -- no network access needed to resolve them.

CATEGORY_A = [
    dict(name="ATP", target_kind="ccd", ccd="ATP", category="A", pairs_with=None,
         notes="nucleotide triphosphate; highly charged, phosphate-rich chemotype"),
    dict(name="NAD", target_kind="ccd", ccd="NAD", category="A", pairs_with=None,
         notes="dinucleotide redox cofactor; two fused ribose-phosphate-base units"),
    dict(name="FAD", target_kind="ccd", ccd="FAD", category="A", pairs_with=None,
         notes="flavin dinucleotide redox cofactor; isoalloxazine ring, distinct from NAD's "
               "nicotinamide/adenine rings despite similar overall size/charge"),
    dict(name="HEM", target_kind="ccd", ccd="HEM", category="A", pairs_with=None,
         notes="metalloporphyrin (heme b); large planar aromatic macrocycle with a chelated Fe"),
    dict(name="SAM", target_kind="ccd", ccd="SAM", category="A", pairs_with=None,
         notes="methyl-donor cofactor; sulfonium center, moderate size, net positive charge"),
    dict(name="GLC", target_kind="ccd", ccd="GLC", category="A", pairs_with=None,
         notes="alpha-D-glucose; simple, small, uncharged monosaccharide"),
    dict(name="ZN", target_kind="ccd", ccd="ZN", category="A", pairs_with=None,
         notes="single-atom divalent cation; no covalent framework at all -- extreme contrast case"),
    dict(name="CIT", target_kind="ccd", ccd="CIT", category="A", pairs_with=None,
         notes="citrate; small, negatively charged tricarboxylic acid"),
]

# DNA targets. No base-pairing / G-tetrad constraint exists in Boltz (only generic
# bond/pocket/contact constraints), so each is handed over as raw sequence(s) and Boltz
# predicts the fold from its own priors, exactly as for a protein sequence.
DNA_TARGETS = [
    dict(name="dna_duplex", target_kind="dna", sequence="CGCGAATTCGCG", num_repeats=2,
         category="DNA", pairs_with=None,
         notes="Dickerson-Drew B-DNA dodecamer (PDB 1BNA). Self-complementary, so two copies of "
               "the same strand form the duplex: ~1.1 helical turns, 12 bp."),
    dict(name="dna_hairpin", target_kind="dna", sequence="GCGCTTCGGCGC", num_repeats=1,
         category="DNA", pairs_with=None,
         notes="single strand: self-complementary GCGC stem closed by a TTCG tetraloop."),
    dict(name="dna_gquad", target_kind="dna", sequence="GGGTTAGGGTTAGGGTTAGGG", num_repeats=1,
         category="DNA", pairs_with=None,
         notes="human telomeric repeat, the canonical intramolecular G-quadruplex former."),
]


# ============================== target dispatch ==============================

def _add_target(spec: JoltzSpec, entry: dict) -> JoltzSpec:
    kind = entry["target_kind"]
    if kind == "ccd":
        return spec.add_ccd(entry["ccd"])
    if kind == "smiles":
        return spec.add_smiles(entry["smiles"])
    if kind == "dna":
        return spec.add_dna(entry["sequence"], num_repeats=entry.get("num_repeats", 1))
    if kind == "rna":
        return spec.add_rna(entry["sequence"], num_repeats=entry.get("num_repeats", 1))
    raise ValueError(f"unknown target_kind {kind!r} in entry {entry.get('name')!r}")


def _num_target_chains(entry: dict) -> int:
    """Chains `_add_target` appends after the binder (chain 0): only polymers with
    num_repeats > 1 (the DNA duplex) add more than one."""
    if entry["target_kind"] in ("dna", "rna"):
        return entry.get("num_repeats", 1)
    return 1


# ============================== fingerprints / diversity ==============================

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _morgan_fp(mol):
    return _MORGAN.GetFingerprint(mol)


def _max_similarity(fp, fps) -> float:
    if not fps:
        return 0.0
    return max(DataStructs.BulkTanimotoSimilarity(fp, fps))


def _greedy_diverse_subset(candidates, n, max_pairwise_similarity, rng,
                            reference_fps=None, max_similarity_to_reference=None):
    """Shuffle `candidates` (dicts with an "fp" key) and greedily accept up to `n` whose
    Tanimoto similarity to every already-accepted candidate stays below
    `max_pairwise_similarity` and, if `reference_fps` is given, to every reference
    fingerprint stays below `max_similarity_to_reference`."""
    pool = list(candidates)
    rng.shuffle(pool)
    selected, selected_fps = [], []
    for cand in pool:
        if len(selected) >= n:
            break
        if reference_fps is not None and \
                _max_similarity(cand["fp"], reference_fps) >= max_similarity_to_reference:
            continue
        if _max_similarity(cand["fp"], selected_fps) >= max_pairwise_similarity:
            continue
        selected.append(cand)
        selected_fps.append(cand["fp"])
    return selected


def _safe_token(raw: str, max_len: int = 40) -> str:
    """Filename- and CSV-safe token (ScoreCSV does no quoting, so a comma corrupts a row)."""
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", str(raw).strip())[:max_len] or "unnamed"


def category_a_fingerprints(mol_dir: str) -> list:
    """Morgan fingerprints of the real Category-A structures, read from the local CCD cache."""
    fps = []
    for entry in CATEGORY_A:
        with open(f"{mol_dir}/{entry['ccd']}.pkl", "rb") as f:
            mol = Chem.RemoveHs(pickle.load(f))
        fps.append(_morgan_fp(mol))
    return fps


# ============================== Enamine REAL '.cxsmiles' loader ==============================

_ID_NAME_PATTERN = re.compile(r"id|catalog|name", re.IGNORECASE)
_HAC_NAME_PATTERN = re.compile(
    r"^(hac|heavy[_ ]?atoms?|num[_ ]?heavy[_ ]?atoms?|heavyatomcount)$", re.IGNORECASE)


def _require_enamine_file(path: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"--enamine_path {path!r} does not exist.\n"
            "Category C ('far' Enamine REAL compounds) needs a local Enamine REAL "
            "'.cxsmiles' export (tab-separated, with a SMILES column and an id/catalog-"
            "number-style column). Point --enamine_path at a real export, or drop 'C' "
            "from --categories (e.g. --categories \"A,B,DNA\") to skip this category.")


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_int(s):
    try:
        int(float(s))
        return True
    except ValueError:
        return False


def _detect_columns(header: list, sample_rows: list, smiles_col: str, id_col: str,
                     min_parse_fraction: float = 0.9):
    """Return (smiles_index, id_index_or_None, hac_index_or_None), autodetected from the
    header and a sample of rows unless overridden. SMILES: the column RDKit parses best
    (must clear min_parse_fraction). ID: a header matching _ID_NAME_PATTERN whose sampled
    values are not all numeric. HAC: a header matching _HAC_NAME_PATTERN with int values,
    used to skip per-row RDKit parsing during the full-file scan."""
    n_cols = len(header)

    def _col_values(idx):
        return [row[idx].strip() for row in sample_rows if idx < len(row) and row[idx].strip()]

    if smiles_col:
        smiles_idx = header.index(smiles_col)
    else:
        best_idx, best_frac = None, 0.0
        for idx in range(n_cols):
            values = _col_values(idx)
            if not values:
                continue
            frac = sum(1 for v in values if Chem.MolFromSmiles(v) is not None) / len(values)
            if frac > best_frac:
                best_idx, best_frac = idx, frac
        if best_idx is None or best_frac < min_parse_fraction:
            raise ValueError(
                f"Could not autodetect a SMILES column (best parse rate {best_frac:.2f} < "
                f"{min_parse_fraction}). Pass --enamine_smiles_col explicitly.")
        smiles_idx = best_idx

    if id_col:
        id_idx = header.index(id_col)
    else:
        candidates = [
            i for i, h in enumerate(header) if i != smiles_idx and _ID_NAME_PATTERN.search(h)
            and not (_col_values(i) and all(_is_float(v) for v in _col_values(i)))]
        if not candidates:
            print("[enamine] WARNING: no id-like column found; falling back to row numbers. "
                  "Pass --enamine_id_col to override.")
            id_idx = None
        else:
            if len(candidates) > 1:
                print(f"[enamine] WARNING: multiple id-like columns "
                      f"({[header[i] for i in candidates]}); using {header[candidates[0]]!r}.")
            id_idx = candidates[0]

    hac_idx = None
    for i, h in enumerate(header):
        if i in (smiles_idx, id_idx) or not _HAC_NAME_PATTERN.match(h):
            continue
        values = _col_values(i)
        if values and all(_is_int(v) for v in values):
            hac_idx = i
            break
    return smiles_idx, id_idx, hac_idx


def load_enamine_ligands(path: str, n: int, seed: int, min_heavy_atoms: int,
                          max_heavy_atoms: int, max_pairwise_similarity: float,
                          pool_size: int = 300, smiles_col: str = "", id_col: str = "",
                          delimiter: str = "\t", sample_rows_for_autodetect: int = 200,
                          max_scan_rows: int = 2_000_000) -> list:
    """Sample `n` mutually dissimilar compounds from an Enamine REAL '.cxsmiles' export.

    Streams the file once, reservoir-sampling a `pool_size`-candidate pool of compounds in the
    heavy-atom window, then greedily selects `n` from the pool whose pairwise Morgan/Tanimoto
    similarity stays below `max_pairwise_similarity`. Returns target entries (SMILES kind).
    """
    _require_enamine_file(path)
    with open(path, "rt", newline="") as f:
        header = f.readline().rstrip("\n").split(delimiter)
        sample_rows = []
        for _ in range(sample_rows_for_autodetect):
            line = f.readline()
            if not line:
                break
            sample_rows.append(line.rstrip("\n").split(delimiter))
        smiles_idx, id_idx, hac_idx = _detect_columns(header, sample_rows, smiles_col, id_col)
        print(f"[enamine] columns: smiles={header[smiles_idx]!r}, "
              f"id={(header[id_idx] if id_idx is not None else '<row-number fallback>')!r}, "
              f"heavy_atoms={'RDKit' if hac_idx is None else header[hac_idx]!r}")

        f.seek(0)
        f.readline()
        rng = random.Random(seed)
        reservoir, seen, line_no = [], 0, -1
        n_cols = len(header)
        for line_no, line in enumerate(f):
            if max_scan_rows and line_no >= max_scan_rows:
                print(f"[enamine] WARNING: hit --enamine_max_scan_rows={max_scan_rows}; the "
                      "pool may be biased toward the file's head.")
                break
            fields = line.rstrip("\n").split(delimiter)
            if len(fields) != n_cols:
                continue
            # Drop a CXSMILES extended-stereo suffix ("... |&1:3,4,6|"): not part of the
            # SMILES, and its commas corrupt ScoreCSV's unquoted rows.
            smiles = fields[smiles_idx].strip().split("|")[0].strip()
            if not smiles:
                continue
            if hac_idx is not None:
                if not _is_int(fields[hac_idx]):
                    continue
                heavy_atoms = int(float(fields[hac_idx]))
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                heavy_atoms = mol.GetNumHeavyAtoms()
            if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
                continue
            catalog_id = fields[id_idx].strip() if id_idx is not None else f"row{line_no}"
            seen += 1
            entry = dict(smiles=smiles, catalog_id=catalog_id, heavy_atoms=heavy_atoms)
            if len(reservoir) < pool_size:
                reservoir.append(entry)
            else:
                j = rng.randint(0, seen - 1)
                if j < pool_size:
                    reservoir[j] = entry

    pool = []
    for e in reservoir:
        mol = Chem.MolFromSmiles(e["smiles"])
        if mol is None:
            print(f"[enamine] WARNING: dropping {e['catalog_id']!r}, SMILES failed RDKit "
                  f"re-validation: {e['smiles']!r}")
            continue
        pool.append(dict(e, fp=_morgan_fp(mol)))
    selected = _greedy_diverse_subset(pool, n, max_pairwise_similarity, rng)
    if len(selected) < n:
        print(f"[enamine] WARNING: only {len(selected)}/{n} mutually dissimilar compounds "
              f"found in a pool of {len(pool)} (scanned {line_no + 1} rows). Raise "
              "--enamine_pool_size / --diversity_max_similarity or widen the heavy-atom range.")
    return [
        dict(name=f"enamine_{_safe_token(e['catalog_id'])}", target_kind="smiles",
             smiles=e["smiles"], category="C", pairs_with=None,
             notes=f"Enamine REAL; catalog_id={e['catalog_id']}, heavy_atoms={e['heavy_atoms']}",
             catalog_id=_safe_token(e["catalog_id"]), heavy_atoms=e["heavy_atoms"])
        for e in selected
    ]


# ============================== ChEBI SDF loader ==============================

def _require_chebi_file(path: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"--chebi_path {path!r} does not exist.\n"
            "Category B (random real biological compounds, far from Category A and from each "
            "other) needs a local ChEBI SDF export (ChEBI_complete.sdf or ChEBI_lite.sdf, "
            "optionally .gz, from the ChEBI downloads page). Point --chebi_path at it, or drop "
            "'B' from --categories (e.g. --categories \"A,C,DNA\") to skip this category.")


def load_chebi_ligands(path: str, n: int, seed: int, min_heavy_atoms: int,
                        max_heavy_atoms: int, category_a_fps: list,
                        max_similarity_to_category_a: float, max_pairwise_similarity: float,
                        pool_size: int = 300, max_scan_mols: int = 200_000) -> list:
    """Sample `n` real ChEBI compounds that are far from every Category-A compound and from
    each other (Morgan/Tanimoto).

    Streams the SDF once, reservoir-sampling a `pool_size`-candidate pool of compounds in the
    heavy-atom window whose similarity to every Category-A fingerprint stays below
    `max_similarity_to_category_a`, then greedily selects `n` from the pool whose pairwise
    similarity stays below `max_pairwise_similarity`. Returns target entries (SMILES kind).
    """
    _require_chebi_file(path)
    rng = random.Random(seed)
    reservoir, seen, scanned = [], 0, 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        for scanned, mol in enumerate(Chem.ForwardSDMolSupplier(f, sanitize=True, removeHs=True)):
            if max_scan_mols and scanned >= max_scan_mols:
                print(f"[chebi] WARNING: hit --chebi_max_scan_mols={max_scan_mols}; the pool "
                      "may be biased toward the file's head.")
                break
            if mol is None:
                continue
            heavy_atoms = mol.GetNumHeavyAtoms()
            if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
                continue
            fp = _morgan_fp(mol)
            if _max_similarity(fp, category_a_fps) >= max_similarity_to_category_a:
                continue
            props = mol.GetPropsAsDict()
            chebi_id = next((str(props[k]) for k in ("ChEBI ID", "ChEBI_ID", "ID") if k in props),
                            f"mol{scanned}")
            chebi_name = next((str(props[k]) for k in ("ChEBI NAME", "ChEBI Name", "Name")
                               if k in props), chebi_id)
            seen += 1
            entry = dict(smiles=Chem.MolToSmiles(mol), chebi_id=chebi_id, chebi_name=chebi_name,
                         heavy_atoms=heavy_atoms, fp=fp)
            if len(reservoir) < pool_size:
                reservoir.append(entry)
            else:
                j = rng.randint(0, seen - 1)
                if j < pool_size:
                    reservoir[j] = entry

    selected = _greedy_diverse_subset(reservoir, n, max_pairwise_similarity, rng)
    print(f"[chebi] scanned {scanned + 1} molecules; {seen} passed the size and "
          f"Category-A-distance filters; pool={len(reservoir)}; selected {len(selected)}")
    if len(selected) < n:
        print(f"[chebi] WARNING: only {len(selected)}/{n} mutually dissimilar compounds found. "
              "Raise --chebi_pool_size / --diversity_max_similarity or widen the heavy-atom range.")
    return [
        dict(name=f"chebi_{_safe_token(e['chebi_id'])}", target_kind="smiles", smiles=e["smiles"],
             category="B", pairs_with=None,
             notes=f"ChEBI; id={e['chebi_id']}, name={e['chebi_name']}, "
                   f"heavy_atoms={e['heavy_atoms']}",
             catalog_id=_safe_token(e["chebi_id"]), heavy_atoms=e["heavy_atoms"])
        for e in selected
    ]


# ============================== CLI ==============================

_DEFAULT_ENAMINE_PATH = "runs/2026.01_Enamine_REAL_DB_13.6M.cxsmiles"

opt = parse_options(
    "Hallucinated-binder biasing analysis for Boltz2/Joltz2 against small-molecule and DNA "
    "targets (see the module docstring for the two --init_mode runs).",
    # -- binder / structure prediction ---------------------------------------------------
    num_aa="100",
    out_path="runs/ligand_bias/",
    boltz_path="./params/boltz/",
    num_samples=4,
    num_recycle=4,
    num_sampling_steps=25,
    deterministic="False",
    attempts_per_target=5,
    contact_distance=8.0,
    top_k_residues=5,
    pmpnn_params="params/pmpnn/v_48_030.pkl",
    pmpnn_temperature=0.1,
    # -- hallucination ---------------------------------------------------------------------
    hallucination_steps=100,
    init_mode="random",                # random | salad_pmpnn
    salad_params="params/salad/default_vp_scaled-200k.jax",   # salad_pmpnn mode only
    init_temperature=1.0,              # salad_pmpnn mode only
    std_buffer=3.0,                    # salad_pmpnn mode only
    save_blueprints="True",            # salad_pmpnn mode only
    save_init="True",                  # Boltz prediction of the hardened INITIAL sequence, <out>/init/
    # -- which targets to run ---------------------------------------------------------------
    categories="A,B,C,DNA",
    only_targets="none",               # comma list of target names to keep (pilot runs), e.g. "ZN,dna_hairpin"
    diversity_max_similarity=0.5,      # pairwise Tanimoto ceiling within categories B and C
    # -- Category B: ChEBI -------------------------------------------------------------------
    chebi_path="",
    chebi_n=5,
    chebi_seed=1,
    chebi_min_heavy_atoms=15,
    chebi_max_heavy_atoms=40,
    chebi_max_similarity_to_a=0.5,     # Tanimoto ceiling against every Category-A compound
    chebi_pool_size=300,
    chebi_max_scan_mols=200000,
    # -- Category C: Enamine REAL ------------------------------------------------------------
    enamine_path=_DEFAULT_ENAMINE_PATH,
    enamine_n=5,
    enamine_seed=0,
    enamine_min_heavy_atoms=15,
    enamine_max_heavy_atoms=40,
    enamine_pool_size=300,
    enamine_smiles_col="",
    enamine_id_col="",
    enamine_delimiter="\t",
    enamine_max_scan_rows=2000000,
    # -- execution ----------------------------------------------------------------------------
    save_predictions="True",
    resume="False",
    seed=42,
)

num_aa = int(opt.num_aa)
BINDER_CHAIN = 0

assert opt.init_mode in ("random", "salad_pmpnn"), \
    f"--init_mode must be 'random' or 'salad_pmpnn', got {opt.init_mode!r}"
CATEGORIES = [c.strip().upper() for c in opt.categories.split(",") if c.strip()]
assert CATEGORIES and set(CATEGORIES) <= {"A", "B", "C", "DNA"}, \
    f"--categories {opt.categories!r} must be a comma list drawn from {{A, B, C, DNA}}"

# ============================== manifest ==============================

MANIFEST = [None]  # apo (no-target) baseline is always condition_index 0
if "A" in CATEGORIES:
    MANIFEST += CATEGORY_A
if "B" in CATEGORIES:
    MANIFEST += load_chebi_ligands(
        opt.chebi_path, n=opt.chebi_n, seed=opt.chebi_seed,
        min_heavy_atoms=opt.chebi_min_heavy_atoms, max_heavy_atoms=opt.chebi_max_heavy_atoms,
        category_a_fps=category_a_fingerprints(f"{opt.boltz_path}/mols"),
        max_similarity_to_category_a=opt.chebi_max_similarity_to_a,
        max_pairwise_similarity=opt.diversity_max_similarity,
        pool_size=opt.chebi_pool_size, max_scan_mols=opt.chebi_max_scan_mols)
if "C" in CATEGORIES:
    MANIFEST += load_enamine_ligands(
        opt.enamine_path, n=opt.enamine_n, seed=opt.enamine_seed,
        min_heavy_atoms=opt.enamine_min_heavy_atoms, max_heavy_atoms=opt.enamine_max_heavy_atoms,
        max_pairwise_similarity=opt.diversity_max_similarity, pool_size=opt.enamine_pool_size,
        smiles_col=opt.enamine_smiles_col, id_col=opt.enamine_id_col,
        delimiter=opt.enamine_delimiter, max_scan_rows=opt.enamine_max_scan_rows)
if "DNA" in CATEGORIES:
    MANIFEST += DNA_TARGETS
if opt.only_targets != "none":
    _keep = {t.strip() for t in opt.only_targets.split(",") if t.strip()}
    _missing = _keep - {e["name"] for e in MANIFEST if e is not None}
    assert not _missing, f"--only_targets names not in the manifest: {sorted(_missing)}"
    MANIFEST = [e for e in MANIFEST if e is None or e["name"] in _keep]
NUM_CONDITIONS = len(MANIFEST)

# ============================== rows (condition x attempt) ==============================
# apo is always a single row; every other condition gets `attempts_per_target` rows. ROWS is
# the flat storage/iteration order for results.csv and the two .npy tensors below.

ROWS = [dict(row_index=0, condition_index=0, attempt_index=0, entry=None)]
for condition_index, entry in enumerate(MANIFEST):
    if condition_index == 0:
        continue
    for attempt_index in range(opt.attempts_per_target):
        ROWS.append(dict(row_index=len(ROWS), condition_index=condition_index,
                          attempt_index=attempt_index, entry=entry))
N_ROWS = len(ROWS)

print(f"Manifest: {NUM_CONDITIONS} conditions (1 apo + "
      + " + ".join(f"{sum(1 for e in MANIFEST if e and e['category'] == c)} {c}"
                   for c in ("A", "B", "C", "DNA"))
      + f"), {opt.attempts_per_target} attempts/target -> {N_ROWS} rows; "
      f"init_mode={opt.init_mode}, {opt.hallucination_steps} hallucination steps")


# ============================== model setup ==============================

joltz_model = Joltz2(cache=opt.boltz_path)
# differentiable path (hallucination) and plain path (final re-prediction of the hardened binder)
evaluator, evaluator_params = joltz_model.evaluator(
    num_recycle=opt.num_recycle, num_sampling_steps=opt.num_sampling_steps,
    deterministic=(opt.deterministic == "True"))
predict_fn = joltz_model.predictor(
    num_recycle=opt.num_recycle, num_samples=opt.num_samples,
    num_sampling_steps=opt.num_sampling_steps, deterministic=(opt.deterministic == "True"))
# mosaic_scan.py scores and filters a design on a SINGLE-sample re-prediction (its
# structure_pred = Joltz2.predictor(num_recycle=4), num_samples defaults to 1). Mirror that for
# the success filter; the best-of-num_samples prediction above stays the one the analysis uses.
score_predict_fn = joltz_model.predictor(
    num_recycle=opt.num_recycle, num_samples=1,
    num_sampling_steps=opt.num_sampling_steps, deterministic=(opt.deterministic == "True"))

_pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_params, eps=0.05, split_params=True)
params = dict(mpnn=pmpnn_params)
jit_pmpnn = jax.jit(_pmpnn)
batched_pmpnn = jax.jit(batch_mpnn(_pmpnn))

sampler = None
if opt.init_mode == "salad_pmpnn":
    import salad.inference as si

    def _salad_model_step(config):
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

    salad_config, salad_params = si.make_salad_model("default_vp_scaled", opt.salad_params)
    _n, _resi, _chain, _is_cyclic, _cyclic_mask = si.parse_num_aa(f"{num_aa}")
    salad_data, init_prev = si.data.from_config(
        salad_config, num_aa=_n, residue_index=_resi, chain_index=_chain, cyclic_mask=_cyclic_mask)
    sampler = si.Sampler(_salad_model_step(salad_config), out_steps=400)

os.makedirs(opt.out_path, exist_ok=True)
os.makedirs(f"{opt.out_path}/predictions", exist_ok=True)
os.makedirs(f"{opt.out_path}/success", exist_ok=True)
if opt.init_mode == "salad_pmpnn":
    os.makedirs(f"{opt.out_path}/blueprints", exist_ok=True)
if opt.save_init == "True":
    os.makedirs(f"{opt.out_path}/init", exist_ok=True)

with open(f"{opt.out_path}/ligands.json", "w") as f:
    json.dump(dict(
        num_aa=num_aa, options=vars(opt),
        conditions=[dict(condition_index=0, entry=None)] + [
            dict(condition_index=i, entry=e) for i, e in enumerate(MANIFEST) if i > 0],
        rows=[dict(row_index=r["row_index"], condition_index=r["condition_index"],
                    attempt_index=r["attempt_index"]) for r in ROWS],
    ), f, indent=2)


# ============================== physicochemical classes ==============================

AA_CLASSES = dict(
    hydrophobic="AVLIM", aromatic="FWY", positive="KRH",
    negative="DE", polar="STNQC", special="GP")
_AA_INDEX = {a: aas.AF2_CODE.index(a) for a in aas.AF2_CODE}


def _class_fractions(prob: np.ndarray, weights: np.ndarray = None, prefix: str = "") -> dict:
    """prob: (L, 20) AF2-ordered distribution. weights: optional (L,) nonnegative weights
    (e.g. interface contacts) restricting the summary toward interface residues; None means
    an unweighted mean over all L residues."""
    if weights is None:
        per_res = prob
        reduce = lambda mass: float(mass.mean())
    else:
        w = weights / max(float(weights.sum()), 1e-8)
        per_res = prob * w[:, None]
        reduce = lambda mass: float(mass.sum())
    out = {}
    for cls, letters in AA_CLASSES.items():
        cls_idx = [_AA_INDEX[a] for a in letters]
        out[f"{prefix}frac_{cls}"] = reduce(per_res[:, cls_idx].sum(axis=-1))
    return out


# ============================== hallucination ==============================

def hallucination_metrics(sequence, key, result, has_target: bool, target_chain_indices: list):
    """mosaic_scan.py's metrics() (L591-643), keeping only the terms the loss uses. The
    binder is always the first `num_aa` tokens, so it is sliced positionally. PAE terms use
    positional slices and iptm/eptm use a boolean binder mask, so they cover any number of
    target chains; chain_contact_score takes the list of target chain indices."""
    res_data = result.to_data(return_samples=True)
    binder_data = res_data[:num_aa]
    dssp = binder_data.p_dssp
    pmpnn_input = res_data.update(aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    is_binder = jnp.zeros((pmpnn_input.aa.shape[0],), dtype=jnp.bool_).at[:num_aa].set(True)
    logits = batched_pmpnn(params["mpnn"], key, pmpnn_input.drop_aa(where=is_binder))["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not result.is_single_sample:
        logits = logits.mean(axis=0)
    logits = logits[:num_aa]
    logits = logits - logits.mean(axis=0)
    prob = jax.lax.stop_gradient(jax.nn.softmax(
        logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    plddt = result.plddt
    pae = 32 * result.pae
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)
    out = dict(
        plddt=plddt[:num_aa].mean(),
        recovery=(sequence * prob).sum(axis=-1).mean(),
        within_binder_contacts=result.chain_contact_score(0, 0, num_contacts=25, min_resi_distance=10),
        within_binder_pae=pae[:num_aa, :num_aa].mean(),
        eptm=result.index_ptm_score(chain_index=is_binder),
        L=dssp["L"], H=dssp["H"], E=dssp["E"],
    )
    if has_target:
        out["binder_target_contacts"] = result.chain_contact_score(
            target_chain_indices, 0, num_contacts=3, contact_distance=20.0)
        out["binder_target_pae"] = pae[:num_aa, num_aa:].mean()
        out["target_binder_pae"] = pae[num_aa:, :num_aa].mean()
        out["iptm"] = result.index_iptm(chain_index=is_binder)
    return out


def make_loss_fn(has_target: bool, target_chain_indices: list):
    """One loss closure per condition. `has_target` and `target_chain_indices` are closed over
    as plain Python values rather than passed through the jitted `params` pytree: a traced
    bool cannot drive a Python `if`, and a list would become pytree leaves."""
    def loss(sequence, key=None, context=None, params=None):
        keys = jax.random.split(key, 3)
        # forbid cysteines
        sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
        sequence = sequence / sequence.sum(axis=1, keepdims=True)
        structure_input = params["predictor_input"].set_aa(sequence)
        result = evaluator(params["structure"]).predict(
            keys[0], structure_input, num_samples=opt.num_samples)
        out = hallucination_metrics(sequence, keys[1], result, has_target, target_chain_indices)
        if has_target:
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
        else:
            # apo: mosaic's loss minus every term that references a target chain
            value = (
                  out["within_binder_contacts"]
                - 10.0 * out["recovery"]
                + 0.4 * out["within_binder_pae"]
                - 0.1 * out["plddt"]
                - 0.025 * out["eptm"]
            )
        out["result"] = result
        out["total"] = value
        return value, out
    return jit_loss_update(loss)


def build_condition(entry):
    """Once per condition (not per attempt): the static differentiable input, the params
    handed to the optimizer, and the jitted loss. Like mosaic_scan.py, the loss retraces once
    per condition (token count differs) and that compile is amortized over its attempts."""
    spec = JoltzSpec().add_protein("X" * num_aa)
    if entry is not None:
        spec = _add_target(spec, entry)
    predictor_input, _ = spec.to_input(pad=False, cache=opt.boltz_path)
    has_target = entry is not None
    target_chain_indices = list(range(1, 1 + _num_target_chains(entry))) if has_target else []
    loss_update = make_loss_fn(has_target, target_chain_indices)
    return dict(predictor_input=predictor_input, structure=evaluator_params), loss_update


# mosaic_scan.py's _passing(): a design passes when ALL of these hold on its re-prediction
# (ipsae > 0.75, binder pLDDT > 0.8, both directional PAEs < 5 A, ProteinMPNN recovery > 0.6).
MOSAIC_PASS = dict(ipsae=0.75, plddt=0.80, binder_target_pae=5.0, target_binder_pae=5.0,
                   recovery=0.60)


def mosaic_scores(sequence, key, result, entry) -> dict:
    """mosaic_scan.py's metrics() on a (single-sample) re-prediction, plus its pass/fail.
    `sequence` is the final SOFT hallucinated distribution (as in mosaic, whose recovery is
    (soft sequence * ProteinMPNN prob) even though the structure is that of the hardened
    sequence). No target means no interface terms, so `passed` is undefined for the apo."""
    has_target = entry is not None
    target_chain_indices = list(range(1, 1 + _num_target_chains(entry))) if has_target else []
    m = hallucination_metrics(sequence, key, result, has_target, target_chain_indices)
    out = dict(mosaic_plddt=float(m["plddt"]), mosaic_recovery=float(m["recovery"]))
    if not has_target:
        return out
    is_binder = jnp.zeros((result.chain_index.shape[0],), dtype=jnp.bool_).at[:num_aa].set(True)
    out.update(mosaic_ipsae=float(result.ipsae(chain_index=is_binder)),
               mosaic_binder_target_pae=float(m["binder_target_pae"]),
               mosaic_target_binder_pae=float(m["target_binder_pae"]))
    out["passed"] = str(bool(
        out["mosaic_ipsae"] > MOSAIC_PASS["ipsae"]
        and out["mosaic_plddt"] > MOSAIC_PASS["plddt"]
        and out["mosaic_binder_target_pae"] < MOSAIC_PASS["binder_target_pae"]
        and out["mosaic_target_binder_pae"] < MOSAIC_PASS["target_binder_pae"]
        and out["mosaic_recovery"] > MOSAIC_PASS["recovery"]))
    return out


def _seed_host_rng(seed_int: int):
    """salad's random_dssp and the cloud_std / Gumbel-scale draws use the global numpy and
    python RNGs; seed both per attempt so a --resume'd attempt reproduces exactly."""
    np.random.seed(seed_int % (2 ** 32))
    random.seed(seed_int)


def _attempt_seed(row_index: int, tag: str) -> int:
    digest = hashlib.sha256(f"{opt.seed}|{tag}|{row_index}".encode()).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)


def make_init_sequence(row_index: int, key):
    """(num_aa, 20) starting distribution, plus the salad blueprint (None in random mode)."""
    _seed_host_rng(_attempt_seed(row_index, "init"))
    if opt.init_mode == "salad_pmpnn":
        salad_key, pmpnn_key = jax.random.split(key)
        salad_data["dssp_condition"], _ = si.random_dssp(
            num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
        salad_data["cloud_std"] = num_aa ** 0.4 + np.random.rand() * opt.std_buffer
        design = data_from_salad(sampler(salad_params, salad_key, salad_data, init_prev))
        logits = jit_pmpnn(pmpnn_params, pmpnn_key, design.drop_aa())["logits"]
        sequence = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
        return jax.nn.softmax(sequence / opt.init_temperature, axis=-1), design
    sequence = jax.random.gumbel(key, (num_aa, 20), dtype=jnp.float32)
    sequence = sequence * np.random.uniform(low=0.75, high=5.0)
    return jax.nn.softmax(jnp.asarray(sequence, dtype=jnp.float32), axis=-1), None


def hallucinate(row_index: int, condition_params: dict, loss_update, key):
    """One single-stage hallucination (mosaic_scan.py's stage 1, `hallucination_steps` long).
    Returns the final soft sequence, the optimizer's final loss, the aux of the returned
    (best) step, the salad blueprint (or None) and the initial soft sequence."""
    init_key, loop_key = jax.random.split(key)
    sequence, design = make_init_sequence(row_index, init_key)
    init_sequence = sequence
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=Keygen(loop_key), params=condition_params,
        num_steps=opt.hallucination_steps,
        lr=0.1 * jnp.sqrt(sequence.shape[0]), momentum=0.3,
        scale=1.0,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(0.0),
        verbose=True, return_last=False, early_stop=None)
    return sequence, loss_val, aux, design, init_sequence


# ============================== structure helpers ==============================

def dssp_to_string(dssp_array) -> str:
    code = np.array(list("LHE"))
    return "".join(code[np.asarray(dssp_array)])


def _binder_slice(result) -> DesignData:
    """Binder-chain-only view of this attempt's single best-of-N structure: result.to_data()
    with no return_samples=True routes through JoltzResult.atom24, which picks the diffusion
    sample with the lowest mean PAE. The returned DesignData is single-sample, so ordinary
    boolean chain-indexing is safe."""
    data = result.to_data()
    chain_index = np.asarray(data.chain_index)
    return data[chain_index == BINDER_CHAIN]


def _dssp_composition(binder_data: DesignData) -> dict:
    p = binder_data.p_dssp
    return dict(dssp_L=float(p["L"]), dssp_H=float(p["H"]), dssp_E=float(p["E"]),
                dssp_hard=dssp_to_string(np.asarray(binder_data.dssp)))


def _rmsd_to_apo(binder_data: DesignData, apo_binder_data: DesignData) -> float:
    return float(RMSD(atoms=["CA"])(binder_data, apo_binder_data))


def _single_sample_binder_contact(result, chain_index: np.ndarray) -> np.ndarray:
    """Binary (0/1) contact indicator per binder residue against ANY non-binder chain, on the
    single best-of-N structure (result.atom24's own selection). Same per-atom "contact atom"
    + pairwise-distance logic as JoltzResult.chain_contact_frequency (get_contact_atom picks
    the representative atom per mol_type: protein pseudo-CB, DNA/RNA base N, a ligand's own
    atom tokens), restricted to the one collapsed structure; `.any(axis=1)` aggregates over a
    multi-atom ligand's tokens and over multiple target chains."""
    atom24, _ = result.atom24
    mol_type = result.data["features"]["mol_type"][0]
    contact_atom = get_contact_atom(atom24, mol_type)
    dist = jnp.linalg.norm(contact_atom[:, None, :] - contact_atom[None, :, :], axis=-1)
    is_binder = jnp.asarray(chain_index) == BINDER_CHAIN
    is_target = ~is_binder
    pair_mask = is_binder[:, None] & is_target[None, :]
    contact = ((dist < opt.contact_distance) & pair_mask).any(axis=1)
    return np.asarray(contact, dtype=np.float32)[np.asarray(chain_index) == BINDER_CHAIN]


def _binder_interface_stats(result, chain_index: np.ndarray) -> tuple:
    """(per-binder-residue 0/1 contact indicator (num_aa,), summary dict)."""
    binder_freq = _single_sample_binder_contact(result, chain_index)
    assert binder_freq.shape[0] == num_aa, (
        f"expected {num_aa} binder residues, got {binder_freq.shape[0]} -- the 'binder is "
        "chain 0' assumption may not hold")
    idx = np.arange(num_aa)
    total = float(binder_freq.sum())
    if total <= 1e-8:
        return binder_freq, dict(
            contact_total=0.0, contact_mean_pos=float("nan"), contact_std_pos=float("nan"),
            contact_frac_first_third=float("nan"), contact_frac_middle_third=float("nan"),
            contact_frac_last_third=float("nan"), contact_topk_residues="none",
            contact_any_frac=float((binder_freq > 0).mean()))
    mean_pos = float((idx * binder_freq).sum() / total)
    std_pos = float(np.sqrt((binder_freq * (idx - mean_pos) ** 2).sum() / total))
    third = num_aa // 3
    frac_first = float(binder_freq[:third].sum() / total)
    frac_last = float(binder_freq[num_aa - third:].sum() / total)
    top_idx = sorted(np.argsort(-binder_freq)[:opt.top_k_residues].tolist())
    return binder_freq, dict(
        contact_total=total, contact_mean_pos=mean_pos, contact_std_pos=std_pos,
        contact_frac_first_third=frac_first, contact_frac_middle_third=1.0 - frac_first - frac_last,
        contact_frac_last_third=frac_last,
        contact_topk_residues="|".join(str(i) for i in top_idx),
        contact_any_frac=float((binder_freq > 0).mean()))


def _confidence_summary(result, best_idx: int, chain_index: np.ndarray, has_target: bool) -> dict:
    """plddt and interface_pae are taken at the best_idx sample. ptm/iptm are left as
    result.ptm/result.iptm: they already reduce to one scalar per predict call (as in
    titration.py/landscape.py) and iptm counts every inter-chain pair, so it is already
    target-chain-count agnostic."""
    plddt = result.plddt if result.is_single_sample else result.plddt[best_idx]
    out = dict(plddt=float(plddt.mean()), ptm=float(result.ptm))
    if has_target:
        out["iptm"] = float(result.iptm)
        pae = np.asarray(result.pae if result.is_single_sample else result.pae[best_idx])
        is_binder = np.asarray(chain_index) == BINDER_CHAIN
        out["interface_pae"] = float(
            (pae[is_binder][:, ~is_binder].mean() + pae[~is_binder][:, is_binder].mean()) / 2)
    return out


def _aa_preference(binder_data: DesignData, key) -> tuple:
    """(num_aa, 20) AF2-ordered ProteinMPNN preference distribution for the best-of-N binder
    structure: feed the sequence-masked, target-excluded binder backbone through ProteinMPNN
    and softmax its logits (cysteine forbidden, as in titration.py)."""
    pmpnn_input = binder_data.update(
        aa=aas.translate(binder_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE)).drop_aa()
    logits = batched_pmpnn(params["mpnn"], key, pmpnn_input)["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    prob = jax.nn.softmax(
        logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / opt.pmpnn_temperature, axis=-1)
    entropy = float((-(prob * jnp.log(prob + 1e-9)).sum(axis=-1)).mean())
    consensus = aas.decode(jnp.argmax(prob, axis=-1), aas.AF2_CODE)
    return np.asarray(prob), dict(aa_entropy_mean=entropy, aa_consensus_sequence=consensus)


# ============================== per-attempt driver ==============================

def run_attempt(row_index: int, condition_index: int, attempt_index: int, entry,
                 condition_params: dict, loss_update, key, apo_binder_data=None) -> tuple:
    """One hallucination + analysis. entry=None is the apo baseline (row 0). Returns (aux dict
    for the CSV row, (num_aa,) 0/1 contact array [NaN for apo], (num_aa, 20) amino-acid
    preference, the best-of-N binder-only DesignData [the apo one is the RMSD reference])."""
    name = "apo" if entry is None else entry["name"]
    has_target = entry is not None
    halluc_key, repredict_key, pmpnn_key = jax.random.split(key, 3)

    sequence, loss_val, halluc_aux, design, init_sequence = hallucinate(
        row_index, condition_params, loss_update, halluc_key)
    tag = f"{row_index:04d}_{name}_a{attempt_index}"
    if design is not None and opt.save_blueprints == "True":
        design.save_pdb(f"{opt.out_path}/blueprints/{tag}.pdb")

    def _full_spec(hard_seq: str) -> JoltzSpec:
        spec = JoltzSpec().add_protein(hard_seq)
        return _add_target(spec, entry) if has_target else spec

    # structure Boltz gives the hardened INITIAL sequence (after salad -> ProteinMPNN in
    # salad_pmpnn mode, the Gumbel start in random mode), before any hallucination step
    init_aux = {}
    if opt.save_init == "True":
        init_hard = aas.decode(np.asarray(jnp.argmax(init_sequence, axis=-1)), aas.AF2_CODE)
        init_pred = predict_fn(jax.random.fold_in(key, 101), _full_spec(init_hard))
        init_result = init_pred.result
        init_best = (0 if init_result.is_single_sample
                     else int(jnp.argmin(init_result.pae.mean(axis=(-1, -2)))))
        init_pred.save_pdb(f"{opt.out_path}/init/{tag}_init.pdb", sample_index=init_best)
        init_plddt = (init_result.plddt if init_result.is_single_sample
                      else init_result.plddt[init_best])
        init_aux = dict(init_sequence=init_hard, init_plddt=float(init_plddt.mean()))

    # final full-atom re-prediction of the hardened binder (mosaic_scan.py's _repredict)
    hard_sequence = aas.decode(np.asarray(jnp.argmax(sequence, axis=-1)), aas.AF2_CODE)
    pred = predict_fn(repredict_key, _full_spec(hard_sequence))
    result = pred.result
    best_idx = 0 if result.is_single_sample else int(jnp.argmin(result.pae.mean(axis=(-1, -2))))

    chain_index = np.asarray(result.chain_index)
    binder_data = _binder_slice(result)

    aux = dict(row_index=row_index, condition_index=condition_index, attempt_index=attempt_index,
               condition_name=name,
               category=(entry["category"] if has_target else "apo"),
               target_kind=(entry["target_kind"] if has_target else "none"),
               target_repr=((entry.get("ccd") or entry.get("smiles") or entry.get("sequence"))
                            if has_target else "none"),
               pairs_with=((entry.get("pairs_with") or "none") if has_target else "none"),
               catalog_id=(entry.get("catalog_id", "none") if has_target else "none"),
               heavy_atoms=(entry.get("heavy_atoms", "none") if has_target else "none"),
               init_mode=opt.init_mode, hallucination_steps=opt.hallucination_steps,
               num_aa=num_aa, num_samples=opt.num_samples, num_recycle=opt.num_recycle,
               num_sampling_steps=opt.num_sampling_steps, contact_distance=opt.contact_distance,
               halluc_loss=float(loss_val), halluc_recovery=float(halluc_aux["recovery"]),
               binder_sequence=hard_sequence)
    aux.update(init_aux)
    aux.update(_confidence_summary(result, best_idx, chain_index, has_target))
    aux.update(_dssp_composition(binder_data))

    # mosaic-style scoring + success filter on a single-sample re-prediction
    score_pred = score_predict_fn(jax.random.fold_in(key, 102), _full_spec(hard_sequence))
    aux.update(mosaic_scores(sequence, pmpnn_key, score_pred.result, entry))
    if aux.get("passed") == "True":
        score_pred.save_pdb(f"{opt.out_path}/success/{tag}.pdb")

    contact_freq = np.full((num_aa,), np.nan, dtype=np.float32)
    if has_target:
        contact_freq, stats = _binder_interface_stats(result, chain_index)
        aux.update(stats)

    if apo_binder_data is not None:
        aux["rmsd_to_apo"] = _rmsd_to_apo(binder_data, apo_binder_data)

    aa_prob, aa_stats = _aa_preference(binder_data, pmpnn_key)
    aux.update(aa_stats)
    aux.update(_class_fractions(aa_prob, weights=None))
    if has_target and np.nansum(contact_freq) > 1e-6:
        aux.update(_class_fractions(
            aa_prob, weights=np.nan_to_num(contact_freq), prefix="interface_"))

    if opt.save_predictions == "True":
        pred.save_pdb(f"{opt.out_path}/predictions/{row_index:04d}_{name}_a{attempt_index}.pdb",
                      sample_index=best_idx)

    aux["status"] = "ok"
    return aux, contact_freq, aa_prob, binder_data


# ============================== output setup / resume ==============================

RESULT_KEYS = [
    "row_index", "condition_index", "attempt_index", "condition_name", "category",
    "target_kind", "target_repr", "pairs_with", "catalog_id", "heavy_atoms",
    "init_mode", "hallucination_steps", "num_aa", "num_samples", "num_recycle",
    "num_sampling_steps", "contact_distance",
    "halluc_loss", "halluc_recovery", "binder_sequence", "init_sequence", "init_plddt",
    "mosaic_plddt", "mosaic_ipsae", "mosaic_binder_target_pae", "mosaic_target_binder_pae",
    "mosaic_recovery", "passed",
    "plddt", "ptm", "iptm", "interface_pae",
    "contact_total", "contact_mean_pos", "contact_std_pos",
    "contact_frac_first_third", "contact_frac_middle_third", "contact_frac_last_third",
    "contact_topk_residues", "contact_any_frac",
    "dssp_L", "dssp_H", "dssp_E", "dssp_hard",
    "rmsd_to_apo",
    "aa_entropy_mean", "aa_consensus_sequence",
] + [f"frac_{c}" for c in AA_CLASSES] + [f"interface_frac_{c}" for c in AA_CLASSES] + [
    "status", "error",
]

results = ScoreCSV(f"{opt.out_path}/results.csv", keys=RESULT_KEYS, append=(opt.resume == "True"))
# designs passing mosaic_scan.py's success filter (MOSAIC_PASS), same schema as results.csv
success_results = ScoreCSV(f"{opt.out_path}/success.csv", keys=RESULT_KEYS,
                           append=(opt.resume == "True"))

contact_path = f"{opt.out_path}/contact_frequency.npy"
aa_path = f"{opt.out_path}/aa_preference.npy"
if opt.resume == "True" and os.path.exists(contact_path) and os.path.exists(aa_path):
    CONTACT = np.load(contact_path)
    AAPREF = np.load(aa_path)
    assert CONTACT.shape == (N_ROWS, num_aa) and AAPREF.shape == (N_ROWS, num_aa, 20)
    done = set(np.where(~np.all(np.isnan(AAPREF.reshape(N_ROWS, -1)), axis=1))[0].tolist())
else:
    CONTACT = np.full((N_ROWS, num_aa), np.nan, dtype=np.float32)
    AAPREF = np.full((N_ROWS, num_aa, 20), np.nan, dtype=np.float32)
    done = set()
print(f"Resuming: {len(done)}/{N_ROWS} rows already complete." if done else
      f"Starting fresh: {N_ROWS} rows to compute.")


def _row_key(row_index: int):
    return jax.random.fold_in(jax.random.key(opt.seed), row_index)


def _save_arrays():
    np.save(contact_path, CONTACT)
    np.save(aa_path, AAPREF)


# ============================== main loop ==============================
#
# The apo baseline (row 0) is always (re-)computed first, even on --resume where it is already
# marked done: its binder structure only exists in memory for this process and every other
# row's RMSD-to-apo needs it. Recomputation is deterministic (same seed -> same keys); its
# CSV/npy row is only re-WRITTEN if it was not already done.

apo_params, apo_loss_update = build_condition(None)
try:
    apo_aux, apo_contact, apo_aa_prob, apo_binder_data = run_attempt(
        0, 0, 0, None, apo_params, apo_loss_update, _row_key(0), apo_binder_data=None)
except Exception as ex:
    raise RuntimeError("apo baseline failed; cannot continue (every other row depends on it "
                        "for RMSD comparison)") from ex
if 0 not in done:
    results.write_line(apo_aux)
    CONTACT[0] = apo_contact
    AAPREF[0] = apo_aa_prob
    _save_arrays()
    done.add(0)
print(f"[ok] row 0000 apo: plddt={apo_aux['plddt']:.3f}")

# conditions outer, attempts inner: build_condition (differentiable input + jitted loss) runs
# once per condition and is reused across its attempts.
rows_by_condition = {}
for row in ROWS:
    if row["row_index"] != 0:
        rows_by_condition.setdefault(row["condition_index"], []).append(row)

for condition_index, entry in enumerate(MANIFEST):
    if condition_index == 0:
        continue
    todo = [r for r in rows_by_condition[condition_index] if r["row_index"] not in done]
    name = entry["name"]
    if not todo:
        print(f"[skip] {name}: all {opt.attempts_per_target} attempts already complete")
        continue
    condition_params, loss_update = build_condition(entry)
    print(f"=== {name} ({entry['category']}): {len(todo)} attempt(s) ===")
    for row in todo:
        row_index, attempt_index = row["row_index"], row["attempt_index"]
        try:
            aux, contact_freq, aa_prob, _ = run_attempt(
                row_index, condition_index, attempt_index, entry, condition_params,
                loss_update, _row_key(row_index), apo_binder_data=apo_binder_data)
            results.write_line(aux)
            if aux.get("passed") == "True":
                success_results.write_line(aux)
            CONTACT[row_index] = contact_freq
            AAPREF[row_index] = aa_prob
            print(f"[ok] row {row_index:04d} {name} attempt {attempt_index}: "
                  f"plddt={aux['plddt']:.3f} contact_total={aux.get('contact_total', 'n/a')}")
        except Exception as ex:
            print(f"[FAILED] row_index={row_index} {name} attempt {attempt_index}: {ex}",
                  file=sys.stderr)
            results.write_line(dict(
                row_index=row_index, condition_index=condition_index,
                attempt_index=attempt_index, condition_name=name,
                category=entry["category"], status="failed",
                error=_safe_token(str(ex), max_len=300)))
        _save_arrays()


print()
print("=" * 70)
print(f"Ligand-bias analysis complete. {NUM_CONDITIONS} conditions, "
      f"{opt.attempts_per_target} attempts/target, N_ROWS={N_ROWS} rows "
      f"(init_mode={opt.init_mode}).")
print(f"  {opt.out_path}/results.csv             one row per (condition, attempt)")
print(f"  {opt.out_path}/ligands.json             manifest + run options")
print(f"  {opt.out_path}/contact_frequency.npy    (N_ROWS, {num_aa}) float32, NaN = apo/failed")
print(f"  {opt.out_path}/aa_preference.npy        (N_ROWS, {num_aa}, 20) float32, NaN = failed")
print(f"  {opt.out_path}/predictions/*.pdb        final re-predicted complex per attempt")
print("=" * 70)
