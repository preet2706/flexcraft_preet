"""Titration scan of ProteinMPNN-averaging initialization settings.

Built on `flexcraft.pipelines.experimental.landscape`'s averaged-hallucination-init
machinery (--avg_init / --avg_mode bias|same_dssp|partial / --init_temperature). That
script showed averaging collapses Boltz to an all-helix fold at init_temperature=1.0, and
that 0.4 recovers single-backbone-like sharpness -- but the transition itself was never
mapped. This script maps it: for every (mode, secondary-structure bias, temperature,
[partial: start_steps]) grid point, it samples a batch of blueprints, averages their
ProteinMPNN distributions, sharpens, runs ONE Boltz2 forward pass (no hallucination), and
records the averaged distribution + predicted structure + metrics.

The grid (see parse_options below for the CLI flags that override these):
    init_temperature   0.1 .. 1.0 step 0.1                              (10 values)
    secondary bias      L=0.2 fixed, E 0.0 .. 0.8 step 0.1, H = 0.8-E    (9 values)
    avg_mode            bias | same_dssp | partial                      (3 modes)
    start_steps         25,50,100,150,200,250,300 (partial only)        (7 values)
    replicate           0 .. num_replicates-1 (default 1 -> unchanged)  (--num_replicates)

Each replicate is an independently resampled blueprint batch -- a fresh salad draw (and,
for same_dssp/partial, a freshly drawn hard DSSP string, and for partial a freshly sampled
parent too), not a reuse of another replicate's randomness. Runtime scales linearly with
--num_replicates (each one repeats the full batch/temperature split below). The default
of 1 reproduces the original single-replicate grid exactly, including its RNG seeds.

Architecture note: init_temperature is applied strictly AFTER averaging, so the sampled
blueprints and their ProteinMPNN log-probs do not depend on it at all. Each blueprint
batch is therefore sampled exactly ONCE and its raw (B, L, 20) log-probs cached; all 10
temperatures are then swept as cheap post-processing (one extra Boltz forward each)
instead of 10 fresh ~11-minute salad trajectories. Same trick for `partial`: the parent
backbone does not depend on start_steps, so it is sampled once per (bias, replicate) and
reused across all 7 start_steps. This is the difference between ~14 hours and ~6 days per
replicate.

Known limitation: `same_dssp` and `partial` are additionally conditioned on a *hard* DSSP
string drawn from the same (L,H,E) as the soft bias (so the two channels reinforce, not
fight), while `bias` mode is soft-bias-only by construction. So `bias` vs `same_dssp` at
the same E conflates "independent vs shared blueprint" with "hard string present or not".
The bias axis itself is clean (the soft dssp_mean channel is active in all three modes),
and `same_dssp` vs `partial` at equal E is a clean pairwise comparison (same string, only
the diffusion relationship between members differs).

Output layout:
    <out_path>/distributions.npy   (N, num_aa, 20) float32, NaN = not yet computed
    <out_path>/results.csv         one row per grid point; `dist_index` -> npy row
    <out_path>/grid.json           axis definitions, offsets, per-row grid metadata
    <out_path>/blueprints/<batch_id>/member_*.pdb   once per BLUEPRINT BATCH, not per point
    <out_path>/blueprints/partial_parents/parent_e*.pdb
    <out_path>/predictions/<dist_index>_<batch_id>_T<temp>.pdb

Navigate the result with:

    import json, numpy as np, pandas as pd
    D    = np.load("<out_path>/distributions.npy")   # (N, L, 20), NaN = not computed
    df   = pd.read_csv("<out_path>/results.csv")
    grid = json.load(open("<out_path>/grid.json"))

    # one grid point -> its (L, 20) averaged distribution (replicate=0 disambiguates
    # when --num_replicates > 1; the column is always present, always 0 by default)
    row = df.query("avg_mode=='bias' and e_bias==0.4 and replicate==0 and init_temperature==0.4").iloc[0]
    p = D[int(row.dist_index)]

    # temperature titration curve for one replicate of one bias point
    sub = df.query("avg_mode=='bias' and e_bias==0.4 and replicate==0").sort_values("init_temperature")

    # all replicates of that same curve, for a spread/error-bar plot
    reps = df.query("avg_mode=='bias' and e_bias==0.4").groupby("init_temperature")["dssp_identity"].agg(["mean", "std"])

    # completion mask (survives a killed/resumed run)
    done_mask = ~np.isnan(D[:, 0, 0])
"""

import os
import sys
import json
import hashlib
import random
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import salad.inference as si
from salad.inference.utils.options import parse_dssp

import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.utils import parse_options, data_from_salad
from flexcraft.files import ScoreCSV
from flexcraft.structure.boltz import Joltz2, JoltzResult, JoltzSpec
from flexcraft.data.data import DesignData
from flexcraft.pipelines.utils import bias_dssp
from flexcraft.structure.metrics import RMSD


def _float_list(s):
    return [float(v) for v in s.split(",") if v.strip() != ""]

def _int_list(s):
    return [int(v) for v in s.split(",") if v.strip() != ""]


# same denoising step as landscape.py -- unchanged, salad requires it hk.transform'ed
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


opt = parse_options(
    "Titration scan over ProteinMPNN-averaging initialization settings "
    "(no hallucination -- one Boltz forward pass per grid point).",
    num_aa="100",
    out_path="runs/titration/",
    boltz_path="./params/boltz/",
    salad_params="params/salad/default_vp_scaled-200k.jax",
    pmpnn_params="params/pmpnn/v_48_030.pkl",
    # --- the grid -----------------------------------------------------------
    modes="bias,same_dssp,partial",
    e_grid="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
    l_bias=0.2,                       # H is derived per point: H = 1 - l_bias - E
    temp_grid="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    start_steps_grid="25,50,100,150,200,250,300",
    num_replicates=1,                 # independent blueprint-batch replicates per grid cell
    init_batch_size=10,
    # --- averaging / sampling knobs, held fixed across the grid -------------
    avg_space="prob",                 # "prob" | "logit", matches landscape.py
    temperature=1.0,                  # pre-averaging per-member pmpnn temperature
    std_buffer=3.0,
    num_steps=500,
    out_steps=400,
    boltz_samples=4,
    boltz_recycles=4,
    share_dssp_across_modes="True",   # same hard string for same_dssp & partial at equal E
    # --- execution ------------------------------------------------------------
    save_blueprints="True",
    save_predictions="True",
    resume="False",
    seed=42,
)

MODE_ORDER = ["bias", "same_dssp", "partial"]
MODES = [m.strip() for m in opt.modes.split(",") if m.strip()]
assert MODES and set(MODES) <= set(MODE_ORDER), \
    f"unknown mode(s) in --modes {opt.modes!r}; expected a subset of {MODE_ORDER}"
active_modes = [m for m in MODE_ORDER if m in MODES]

E_GRID = _float_list(opt.e_grid)
T_GRID = _float_list(opt.temp_grid)
S_GRID = _int_list(opt.start_steps_grid) if "partial" in active_modes else []
assert all(0.0 <= e <= 1.0 for e in E_GRID), "--e_grid values must be in [0, 1]"
assert all(t > 0.0 for t in T_GRID), "--temp_grid values must be > 0"
assert all(0 < s < opt.out_steps for s in S_GRID), \
    f"--start_steps_grid values must lie strictly within (0, {opt.out_steps})"
assert opt.num_replicates >= 1, "--num_replicates must be >= 1"
NR = opt.num_replicates
assert opt.init_batch_size >= 2, "--init_batch_size must be >= 2 for averaging to mean anything"
for e in E_GRID:
    if e > 1.0 - opt.l_bias + 1e-9:
        print(f"WARNING: e_bias={e} + l_bias={opt.l_bias} exceeds 1.0; "
              "h_bias will be clamped to 0 for this grid point.")


def cloud_std_default(num_aa):
    minval = num_aa ** 0.4
    return minval + np.random.rand() * opt.std_buffer


# --- deterministic, order-independent RNG -----------------------------------
# Unlike landscape.py's stateful Keygen, this script must be resumable: a blueprint batch
# re-sampled after a --resume has to come out byte-identical to the original run, whether
# it runs first or last. So every draw is derived purely from its grid coordinates via a
# hash, never from how many draws happened before it in this process.

def _batch_seed(tag_prefix, e_index, r_index, start_steps) -> int:
    """Stable 32-bit seed for one blueprint batch's host-side RNG. The replicate index
    is only folded into the tag when nonzero, so replicate 0 hashes to EXACTLY the same
    seed the pre-replicate script used (--num_replicates=1 is then byte-identical to the
    original single-replicate grid), while replicate 1, 2, ... draw independently."""
    suffix = f"|{r_index}" if r_index else ""
    tag = f"{opt.seed}|host|{tag_prefix}|{e_index}|{start_steps}{suffix}"
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)

def _seed_host_rng(s: int):
    """salad's random_dssp/biased_random_dssp draw from BOTH np.random (randint) and
    python's random (random(), shuffle); cloud_std_default draws from np.random. Seed
    both, or the segment shuffle stays non-reproducible even though lengths are."""
    np.random.seed(s % (2**32))
    random.seed(s)

def _batch_key(mode, e_index, r_index, start_steps, tag) -> jax.Array:
    """Deterministic JAX PRNG key for one salad/ProteinMPNN draw within a batch.
    Same replicate-compatibility rule as _batch_seed: r_index=0 reproduces the
    pre-replicate script's keys exactly."""
    suffix = f"|{r_index}" if r_index else ""
    full_tag = f"{opt.seed}|jax|{mode}|{e_index}|{start_steps}{suffix}|{tag}"
    seed_int = int(hashlib.sha256(full_tag.encode()).hexdigest()[:8], 16)
    return jax.random.key(seed_int)

def _grid_key(dist_index: int) -> jax.Array:
    """Deterministic JAX key for one grid point's Boltz forward + metrics."""
    return jax.random.fold_in(jax.random.key(opt.seed), dist_index)


def biased_random_dssp(count, l, h, e, p=0.0, p_keep_loop=1.0, return_string=False):
    """salad.inference.utils.options.random_dssp with the (loop, helix, sheet)
    composition supplied directly, instead of drawn from random_dssp_mean() (which
    samples helix ~ U(0,1), sheet = 1-helix, with no exposed way to bias it toward
    strand -- there is no seam to inject through, hence vendoring rather than patching).

    Body below is a verbatim copy of random_dssp's segment-layout logic
    (salad/inference/utils/options.py:158-213) with only the composition swapped in for
    the first line. If salad changes its layout rules, re-copy this rather than
    re-deriving it. Draws from BOTH np.random and python's `random`, exactly as the
    original -- seed both via _seed_host_rng before calling.
    """
    total = l + h + e
    if total > 0:
        l, h, e = l / total, h / total, e / total
    loop, helix, sheet = l, h, e
    # ---- verbatim from salad's random_dssp from here on ----
    helix = int(helix * count)
    if helix < 6:
        helix = 0
    min_helix_count = 1
    max_helix_count = helix // 6
    min_helix_count = min(min_helix_count, max_helix_count)
    if helix == 0:
        num_helices = 0
    elif min_helix_count == max_helix_count:
        num_helices = min_helix_count
    else:
        num_helices = np.random.randint(min_helix_count, max_helix_count)
    sheet = int(sheet * count)
    if sheet < 8:
        sheet = 0
    loop = count - (helix + sheet)
    min_sheet_count = 2
    max_sheet_count = sheet // 4
    if sheet == 0:
        num_sheets = 0
    elif min_sheet_count == max_sheet_count:
        num_sheets = min_sheet_count
    else:
        num_sheets = np.random.randint(min_sheet_count, max_sheet_count)
    helices = [6 for _ in range(num_helices)]
    sheets = [4 for _ in range(num_sheets)]
    loops = [0 for _ in range(num_helices + num_sheets + 1)]
    while sum(sheets) < sheet:
        index = np.random.randint(num_sheets)
        sheets[index] += 1
    while sum(helices) < helix:
        index = np.random.randint(num_helices)
        helices[index] += 1
    while sum(loops) < loop:
        index = np.random.randint(0, len(loops))
        loops[index] += 1
    helices = ["_" + "H" * (num - 2) + "_" if random.random() > p else "_" * num for num in helices]
    sheets = ["_" + "E" * (num - 2) + "_" if random.random() > p else "_" * num for num in sheets]
    loops = ["L" * num if random.random() < p_keep_loop else "_" * num for num in loops]
    structured = helices + sheets
    random.shuffle(structured)
    dssp_string = loops[0] + "".join([s + lp for s, lp in zip(structured, loops[1:])])
    # ---- end verbatim ----
    assert len(dssp_string) == count, f"biased_random_dssp: {len(dssp_string)} != {count}"
    dssp = parse_dssp(dssp_string)
    if return_string:
        return dssp, dssp_string
    return dssp


# ============================== setup (mirrors landscape.py) ==============================

salad_config, salad_params = si.make_salad_model("default_vp_scaled", opt.salad_params)
num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(opt.num_aa)

# quantization warning: salad's segment-layout rules zero out any segment shorter than its
# own minimum, so a nonzero bias can silently vanish from the HARD dssp string at small
# num_aa (the soft dssp_mean bias is unaffected either way).
for e in E_GRID:
    h_val = max(0.0, 1.0 - opt.l_bias - e)
    if 0 < e < 8.0 / num_aa:
        print(f"WARNING: e_bias={e} at num_aa={num_aa} quantizes to 0 strand residues "
              f"in the HARD dssp string (int({e}*{num_aa}) < 8); same_dssp/partial's "
              "string will show no E even though the soft bias is unaffected.")
    if 0 < h_val < 6.0 / num_aa:
        print(f"WARNING: h_bias={h_val:.3f} (from e_bias={e}) at num_aa={num_aa} "
              "quantizes to 0 helix residues in the HARD dssp string.")

BASE_DATA, INIT_PREV = si.data.from_config(
    salad_config, num_aa=num_aa, residue_index=resi,
    chain_index=chain, cyclic_mask=cyclic_mask)
UNSPEC_DSSP = np.full((num_aa,), 3, dtype=np.int32)   # salad's own "unspecified" code

salad_step = model_step(salad_config)
sampler = si.Sampler(salad_step, num_steps=opt.num_steps, out_steps=opt.out_steps)

_pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_params, eps=0.05, split_params=True)
params = dict(mpnn=pmpnn_params)

joltz_model = Joltz2(cache=opt.boltz_path)
joltz, params["joltz"] = joltz_model.evaluator(num_recycle=opt.boltz_recycles)
joltz_spec = JoltzSpec().add_protein("X" * num_aa)
params["joltz_input"], _writer = joltz_spec.to_input(pad=False)


def batch_mpnn(mpnn):
    def _mpnn_map(params, key, data):
        if len(data["atom_positions"].shape) == 4:
            num_samples = data["atom_positions"].shape[0]
            xx = {
                key: jnp.repeat(data[key][None, :], num_samples, axis=0)
                for key in ["atom_mask", "mask", "aa", "residue_index", "chain_index", "batch_index"]
            }
            xx["atom_positions"] = data["atom_positions"]
            keys = jax.random.split(key, num_samples)
            return jax.vmap(mpnn, (None, 0, 0), 0)(params, keys, xx)
        else:
            return mpnn(params, key, data)
    return _mpnn_map

# one jitted MPNN used both for the B=init_batch_size blueprint average (raw_mpnn_logits)
# and the B=boltz_samples self-consistency pass inside metrics() -- two traces (different
# B), both cached across the whole run.
batched_pmpnn = jax.jit(batch_mpnn(_pmpnn))


def make_data(l, h, e, dssp_condition, cloud_std):
    """Fresh salad input dict for one trajectory. ALWAYS sets dssp_condition, and (since
    l_bias > 0 in the default grid) always triggers bias_dssp's dssp_mean/dssp_mean_mask,
    so every call across all three modes shares the same key set. This matters because
    Sampler.step is jax.jit'd over this dict -- a key appearing in one mode and not
    another would force a full salad recompile per mode instead of one for the run."""
    d = dict(BASE_DATA)
    d = bias_dssp(d, L=l, H=h, E=e)
    d["dssp_condition"] = jnp.asarray(dssp_condition)
    d["cloud_std"] = cloud_std
    return d


# ============================== grid / offsets ==============================
#
# Replicate sits between e_bias and (start_steps/)temperature: dist_index groups mode ->
# e_bias -> replicate -> [start_steps ->] temperature, so all 10 temperatures of one
# replicate stay contiguous (the axis the batch/temperature cache sweeps for free) and
# NR=1 makes every formula below reduce exactly to the pre-replicate one (r_index is
# always 0, contributing nothing to the offset).

NE, NT, NS = len(E_GRID), len(T_GRID), len(S_GRID)
offsets = {}
_cursor = 0
for _m in active_modes:
    offsets[_m] = _cursor
    _cursor += NE * NR * NS * NT if _m == "partial" else NE * NR * NT
N = _cursor

def dist_index(mode, e_index, r_index, s_index, t_index):
    if mode == "partial":
        return (offsets["partial"] + e_index * (NR * NS * NT)
                + r_index * (NS * NT) + s_index * NT + t_index)
    return offsets[mode] + e_index * (NR * NT) + r_index * NT + t_index

GRID = []
for mode in active_modes:
    for e_index, e in enumerate(E_GRID):
        h_val = max(0.0, 1.0 - opt.l_bias - e)
        for r_index in range(NR):
            s_range = enumerate(S_GRID) if mode == "partial" else [(0, 0)]
            for s_index, S in s_range:
                for t_index, T in enumerate(T_GRID):
                    GRID.append(dict(
                        dist_index=dist_index(mode, e_index, r_index, s_index, t_index),
                        avg_mode=mode, e_bias=e, h_bias=h_val, l_bias=opt.l_bias,
                        replicate=r_index,
                        start_steps=(S if mode == "partial" else 0),
                        init_temperature=T))
GRID.sort(key=lambda r: r["dist_index"])
assert len(GRID) == N

os.makedirs(opt.out_path, exist_ok=True)
os.makedirs(f"{opt.out_path}/blueprints/partial_parents", exist_ok=True)
os.makedirs(f"{opt.out_path}/predictions", exist_ok=True)

with open(f"{opt.out_path}/grid.json", "w") as f:
    json.dump(dict(
        N=N, num_aa=num_aa, init_batch_size=opt.init_batch_size, num_replicates=NR,
        modes=active_modes, e_grid=E_GRID, t_grid=T_GRID, s_grid=S_GRID,
        l_bias=opt.l_bias, seed=opt.seed, offsets=offsets, grid=GRID,
    ), f, indent=2)


# ============================== reporting (mirrors landscape.py) ==============================

def metrics(sequence, key, result: JoltzResult):
    data = result.to_data(return_samples=True)
    dssp = data.p_dssp
    pmpnn_input = data.update(aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    logits = batched_pmpnn(params["mpnn"], key, pmpnn_input.drop_aa())["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not result.is_single_sample:
        logits = logits.mean(axis=0)
    prob = jax.lax.stop_gradient(jax.nn.softmax(
        logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    entropy = -(logits * prob).sum(axis=-1).mean()
    plddt = result.plddt
    pae = 32 * result.pae
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)
    return dict(
        entropy=entropy,
        plddt=plddt.mean(),
        recovery=(sequence * prob).sum(axis=-1).mean(),
        contacts=result.contact_score(num_contacts=25, min_resi_distance=10),
        pae=pae.mean(),
        average_num_nonzero=(sequence > 0.01).sum(axis=-1).mean(),
        L=dssp["L"], H=dssp["H"], E=dssp["E"],
        ptm=result.ptm,
    )

def dssp_to_string(dssp_array) -> str:
    code = np.array(list("LHE"))
    return "".join(code[np.asarray(dssp_array)])

def _structure_comparison(design: DesignData, predicted_data: DesignData) -> dict:
    salad_dssp = np.asarray(design.dssp)
    predicted_dssp = np.asarray(predicted_data.dssp)
    return dict(
        rmsd_to_blueprint=RMSD(atoms=["CA"])(predicted_data, design),
        dssp_salad=dssp_to_string(salad_dssp),
        dssp_predicted=dssp_to_string(predicted_dssp),
        dssp_identity=(salad_dssp == predicted_dssp).mean(),
    )

def _structure_comparison_batch(designs, predicted_data: DesignData) -> dict:
    per_member = [_structure_comparison(d, predicted_data) for d in designs]
    if len(per_member) == 1:
        return per_member[0]
    rmsds = np.array([float(p["rmsd_to_blueprint"]) for p in per_member])
    best = int(rmsds.argmin())
    result = dict(per_member[best])
    p_dssp = [d.p_dssp for d in designs]
    result.update(
        rmsd_to_blueprint_mean=np.float32(rmsds.mean()),
        dssp_identity_mean=np.float32(np.mean([float(p["dssp_identity"]) for p in per_member])),
        blueprint_index=np.float32(best),
        batch_L=np.float32(np.mean([float(x["L"]) for x in p_dssp])),
        batch_H=np.float32(np.mean([float(x["H"]) for x in p_dssp])),
        batch_E=np.float32(np.mean([float(x["E"]) for x in p_dssp])),
    )
    return result

def _init_sharpness(sequence) -> dict:
    s = np.asarray(sequence)
    return dict(
        init_num_nonzero=np.float32((s > 0.01).sum(axis=-1).mean()),
        init_max_prob=np.float32(s.max(axis=-1).mean()),
    )

def _batch_spread(designs, parent=None) -> dict:
    reference = parent if parent is not None else designs[0]
    others = designs if parent is not None else designs[1:]
    if not others:
        return {}
    rmsds = [float(RMSD(atoms=["CA"])(d, reference)) for d in others]
    return dict(batch_rmsd_to_ref=np.float32(np.mean(rmsds)))

def _grid_provenance(mode, l, h, e, r_index, start_steps, init_temperature, di, batch_id, dssp_string) -> dict:
    """Numeric fields are np.float32 because _report_result calls .mean() on every
    non-str value -- a plain Python int/float would raise."""
    return dict(
        avg_mode=mode,
        batch_id=batch_id,
        l_bias=np.float32(l), h_bias=np.float32(h), e_bias=np.float32(e),
        bias_lhe=f"{l:.3f}|{h:.3f}|{e:.3f}",
        replicate=np.float32(r_index),
        start_steps=np.float32(start_steps),
        init_temperature=np.float32(init_temperature),
        init_batch=np.float32(opt.init_batch_size),
        dist_index=np.float32(di),
        dssp_target=dssp_string,
    )

def _report_result(attempt, stage, writer: ScoreCSV, sequence, aux):
    line_info = dict(attempt=attempt, stage=stage,
                     sequence=aas.decode(jnp.argmax(sequence, axis=-1), aas.AF2_CODE))
    for key in writer.keys:
        if key in aux:
            value = aux[key]
            line_info[key] = value if isinstance(value, str) else float(value.mean())
    writer.write_line(line_info)


AXIS_KEYS = ["dist_index", "avg_mode", "batch_id", "init_temperature",
             "l_bias", "h_bias", "e_bias", "bias_lhe", "replicate", "start_steps",
             "init_batch", "dssp_target"]
COMMON_KEYS = ["plddt", "entropy", "recovery", "contacts", "pae",
               "average_num_nonzero", "L", "H", "E", "ptm"]
STRUCT_KEYS = ["rmsd_to_blueprint", "dssp_salad", "dssp_predicted", "dssp_identity",
               "rmsd_to_blueprint_mean", "dssp_identity_mean", "blueprint_index",
               "batch_L", "batch_H", "batch_E"]
INIT_KEYS = ["init_num_nonzero", "init_max_prob", "batch_rmsd_to_ref"]
results_keys = ["attempt", "stage", "init", "sequence"] + AXIS_KEYS + COMMON_KEYS + STRUCT_KEYS + INIT_KEYS


def _open_results(path, keys, resume):
    """ScoreCSV.__init__ truncates (files/csv.py opens with 'wt'). To resume, snapshot the
    existing data lines first, let ScoreCSV rewrite the header, then append them back --
    done here rather than by changing ScoreCSV, which every other pipeline depends on."""
    preserved, done_set = [], set()
    if resume and os.path.exists(path):
        with open(path, "rt") as f:
            lines = f.read().splitlines()
        if lines:
            header = lines[0].split(",")
            if "dist_index" in header:
                di_col = header.index("dist_index")
                for line in lines[1:]:
                    fields = line.split(",")
                    if len(fields) != len(header):
                        continue  # torn last line from a killed job
                    preserved.append(line)
                    try:
                        done_set.add(int(float(fields[di_col])))
                    except ValueError:
                        pass
    csv = ScoreCSV(path, keys=keys)
    if preserved:
        with open(path, "at") as f:
            f.write("\n".join(preserved) + "\n")
    return csv, done_set


results, done = _open_results(f"{opt.out_path}/results.csv", results_keys, opt.resume == "True")

dist_path = f"{opt.out_path}/distributions.npy"
if opt.resume == "True" and os.path.exists(dist_path):
    DIST = np.load(dist_path)
    assert DIST.shape == (N, num_aa, 20), f"resumed npy shape {DIST.shape} != {(N, num_aa, 20)}"
else:
    DIST = np.full((N, num_aa, 20), np.nan, dtype=np.float32)

npy_done = set(np.where(~np.isnan(DIST[:, 0, 0]))[0].tolist())
if opt.resume == "True" and done != npy_done:
    print(f"WARNING: results.csv done ({len(done)}) != distributions.npy done "
          f"({len(npy_done)}); using their intersection.")
    done = done & npy_done
print(f"Resuming: {len(done)}/{N} grid points already complete." if done else
      f"Starting fresh: {N} grid points to compute.")


# ============================== the batch/temperature split ==============================

def raw_mpnn_logits(designs, key_):
    """One vmapped ProteinMPNN pass over a blueprint batch -> raw AF2-ordered
    log-probabilities, shape (B, L, 20). No temperature, no averaging: this is exactly
    the object that gets cached, since init_temperature is applied strictly after
    averaging and does not change it."""
    rep = designs[0]
    mpnn_batch = dict(
        atom_positions=jnp.stack([jnp.asarray(d["atom_positions"]) for d in designs], axis=0),
        atom_mask=jnp.asarray(rep["atom_mask"]),
        mask=jnp.asarray(rep["mask"]),
        aa=jnp.full((num_aa,), 20, dtype=jnp.int32),
        residue_index=jnp.asarray(rep["residue_index"]),
        chain_index=jnp.asarray(rep["chain_index"]),
        batch_index=jnp.asarray(rep["batch_index"]),
    )
    logits = batched_pmpnn(params["mpnn"], key_, mpnn_batch)["logits"]
    return aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]

def average_and_sharpen(logits, init_temperature):
    """Average (B, L, 20) raw logits into one (L, 20) distribution, then sharpen.
    Mirrors landscape.py's sample_init_batch exactly given the same logits."""
    if opt.avg_space == "logit":
        seq = jax.nn.softmax(logits.mean(axis=0) / opt.temperature, axis=-1)
    else:
        seq = jax.nn.softmax(logits / opt.temperature, axis=-1).mean(axis=0)
    if init_temperature != 1.0:
        # Temperature acts on LOGITS: softmax(log p / T) == normalized p^(1/T).
        # softmax(p / T) directly would instead flatten toward uniform -- landscape.py
        # measured 20.00 nonzero / 0.065 max_prob for the wrong form vs 10.53 / 0.352.
        seq = jax.nn.softmax(jnp.log(seq + 1e-9) / init_temperature, axis=-1)
    return seq


def run_batch(mode, e_index, e, l, h, r_index, s_index, start_steps, dssp_cond, dssp_string, parent_raw):
    """Sample one blueprint batch (or reuse the parent, for `partial`), cache its raw
    ProteinMPNN logits, then sweep every not-yet-done temperature at this grid coordinate
    as cheap post-processing (one Boltz forward each). One independent call per replicate:
    the batch (and, for `partial`, the parent) is freshly sampled every time this is
    called -- nothing here is shared across replicates except the grid coordinates."""
    r_tag = f"_r{r_index}" if NR > 1 else ""
    batch_id = (f"{mode}_e{e:.2f}{r_tag}" if mode != "partial"
                else f"{mode}_e{e:.2f}{r_tag}_s{start_steps}")
    t_indices_needed = [t for t in range(NT)
                        if dist_index(mode, e_index, r_index, s_index, t) not in done]
    if not t_indices_needed:
        print(f"[skip] {batch_id}: already complete")
        return

    _seed_host_rng(_batch_seed(mode, e_index, r_index, start_steps))
    designs = []
    for m in range(opt.init_batch_size):
        member_key = _batch_key(mode, e_index, r_index, start_steps, f"m{m}")
        data_in = make_data(l, h, e, dssp_cond, cloud_std_default(num_aa))
        if mode == "partial":
            data_in["pos"] = parent_raw["pos"]
            raw = sampler(salad_params, member_key, data_in, INIT_PREV, start_steps=start_steps)
        else:
            raw = sampler(salad_params, member_key, data_in, INIT_PREV)
        designs.append(data_from_salad(raw))

    if opt.save_blueprints == "True":
        os.makedirs(f"{opt.out_path}/blueprints/{batch_id}", exist_ok=True)
        for m, d in enumerate(designs):
            d.save_pdb(f"{opt.out_path}/blueprints/{batch_id}/member_{m}.pdb")

    parent_design = data_from_salad(parent_raw) if (mode == "partial" and parent_raw is not None) else None
    spread = _batch_spread(designs, parent_design)
    raw_logits = raw_mpnn_logits(designs, _batch_key(mode, e_index, r_index, start_steps, "mpnn"))

    for t_index in t_indices_needed:
        T = T_GRID[t_index]
        di = dist_index(mode, e_index, r_index, s_index, t_index)
        try:
            sequence = average_and_sharpen(raw_logits, T)
            ks = jax.random.split(_grid_key(di), 2)
            joltz_input = params["joltz_input"].set_aa(sequence)
            result = joltz(params["joltz"]).predict(ks[0], joltz_input, num_samples=opt.boltz_samples)
            predicted = result.to_data()

            aux = metrics(sequence, ks[1], result)
            aux.update(_structure_comparison_batch(designs, predicted))
            aux.update(_init_sharpness(sequence))
            aux.update(spread)
            aux.update(_grid_provenance(mode, l, h, e, r_index, start_steps, T, di, batch_id, dssp_string))
            aux["init"] = "soft"

            _report_result(di, mode, results, sequence, aux)
            DIST[di] = np.asarray(sequence, dtype=np.float32)
            if opt.save_predictions == "True":
                predicted.save_pdb(f"{opt.out_path}/predictions/{di:05d}_{batch_id}_T{T:.2f}.pdb")
            done.add(di)
        except Exception as ex:
            print(f"[FAILED] dist_index={di} {batch_id} T={T}: {ex}", file=sys.stderr)
            fail_aux = _grid_provenance(mode, l, h, e, r_index, start_steps, T, di, batch_id, dssp_string)
            results.write_line(dict(attempt=di, stage=f"{mode}_FAILED", sequence="none", **fail_aux))

    np.save(dist_path, DIST)


# ============================== main loop ==============================

for mode in active_modes:
    for e_index, e in enumerate(E_GRID):
        h = max(0.0, 1.0 - opt.l_bias - e)
        l = opt.l_bias

        for r_index in range(NR):
            # Everything below is per-replicate: a fresh DSSP-string draw (same_dssp /
            # partial), and -- inside run_batch -- a fresh blueprint batch and (for
            # partial) a fresh parent. Nothing is reused across replicates; only the
            # RNG derivation (grid coordinates + r_index) is what makes each one
            # reproducible on its own.
            if mode == "bias":
                dssp_cond, dssp_string = UNSPEC_DSSP, "none"
            else:
                tag = "dssp" if opt.share_dssp_across_modes == "True" else mode
                _seed_host_rng(_batch_seed(tag, e_index, r_index, 0))
                dssp_cond, dssp_string = biased_random_dssp(num_aa, l, h, e, return_string=True)

            if mode != "partial":
                run_batch(mode, e_index, e, l, h, r_index, 0, 0, dssp_cond, dssp_string, None)
            else:
                need_parent = any(
                    dist_index("partial", e_index, r_index, s_index, t_index) not in done
                    for s_index in range(NS) for t_index in range(NT))
                parent_raw = None
                if need_parent:
                    _seed_host_rng(_batch_seed("partial|parent", e_index, r_index, 0))
                    parent_key = _batch_key("partial|parent", e_index, r_index, 0, "parent")
                    parent_raw = sampler(salad_params, parent_key,
                                         make_data(l, h, e, dssp_cond, cloud_std_default(num_aa)),
                                         INIT_PREV)
                    if opt.save_blueprints == "True":
                        r_tag = f"_r{r_index}" if NR > 1 else ""
                        data_from_salad(parent_raw).save_pdb(
                            f"{opt.out_path}/blueprints/partial_parents/parent_e{e:.2f}{r_tag}.pdb")
                for s_index, S in enumerate(S_GRID):
                    run_batch(mode, e_index, e, l, h, r_index, s_index, S, dssp_cond, dssp_string, parent_raw)


print()
print("=" * 70)
print(f"Titration scan complete. N={N} grid points across "
      f"{len(active_modes)} mode(s): {', '.join(active_modes)}.")
print(f"  {opt.out_path}/distributions.npy   (N, {num_aa}, 20) float32, NaN = not computed")
print(f"  {opt.out_path}/results.csv          one row per grid point, dist_index -> npy row")
print(f"  {opt.out_path}/grid.json            axis definitions + per-row grid metadata")
print()
print("Navigate with:")
print(f'''
    import json, numpy as np, pandas as pd
    D    = np.load("{opt.out_path}/distributions.npy")
    df   = pd.read_csv("{opt.out_path}/results.csv")
    grid = json.load(open("{opt.out_path}/grid.json"))

    # one grid point -> its (L, 20) averaged distribution (replicate=0 disambiguates
    # when --num_replicates > 1; the column is always present, always 0 by default)
    row = df.query("avg_mode=='bias' and e_bias==0.4 and replicate==0 and init_temperature==0.4").iloc[0]
    p = D[int(row.dist_index)]

    # temperature titration curve for one replicate of one bias point
    sub = df.query("avg_mode=='bias' and e_bias==0.4 and replicate==0").sort_values("init_temperature")

    # all replicates of that same curve, for a spread/error-bar plot
    reps = df.query("avg_mode=='bias' and e_bias==0.4").groupby("init_temperature")["dssp_identity"].agg(["mean", "std"])

    # completion mask (survives a killed/resumed run)
    done_mask = ~np.isnan(D[:, 0, 0])
''')
print("=" * 70)
