
import os
import time
from copy import deepcopy, copy
import optax
import random
import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk

import salad.inference as si
from salad.modules.utils.geometry import positions_to_ncacocb

import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.utils import Keygen, parse_options, data_from_salad
from flexcraft.files import ScoreCSV, PDBFile
from flexcraft.structure.boltz import Joltz2, JoltzInput, JoltzResult, JoltzSpec
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.hallucination.losses.distogram import blueprint_distogram, distogram_nll
from flexcraft.data.data import DesignData
from flexcraft.pipelines.utils import bias_dssp

from flexcraft.rosetta.relax import fastrelax, pair_energies
import pyrosetta as pr
from flexcraft.structure.metrics import RMSD, sc_lddt, lddt

# here, we implement a denoising step
def model_step(config):
    # get the configuration and set eval to true
    # this turns off any model components that
    # are only used during training.
    config = deepcopy(config)
    config.eval = True
    # salad is built on top of haiku, which means
    # that any functions using salad modules need
    # to be hk.transform-ed before use.
    @hk.transform
    def step(data, prev):
        # instantiate a noise generator
        noise = si.StructureDiffusionNoise(config)
        # and a denoising model
        predict = si.StructureDiffusionPredict(config)
        # apply noise
        data.update(noise(data))
        # run model
        out, prev = predict(data, prev)
        return out, prev
    # return the pure apply function generated
    # by haiku from our step
    return step.apply


opt = parse_options(
    "Protein sequence design with custom protein MPNN",
    num_aa="100",
    out_path="outputs/",
    boltz_path="./params/boltz/",
    salad_params="params/salad/default_vp_scaled-200k.jax",
    pmpnn_params="params/pmpnn/v_48_030.pkl",
    af2_params="params/af/",
    af2_model="model_1_ptm",
    num_blocks=10,
    num_designs=100,
    num_samples=20,
    std_buffer=3.0,
    temperature=1.0,
    strict="False",
    hallucinate="True",
    avg_init="False",
    init_batch_size=10,
    avg_space="prob",
    avg_mode="bias",
    init_temperature=0.4,
    partial_start_steps=200,
    l_bias=0.0,
    h_bias=0.0,
    e_bias=0.0,
    distogram_loss_weights="0.0",
    loss_ablation="False",
    ablation_weights="0.1,1.0",
    seed=42
)
def _float_list(s: str) -> list:
    return [float(v) for v in s.split(",") if v.strip() != ""]

hallucinate = opt.hallucinate == "True"
avg_init = opt.avg_init == "True"
distogram_loss_weights = _float_list(opt.distogram_loss_weights)
loss_ablation = opt.loss_ablation == "True"
ablation_weights = _float_list(opt.ablation_weights)
use_bias = (opt.l_bias > 0.0) or (opt.h_bias > 0.0) or (opt.e_bias > 0.0)
init_batch_size = opt.init_batch_size if avg_init else 1
assert init_batch_size >= 1, "--init_batch_size must be >= 1"
assert opt.avg_mode in ("bias", "partial", "same_dssp"), \
    f"unknown --avg_mode {opt.avg_mode!r}; expected bias | partial | same_dssp"
# Post-averaging sharpening only makes sense when the batch members are structurally
# unrelated (bias mode). In partial/same_dssp the members already share a fold, so the
# average stays sharp on its own and sharpening would only over-commit it.
init_temperature = opt.init_temperature if opt.avg_mode == "bias" else 1.0
if avg_init and opt.avg_mode != "bias" and opt.init_temperature != 1.0:
    print(f"NOTE: --init_temperature ignored in --avg_mode {opt.avg_mode} "
          "(batch members are structurally similar by construction).")
if avg_init and opt.avg_mode == "bias" and not use_bias:
    print("WARNING: --avg_mode bias with all biases 0.0 -- each batch member keeps its own "
          "random hard DSSP target, so the averaged distribution mixes unrelated folds.")
if use_bias:
    _bias_sum = opt.l_bias + opt.h_bias + opt.e_bias
    if not (0.9 <= _bias_sum <= 1.1):
        print(f"WARNING: --l/h/e_bias sum to {_bias_sum:.3f}. salad's dssp_mean "
              "conditioning was trained on normalized L/H/E frequencies; "
              "values summing to ~1.0 are strongly recommended.")

def cloud_std_default(num_aa):
    minval = num_aa ** 0.4
    return minval + np.random.rand() * opt.std_buffer

os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)
os.makedirs(f"{opt.out_path}/relaxed/", exist_ok=True)
os.makedirs(f"{opt.out_path}/data/", exist_ok=True)

key = Keygen(opt.seed)
# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# set up logit transform
transform = lambda center, T, do_center: transform_logits([
    toggle_transform(
        center_logits(center=center), use=do_center),
    scale_by_temperature(temperature=T),
    forbid("C", aas.PMPNN_CODE),
    norm_logits
])
# set up sampler
pmpnn_sampler = lambda center, T, do_center: sample(
    pmpnn, logit_transform=transform(center, T, do_center))

# make salad model
salad_config, salad_params = si.make_salad_model("default_vp_scaled", opt.salad_params)

# initialize salad data and prev from the num_aa specification
num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(opt.num_aa)
salad_data, init_prev = si.data.from_config(
    salad_config,
    num_aa=num_aa,
    residue_index=resi,
    chain_index=chain,
    cyclic_mask=cyclic_mask)

# Soft secondary-structure conditioning. salad reads dssp_mean/dssp_mean_mask as a
# channel independent of (and additive with) the hard per-residue dssp_condition, so
# biasing means setting this once and never setting dssp_condition.
if use_bias:
    salad_data = bias_dssp(salad_data, L=opt.l_bias, H=opt.h_bias, E=opt.e_bias)
    assert "dssp_condition" not in salad_data
    print(f"soft DSSP bias (L,H,E) = ({opt.l_bias}, {opt.h_bias}, {opt.e_bias})")

salad_step = model_step(salad_config)
# build a sampler object for sampling
sampler = si.Sampler(salad_step, out_steps=400)

# set up params
_pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_params, eps=0.05, split_params=True)
params = dict(
    mpnn=pmpnn_params
)
# set up Joltz
joltz_model = Joltz2(cache=opt.boltz_path)
joltz, params["joltz"] = joltz_model.evaluator(num_recycle=4)
joltz_spec = JoltzSpec().add_protein("X" * num_aa)
params["joltz_input"], writer = joltz_spec.to_input(pad=False)
joltz_pred = joltz_model.predictor(num_recycle=4, num_samples=4)

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

batched_pmpnn = jax.jit(batch_mpnn(_pmpnn))

def _sample_bias_batch(batch_size):
    """`bias` mode: `batch_size` independent full-diffusion backbones under the run's
    soft dssp_mean bias. Members share only a statistical L/H/E composition, so their
    folds -- and therefore what residue index i means -- are unrelated."""
    raw = []
    for _ in range(batch_size):
        if not use_bias:
            # unbiased: keep the hard random-DSSP target, redrawn per batch member
            salad_data["dssp_condition"], dssp_string = si.random_dssp(
                num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
            print("DSSP target:", dssp_string)
        salad_data["cloud_std"] = cloud_std_default(num_aa)
        raw.append(sampler(salad_params, key(), salad_data, init_prev))
    return raw, None

def _sample_same_dssp_batch(batch_size):
    """`same_dssp` mode: one random DSSP target for this design, then `batch_size`
    independent full-diffusion backbones all conditioned on it. Same secondary-structure
    blueprint, independently sampled folds."""
    salad_data["dssp_condition"], dssp_string = si.random_dssp(
        num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
    print("DSSP target (shared):", dssp_string)
    raw = []
    for _ in range(batch_size):
        # cloud_std and the RNG key are the only sources of variation between members
        salad_data["cloud_std"] = cloud_std_default(num_aa)
        raw.append(sampler(salad_params, key(), salad_data, init_prev))
    return raw, None

def _sample_partial_batch(batch_size):
    """`partial` mode: one parent backbone by full diffusion, then `batch_size` variants
    by partial diffusion of that same parent.

    salad re-noises `data["pos"]` inside the step function on every iteration, so entering
    the loop at `start_steps` with the parent's coordinates in `pos` re-noises the parent
    to sigma(1 - start_steps/500) and denoises from there. `sample()` shallow-copies its
    data argument, so every variant restarts from the parent rather than drifting."""
    salad_data["dssp_condition"], dssp_string = si.random_dssp(
        num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
    print("DSSP target (parent):", dssp_string)
    salad_data["cloud_std"] = cloud_std_default(num_aa)
    parent = sampler(salad_params, key(), salad_data, init_prev)
    raw = []
    for _ in range(batch_size):
        # fresh shallow copy so the module-level salad_data is never mutated; the parent's
        # own output is already in salad's (L, 5 + augment_size, 3) layout with real
        # augmented atoms, so no PDB round-trip or positions_to_ncacocb re-derivation
        variant_data = dict(salad_data)
        variant_data["pos"] = parent["pos"]
        variant_data["cloud_std"] = cloud_std_default(num_aa)
        raw.append(sampler(salad_params, key(), variant_data, init_prev,
                           start_steps=opt.partial_start_steps))
    return raw, parent

def _average_mpnn(designs):
    """Elementwise average of the ProteinMPNN distributions over a batch of blueprints.

    One vmapped ProteinMPNN pass over the batch. Only atom_positions differs between
    members; batch_mpnn repeats the rest, which is valid here because num_aa is fixed
    for the whole run and every member is fully sequence-masked. A plain dict is passed
    rather than a DesignData so the jitted pytree structure stays constant.
    """
    rep = designs[0]
    mpnn_batch = dict(
        atom_positions=jnp.stack(
            [jnp.asarray(d["atom_positions"]) for d in designs], axis=0),
        atom_mask=jnp.asarray(rep["atom_mask"]),
        mask=jnp.asarray(rep["mask"]),
        aa=jnp.full((num_aa,), 20, dtype=jnp.int32),
        residue_index=jnp.asarray(rep["residue_index"]),
        chain_index=jnp.asarray(rep["chain_index"]),
        batch_index=jnp.asarray(rep["batch_index"]),
    )
    logits = batched_pmpnn(params["mpnn"], key(), mpnn_batch)["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if opt.avg_space == "logit":
        return jax.nn.softmax(logits.mean(axis=0) / opt.temperature, axis=-1)
    return jax.nn.softmax(logits / opt.temperature, axis=-1).mean(axis=0)

def sample_init_batch(batch_size):
    """Generate a batch of blueprints per `--avg_mode` and average their ProteinMPNN
    distributions into one starting sequence.

    Returns:
        designs: list of `batch_size` DesignData blueprints.
        sequence: (L, 20) float32 amino acid distribution, rows summing to 1.
        parent: DesignData parent blueprint in `partial` mode, else None.
    """
    if opt.avg_mode == "partial":
        raw, parent_raw = _sample_partial_batch(batch_size)
    elif opt.avg_mode == "same_dssp":
        raw, parent_raw = _sample_same_dssp_batch(batch_size)
    else:
        raw, parent_raw = _sample_bias_batch(batch_size)
    designs = [data_from_salad(r) for r in raw]
    parent = data_from_salad(parent_raw) if parent_raw is not None else None

    sequence = _average_mpnn(designs)
    if init_temperature != 1.0:
        # Temperature acts on logits, so go back to log-space: softmax(log p / T) is
        # p^(1/T) normalized. Applying softmax(p / T) directly would instead compute
        # exp(p/T)/sum(exp(p/T)), which flattens toward uniform (measured: num_nonzero
        # 20.00 / max_prob 0.065, versus the correct 10.53 / 0.352).
        sequence = jax.nn.softmax(jnp.log(sequence + 1e-9) / init_temperature, axis=-1)
    # No explicit re-normalization: softmax already normalizes exactly when sharpening
    # runs, and otherwise the mean-of-B drift is float32 rounding (max 1.19e-07) that
    # loss() and project_to_simplex both absorb downstream.
    return designs, sequence, parent

def metrics(sequence, key, result: JoltzResult, target_distogram=None):
    """`target_distogram` must be passed in explicitly, NOT read from the module-level
    `params`: inside the jitted loss_update a global is baked into the compiled program as
    a constant at first trace, so a per-attempt value read that way stays frozen at attempt
    0's blueprint for the whole run. Pass the traced `params` argument's entry instead."""
    data = result.to_data(return_samples=True)
    dssp = data.p_dssp
    pmpnn_input = data.update(aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    logits = batch_mpnn(_pmpnn)(params["mpnn"], key, pmpnn_input.drop_aa())["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not result.is_single_sample:
        logits = logits.mean(axis=0)
    prob = jax.lax.stop_gradient(jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    entropy = -(logits * prob).sum(axis=-1).mean()
    # losses
    plddt = result.plddt
    pae = 32 * result.pae # unnormalized pAE
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)
    out = dict(
        logits = logits,
        entropy = entropy,
        plddt = plddt.mean(),
        recovery = (sequence * prob).sum(axis=-1).mean(),
        contacts = result.contact_score(num_contacts=25, min_resi_distance=10),
        pae = pae.mean(),
        average_num_nonzero = (sequence > 0.01).sum(axis=-1).mean(),
        L = dssp["L"], H = dssp["H"], E = dssp["E"],
        ptm = result.ptm,
    )
    if target_distogram is not None:
        out["distogram_nll"] = distogram_nll(result, target_distogram)
    return out

def dssp_to_string(dssp_array) -> str:
    """Convert an int DSSP array (0=loop, 1=helix, 2=strand) to an 'LHE' string."""
    code = np.array(list("LHE"))
    return "".join(code[np.asarray(dssp_array)])

def _structure_comparison(design: DesignData, predicted_data: DesignData) -> dict:
    """Once-per-attempt comparison between the salad blueprint backbone and the final
    predicted structure. Not called from metrics()/loss(): must run once per attempt
    against the converged final structure, not every gradient step."""
    salad_dssp = np.asarray(design.dssp)
    predicted_dssp = np.asarray(predicted_data.dssp)
    return dict(
        rmsd_to_blueprint=RMSD(atoms=["CA"])(predicted_data, design),
        dssp_salad=dssp_to_string(salad_dssp),
        dssp_predicted=dssp_to_string(predicted_dssp),
        dssp_identity=(salad_dssp == predicted_dssp).mean(),
    )

def _structure_comparison_batch(designs, predicted_data: DesignData) -> dict:
    """_structure_comparison generalized to a batch of blueprints.

    With an averaged initialization there is no single blueprint, so the canonical
    rmsd_to_blueprint / dssp_salad / dssp_identity are taken from the *closest* batch
    member (all three from the same member, so the row stays internally consistent and
    dssp_salad refers to a real on-disk PDB named by blueprint_index). The batch mean is
    reported alongside: a small min-vs-mean gap means the batch was structurally
    homogeneous, a large gap means the design collapsed onto one member.
    Also records the batch's own mean L/H/E composition, which is the direct check that
    the secondary-structure bias took effect.
    """
    per_member = [_structure_comparison(d, predicted_data) for d in designs]
    if len(per_member) == 1:
        return per_member[0]
    rmsds = np.array([float(p["rmsd_to_blueprint"]) for p in per_member])
    best = int(rmsds.argmin())
    result = dict(per_member[best])
    p_dssp = [d.p_dssp for d in designs]
    result.update(
        rmsd_to_blueprint_mean=np.float32(rmsds.mean()),
        dssp_identity_mean=np.float32(
            np.mean([float(p["dssp_identity"]) for p in per_member])),
        blueprint_index=np.float32(best),
        batch_L=np.float32(np.mean([float(x["L"]) for x in p_dssp])),
        batch_H=np.float32(np.mean([float(x["H"]) for x in p_dssp])),
        batch_E=np.float32(np.mean([float(x["E"]) for x in p_dssp])),
    )
    return result

def _init_provenance() -> dict:
    """Record how this design's starting distribution was built.
    Numeric values are numpy scalars because _report_result calls .mean() on them."""
    return dict(
        init_mode="avg" if avg_init else "single",
        init_batch=np.float32(init_batch_size),
        bias_lhe=(f"{opt.l_bias}|{opt.h_bias}|{opt.e_bias}" if use_bias else "none"),
        avg_mode=opt.avg_mode if avg_init else "none",
        partial_steps=np.float32(
            opt.partial_start_steps if (avg_init and opt.avg_mode == "partial") else 0),
        init_temp=np.float32(init_temperature),
    )

def _init_sharpness(sequence) -> dict:
    """Flatness of the starting distribution -- the early-warning column for the
    all-helix collapse. A single-backbone ProteinMPNN pass at T=1.0 sits at
    num_nonzero 11.95 / max_prob 0.370; num_nonzero >= 15 with max_prob <= 0.25 is the
    zone where Boltz fell back to a pure helix in practice."""
    s = np.asarray(sequence)
    return dict(
        init_num_nonzero=np.float32((s > 0.01).sum(axis=-1).mean()),
        init_max_prob=np.float32(s.max(axis=-1).mean()),
    )

def _batch_spread(designs, parent=None) -> dict:
    """Mean CA-RMSD of the batch members to a reference (the parent in `partial` mode,
    otherwise member 0). This is the direct check that partial/same_dssp actually produce
    structurally SIMILAR blueprints -- the premise that makes averaging meaningful."""
    reference = parent if parent is not None else designs[0]
    others = designs if parent is not None else designs[1:]
    if not others:
        return {}
    rmsds = [float(RMSD(atoms=["CA"])(d, reference)) for d in others]
    return dict(batch_rmsd_to_ref=np.float32(np.mean(rmsds)))

def _report_trajectory(index: int, writer: ScoreCSV, start_step=0, extra: dict = None):
    """`extra` carries static, non-jitted fields (e.g. `loss_config`) into each row: `aux`
    comes straight from the jitted loss_update every step, so it can never hold a string --
    extra is merged in afterward, entirely in plain Python."""
    def _report(step: int, loss: float, aux: dict):
        print(loss.shape)
        line_info = dict(attempt=index, step=start_step + step, total=float(loss.mean()))
        for key in writer.keys:
            if key in aux:
                line_info[key] = float(aux[key].mean())
        if extra:
            line_info.update(extra)
        writer.write_line(line_info)
    return _report

def _report_result(attempt: int, stage: str, key, writer: ScoreCSV, sequence, result: JoltzResult, aux=None):
    if aux is None:
        aux = metrics(sequence, key, result, target_distogram=params.get("target_distogram"))
    line_info = dict(attempt=attempt, stage=stage, sequence=aas.decode(jnp.argmax(sequence, axis=-1), aas.AF2_CODE))
    for key in writer.keys:
        if key in aux:
            value = aux[key]
            if isinstance(value, str):
                line_info[key] = value
            else:
                line_info[key] = float(value.mean())
    writer.write_line(line_info)
    return aux

# shared prefix for every loss variant below: set the sequence, predict structure, and
# compute every metric regardless of which terms the caller actually sums into `value`.
def _predict_and_score(sequence, key, params):
    keys = jax.random.split(key, 4)
    # forbid cysteines
    sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
    sequence = sequence / sequence.sum(axis=1, keepdims=True)
    # set sequence
    joltz_input: JoltzInput = params["joltz_input"].set_aa(sequence)
    # predict structure
    result: JoltzResult = joltz(params["joltz"]).predict(keys[0], joltz_input, num_samples=4)
    # compute losses
    # `params` here is the jitted function's traced argument, so target_distogram is a
    # real input that updates per attempt (unlike the module-level global params).
    out = metrics(sequence, keys[1], result, target_distogram=params.get("target_distogram"))
    out["result"] = result
    return out

# fold in the (weighted) distogram term shared by every loss variant, and finalize `out`.
def _with_distogram(value, out, params):
    distogram_weight = params.get("distogram_loss_weight", 0.0)
    if "distogram_nll" in out:
        value = value + distogram_weight * out["distogram_nll"]
    out["distogram_loss_weight"] = distogram_weight
    out["total"] = value
    return value, out

# fitness function for hallucination -- the normal landscape loss (all 4 terms + distogram)
def loss_full(sequence, key=None, context=None, params=None):
    out = _predict_and_score(sequence, key, params)
    value = (
          out["contacts"]
        - 10.0 * out["recovery"]
        + 0.4 * out["pae"]
        - 0.1 * out["plddt"]
    )
    return _with_distogram(value, out, params)
loss_update_full = jit_loss_update(loss_full)

# ablation variant: only recovery + plddt (+ distogram) -- contacts and pae dropped entirely
def loss_distogram_recovery_plddt(sequence, key=None, context=None, params=None):
    out = _predict_and_score(sequence, key, params)
    value = (
        - 10.0 * out["recovery"]
        - 0.1 * out["plddt"]
    )
    return _with_distogram(value, out, params)
loss_update_distogram_recovery_plddt = jit_loss_update(loss_distogram_recovery_plddt)

os.makedirs(f"{opt.out_path}/blueprints/", exist_ok=True)
os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
os.makedirs(f"{opt.out_path}/direct/", exist_ok=True)
os.makedirs(f"{opt.out_path}/success/", exist_ok=True)
if avg_init:
    os.makedirs(f"{opt.out_path}/init_dist/", exist_ok=True)
common_keys = [
    "plddt",
    "entropy",
    "recovery",
    "contacts",
    "pae",
    "distogram_nll",
    "distogram_loss_weight",
    "loss_config",
    "average_num_nonzero",
    "L", "H", "E",
    "ptm",
]
trajectory_keys = [k for k in common_keys]
trajectory = ScoreCSV(f"{opt.out_path}/trajectory.csv", keys=["attempt", "step"] + trajectory_keys)
results_keys = [
    "attempt", "stage", "init", "sequence"
] + common_keys + [
    "rmsd_to_blueprint", "dssp_salad", "dssp_predicted", "dssp_identity"
] + ([
    "init_mode", "init_batch", "bias_lhe",
    "avg_mode", "partial_steps", "init_temp",
    "init_num_nonzero", "init_max_prob", "batch_rmsd_to_ref",
    "rmsd_to_blueprint_mean", "dssp_identity_mean", "blueprint_index",
    "batch_L", "batch_H", "batch_E",
] if avg_init else [])
results = ScoreCSV(f"{opt.out_path}/results.csv", keys=results_keys)
success = ScoreCSV(f"{opt.out_path}/success.csv", keys=results_keys)

def _hallucinate_once(attempt, sequence, params, context, designs, init_diagnostics,
                      weight=None, loss_update=loss_update_full, config_label=None):
    """Run one hallucination optimization (loss_update init call + simplex_agpm + final
    prediction/report) starting from `sequence`. `weight`/`config_label` are only used to
    tag output filenames/CSV rows -- reassigning `sequence` here does not affect the
    caller's copy, so calling this repeatedly with the same outer `sequence` (e.g. once per
    distogram_loss_weight and/or loss configuration, against the same blueprint) restarts
    optimization from the same starting distribution each time. `loss_update` selects which
    loss formula to optimize (see loss_update_full / loss_update_distogram_recovery_plddt) --
    it must be a distinct jitted function, never a value threaded through `params`, since a
    formula choice can't be a jit-traced leaf."""
    suffix = f"_{config_label}" if config_label is not None else ""
    if weight is not None:
        suffix += f"_w{weight:g}"
    (_, aux), _ = loss_update(sequence, key=key(), params=params)
    aux_data: DesignData = aux["result"].to_data()
    aux_data.save_pdb(f"{opt.out_path}/attempts/design_{attempt}{suffix}_init.pdb")
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=key, params=params, context=context,
        num_steps=50,
        lr=0.1 * jnp.sqrt(sequence.shape[0]), momentum=0.3,
        scale=1.0,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(0.0),
        verbose=True, return_last=False,
        early_stop=None,
        report=_report_trajectory(attempt, trajectory, start_step=0,
                                  extra=dict(loss_config=config_label) if config_label is not None else None))
    result = aux["result"]
    data = result.to_data()
    hardened_sequence = data.to_sequence_string()
    prediction_1 = joltz_pred(key(), JoltzSpec().add_protein(hardened_sequence))
    final_result: JoltzResult = prediction_1.result
    predicted_data = final_result.to_data()
    report_aux = metrics(sequence, key(), final_result,
                         target_distogram=params.get("target_distogram"))
    report_aux.update(_structure_comparison_batch(designs, predicted_data))
    report_aux["init"] = "soft"
    if weight is not None:
        report_aux["distogram_loss_weight"] = np.float32(weight)
    if config_label is not None:
        report_aux["loss_config"] = config_label
    report_aux.update(_init_provenance())
    report_aux.update(init_diagnostics)
    _report_result(attempt, "stage_1", key(), results, sequence, final_result, aux=report_aux)
    prediction_1.save_pdb(f"{opt.out_path}/attempts/design_{attempt}{suffix}.pdb")

success_count = 0
attempt = 0
_attempt = 0
while success_count < opt.num_designs:
    attempt = _attempt
    _attempt += 1
    context = dict(
        recovery=10.0,
        ipae=0.05,
    )
    if avg_init:
        designs, sequence, parent_design = sample_init_batch(init_batch_size)
        for member, member_design in enumerate(designs):
            member_design.save_pdb(
                f"{opt.out_path}/blueprints/design_{attempt}_{member}.pdb")
        if parent_design is not None:
            # reference for batch_rmsd_to_ref; not recoverable from the batch otherwise
            parent_design.save_pdb(
                f"{opt.out_path}/blueprints/design_{attempt}_parent.pdb")
        np.save(f"{opt.out_path}/init_dist/design_{attempt}.npy", np.asarray(sequence))
    else:
        parent_design = None
        if not use_bias:
            salad_data["dssp_condition"], dssp_string = si.random_dssp(
                num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
            print("DSSP target:", dssp_string)
        salad_data["cloud_std"] = cloud_std_default(num_aa)
        salad_design = sampler(salad_params, key(), salad_data, init_prev)
        design = data_from_salad(salad_design)
        design.save_pdb(f"{opt.out_path}/blueprints/design_{attempt}.pdb")
        params["target_distogram"] = blueprint_distogram(design)
        sequence = aas.translate_onehot(
            pmpnn(key(), design.drop_aa())["logits"],
            aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
        sequence = jax.nn.softmax(sequence / opt.temperature, axis=-1)
        designs = [design]

    # capture before hallucination overwrites `sequence`
    init_diagnostics = _init_sharpness(sequence)
    init_diagnostics.update(_batch_spread(designs, parent_design))
    print(f"init distribution: num_nonzero={float(init_diagnostics['init_num_nonzero']):.2f} "
          f"max_prob={float(init_diagnostics['init_max_prob']):.3f}"
          + (f" batch_rmsd_to_ref={float(init_diagnostics['batch_rmsd_to_ref']):.2f}"
             if "batch_rmsd_to_ref" in init_diagnostics else ""))

    if hallucinate:
        if avg_init:
            _hallucinate_once(attempt, sequence, params, context, designs, init_diagnostics)
        elif loss_ablation:
            # Ablation: which loss TERMS matter, not just the distogram weight. 1 baseline
            # (no distogram) + `ablation_weights` runs of the full loss + distogram +
            # `ablation_weights` runs of a reduced loss (distogram + recovery + plddt only).
            # All against the SAME blueprint/target_distogram, restarting from the same
            # initial `sequence` each time.
            run_specs = [("no_distogram", loss_update_full, 0.0)]
            run_specs += [("full_with_distogram", loss_update_full, w) for w in ablation_weights]
            run_specs += [("distogram_recovery_plddt", loss_update_distogram_recovery_plddt, w)
                          for w in ablation_weights]
            for config_label, lu, w in run_specs:
                params["distogram_loss_weight"] = w
                _hallucinate_once(attempt, sequence, params, context, designs, init_diagnostics,
                                  weight=w, loss_update=lu, config_label=config_label)
        else:
            # Sweep distogram_loss_weight against the SAME blueprint/target_distogram
            # (set once per attempt above), restarting from the same initial `sequence`
            # for each weight.
            for w in distogram_loss_weights:
                params["distogram_loss_weight"] = w
                _hallucinate_once(attempt, sequence, params, context, designs, init_diagnostics, weight=w)
    else:
        hard_sequence = jax.nn.one_hot(jnp.argmax(sequence, axis=-1), 20)
        for init_label, seq_variant in (("soft", sequence), ("hard", hard_sequence)):
            joltz_input: JoltzInput = params["joltz_input"].set_aa(seq_variant)
            final_result: JoltzResult = joltz(params["joltz"]).predict(key(), joltz_input, num_samples=4)
            predicted_data = final_result.to_data()
            report_aux = metrics(seq_variant, key(), final_result,
                                 target_distogram=params.get("target_distogram"))
            report_aux.update(_structure_comparison_batch(designs, predicted_data))
            report_aux["init"] = init_label
            report_aux.update(_init_provenance())
            report_aux.update(init_diagnostics)
            _report_result(attempt, "direct", key(), results, seq_variant, final_result, aux=report_aux)
            predicted_data.save_pdb(f"{opt.out_path}/direct/design_{attempt}_{init_label}.pdb")
    success_count += 1
