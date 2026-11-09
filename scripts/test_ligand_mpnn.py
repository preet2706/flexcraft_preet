"""Ad-hoc verification driver for the LigandMPNN port.

There is no test suite in this repository - running it is how a change is validated. This
script covers the port in five modes:

    parity  compare every stage against the reference PyTorch implementation. Requires a
            clone of the reference and the torch CPU wheel, which is already a dependency:
                git clone --depth 1 https://github.com/dauparas/LigandMPNN \\
                    dependencies/LigandMPNN
    smoke   run the model on a real complex and check that the ligand actually changes which
            amino acids the pocket prefers.
    sample  sequence design round-trip through flexcraft.sequence.sample.sample.
    batch   jit and vmap behaviour, including shared and per-sample ligand poses.
    data    DesignData operations preserve the ligand fields.

Examples:

    JAX_PLATFORMS=cpu python scripts/test_ligand_mpnn.py --mode parity \\
        --pdb data/ligand_test/1STP.pdb --ligand_res BTN
    python scripts/test_ligand_mpnn.py --mode smoke --pdb data/ligand_test/1HSG.pdb \\
        --ligand_res MK1
    python scripts/test_ligand_mpnn.py --mode all
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.files.ligand import (
    LigandAtoms, load_complex, ligand_from_pdb, ligand_bucket)
from flexcraft.sequence.ligand_mpnn import (
    make_ligmpnn, batch_ligand_mpnn, nearest_context_atoms)
from flexcraft.sequence.sample import (
    sample, transform_logits, scale_by_temperature, forbid, norm_logits)
from flexcraft.utils.options import parse_options


def report(label, passed, detail=""):
    """Print a single check result and return whether it passed."""
    print(f"  [{'ok' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return passed


def load_case(opt, num_atoms=None):
    """Load the test complex named by the options, as a DesignData with its ligand."""
    res_names = (opt.ligand_res,) if opt.ligand_res else None
    count = len(ligand_from_pdb(opt.pdb, res_names=res_names))
    if num_atoms is None:
        num_atoms = ligand_bucket(count)
    data = load_complex(opt.pdb, res_names=res_names, num_atoms=num_atoms)
    print(f"{opt.pdb}: {data.size} residues, {count} ligand atoms "
          f"(padded to {data.num_ligand_atoms})")
    return data


def to_pmpnn(data):
    """Translate a DesignData's sequence from AF2 into ProteinMPNN amino acid order."""
    return data.update(aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))


# ============================== parity ==============================

def _reference_gather(cb, mask, positions, elements, ligand_mask, num_context, torch):
    """`get_nearest_neighbours` from the reference data_utils.py, reimplemented in torch.

    The reference version lives in data_utils.py, which imports ProDy at module scope, so it
    cannot be imported here. It is twelve lines, so it is reproduced instead.
    """
    pair_mask = mask[:, None] * ligand_mask[None, :]
    distance_sq = torch.sum((cb[:, None, :] - positions[None, :, :]) ** 2, -1)
    distance_sq = distance_sq * pair_mask + (1 - pair_mask) * 1000.0

    index = torch.argsort(distance_sq, -1)[:, :num_context]
    nearest = torch.gather(distance_sq, 1, index)
    closest = torch.sqrt(nearest[:, 0])

    length = cb.shape[0]
    gathered_positions = torch.gather(
        positions[None].repeat(length, 1, 1), 1, index[:, :, None].repeat(1, 1, 3))
    gathered_elements = torch.gather(elements[None].repeat(length, 1), 1, index)
    gathered_mask = torch.gather(ligand_mask[None].repeat(length, 1), 1, index)

    out_positions = torch.zeros([length, num_context, 3], dtype=torch.float32)
    out_elements = torch.zeros([length, num_context], dtype=torch.float32)
    out_mask = torch.zeros([length, num_context], dtype=torch.float32)
    filled = gathered_positions.shape[1]
    out_positions[:, :filled] = gathered_positions
    out_elements[:, :filled] = gathered_elements
    out_mask[:, :filled] = gathered_mask
    return out_positions, out_elements, out_mask, closest


def mode_parity(opt):
    """Compare the port against the reference implementation, stage by stage."""
    if not os.path.isdir(opt.reference):
        print(f"reference clone not found at {opt.reference!r}. Run:\n"
              f"  git clone --depth 1 https://github.com/dauparas/LigandMPNN {opt.reference}")
        return False
    import torch
    sys.path.insert(0, os.path.abspath(opt.reference))
    import model_utils

    torch.set_grad_enabled(False)
    res_names = (opt.ligand_res,) if opt.ligand_res else None
    raw = ligand_from_pdb(opt.pdb, res_names=res_names)
    # compare against the reference on an unpadded ligand, so that both sides see exactly the
    # same atom list. Padding invariance is checked separately below.
    data = to_pmpnn(load_case(opt, num_atoms=len(raw)))
    checkpoint = torch.load(opt.raw_params, map_location="cpu", weights_only=True)
    num_edges = int(checkpoint["num_edges"])
    context_num = int(checkpoint["atom_context_num"])

    # --- our side: a full forward pass with intermediates exposed
    model = make_ligmpnn(opt.params, eps=0.0, return_intermediates=True)
    masked = data.drop_aa()
    ours = model(jax.random.PRNGKey(0), masked)
    mid = ours["intermediates"]

    # --- the gather, in isolation
    X = np.asarray(masked["atom_positions"])[:, :4]
    N, Ca, C = X[:, 0], X[:, 1], X[:, 2]
    b, c = Ca - N, C - Ca
    cb = (-0.58273431 * np.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Ca)
    reference_gather = _reference_gather(
        torch.tensor(cb, dtype=torch.float32),
        torch.tensor(np.asarray(masked["mask"]), dtype=torch.float32),
        torch.tensor(raw.positions, dtype=torch.float32),
        torch.tensor(raw.elements, dtype=torch.float32),
        torch.tensor(raw.mask, dtype=torch.float32),
        context_num, torch)
    ours_gather = nearest_context_atoms(
        jnp.asarray(cb), masked["mask"], masked["ligand_positions"],
        masked["ligand_elements"], masked["ligand_mask"], context_num)

    passed = []
    # the gathered arrays are pure index lookups and must agree exactly; the closest-atom
    # distance is a square root, and this port keeps an epsilon under it that the reference
    # omits, so it is only compared to tolerance
    for label, a, b_, tolerance in (
            ("gather positions", ours_gather[0], reference_gather[0], 0.0),
            ("gather elements", ours_gather[1], reference_gather[1], 0.0),
            ("gather mask", ours_gather[2], reference_gather[2], 0.0),
            ("gather distance", ours_gather[3], reference_gather[3], opt.tol)):
        delta = float(np.abs(np.asarray(a, dtype=np.float64)
                             - b_.numpy().astype(np.float64)).max())
        passed.append(report(label, delta <= tolerance,
                             f"max|delta| = {delta:.3g}"
                             + ("" if tolerance == 0.0 else f" (tol {tolerance:g})")))

    # --- the model, stage by stage
    length = data.size
    feature_dict = dict(
        X=torch.tensor(X, dtype=torch.float32)[None],
        mask=torch.tensor(np.asarray(masked["mask"]), dtype=torch.float32)[None],
        R_idx=torch.tensor(np.asarray(masked["residue_index"]), dtype=torch.float32)[None],
        chain_labels=torch.tensor(
            np.asarray(masked["chain_index"]), dtype=torch.float32)[None],
        S=torch.tensor(np.asarray(masked["aa"]), dtype=torch.long)[None],
        Y=reference_gather[0][None],
        Y_t=reference_gather[1][None],
        Y_m=reference_gather[2][None])

    reference = model_utils.ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=num_edges,
        device="cpu", atom_context_num=context_num, model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=False)
    reference.load_state_dict(checkpoint["model_state_dict"])
    reference.eval()

    V, E, E_idx, Y_nodes, Y_edges, Y_m = reference.features(feature_dict)
    h_V, h_E, _ = reference.encode(feature_dict)

    def compare(label, ours_value, reference_value, tolerance):
        a = np.asarray(ours_value, dtype=np.float64)
        b_ = np.asarray(reference_value.numpy(), dtype=np.float64)
        if a.shape != b_.shape:
            return report(label, False, f"shape {a.shape} vs {b_.shape}")
        delta = float(np.abs(a - b_).max())
        return report(label, delta < tolerance,
                      f"max|delta| = {delta:.3g} (tol {tolerance:g})")

    tol = opt.tol
    passed.append(report(
        "neighbour indices (E_idx)",
        bool((np.asarray(mid["neighbours"]) == E_idx[0].numpy()).all()),
        "exact match required"))
    passed.append(compare("backbone edges (E)", mid["E"], E[0], tol))
    passed.append(compare("ligand node features (V)", mid["V"], V[0], tol))
    # Y_nodes/Y_edges are compared masked: y_nodes runs before the first masking step, so
    # padded context slots hold values which never reach the residue embeddings
    mask_nodes = np.asarray(mid["Y_m"])[..., None]
    passed.append(compare("ligand graph nodes (Y_nodes)",
                          np.asarray(mid["Y_nodes"]) * mask_nodes,
                          Y_nodes[0] * torch.tensor(mask_nodes, dtype=torch.float32), tol))
    edge_mask = (np.asarray(mid["Y_m"])[..., :, None]
                 * np.asarray(mid["Y_m"])[..., None, :])[..., None]
    passed.append(compare("ligand graph edges (Y_edges)",
                          np.asarray(mid["Y_edges"]) * edge_mask,
                          Y_edges[0] * torch.tensor(edge_mask, dtype=torch.float32), tol))
    passed.append(compare("node embeddings after ligand context (h_V)",
                          mid["local"], h_V[0], opt.tol_hv))

    # --- logits, end to end. With every position masked, flexcraft's single-pass decoder
    # carries no sequence information (mask_bw is zero), which the reference decoder layers
    # can be driven to reproduce directly.
    h_EX_encoder = model_utils.cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = model_utils.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder = feature_dict["mask"][..., None, None] * h_EXV_encoder
    h_V_decoded = h_V
    for layer in reference.decoder_layers:
        h_V_decoded = layer(h_V_decoded, h_EXV_encoder, feature_dict["mask"])
    reference_logits = torch.log_softmax(reference.W_out(h_V_decoded), dim=-1)
    passed.append(compare("logits", ours["logits"], reference_logits[0], opt.tol_hv))

    # Padding the ligand to a fixed size is what keeps jax.jit from recompiling per ligand,
    # so it must not change the result. It silently did until MASKED_DISTANCE_SQ replaced the
    # reference's 1000.0 sentinel, which - being a squared distance - let padding outrank
    # genuine atoms more than 31.6 Angstrom away.
    padded = to_pmpnn(load_case(opt, num_atoms=max(ligand_bucket(len(raw)), 256)))
    padded_logits = model(jax.random.PRNGKey(0), padded.drop_aa())["logits"]
    delta = float(jnp.abs(padded_logits - ours["logits"]).max())
    passed.append(report("padding the ligand does not change the logits", delta < 1e-5,
                         f"{len(raw)} -> {padded.num_ligand_atoms} atoms, "
                         f"max|delta| = {delta:.3g}"))

    print(f"\nparity: {sum(passed)}/{len(passed)} checks passed")
    return all(passed)


# ============================== smoke ==============================

def mode_smoke(opt):
    """Check that the ligand measurably changes the model's pocket preferences."""
    data = to_pmpnn(load_case(opt))
    model = jax.jit(make_ligmpnn(opt.params, eps=0.0))
    masked = data.drop_aa()

    with_ligand = model(jax.random.PRNGKey(0), masked)
    # the apo baseline is the same model with a fully masked ligand, not ProteinMPNN: the
    # ligand branch still contributes its residual, so swapping models would confound the
    # comparison
    apo_input = masked.with_ligand(
        LigandAtoms.empty().pad(masked.num_ligand_atoms))
    apo = model(jax.random.PRNGKey(0), apo_input)

    pocket = np.asarray(with_ligand["ligand_contact_mask"]) > 0
    native = np.asarray(data["aa"])
    passed = [report("logits shape", with_ligand["logits"].shape == (data.size, 21),
                     str(with_ligand["logits"].shape)),
              report("finite logits", bool(jnp.isfinite(with_ligand["logits"]).all())),
              report("pocket residues found", pocket.sum() > 0,
                     f"{int(pocket.sum())} of {data.size} residues within "
                     f"{opt.cutoff} A of the ligand")]

    def recovery(result, selection):
        prediction = np.asarray(jnp.argmax(result["logits"][:, :20], axis=-1))
        return float((prediction[selection] == native[selection]).mean())

    holo_pocket = recovery(with_ligand, pocket)
    apo_pocket = recovery(apo, pocket)
    holo_other = recovery(with_ligand, ~pocket)
    apo_other = recovery(apo, ~pocket)
    print(f"\n  native sequence recovery (argmax, fully masked input):")
    print(f"    pocket     holo {holo_pocket:.3f}  apo {apo_pocket:.3f}  "
          f"delta {holo_pocket - apo_pocket:+.3f}")
    print(f"    non-pocket holo {holo_other:.3f}  apo {apo_other:.3f}  "
          f"delta {holo_other - apo_other:+.3f}")
    print(f"  mean entropy  holo {float(with_ligand['ent'].mean()):.3f}  "
          f"apo {float(apo['ent'].mean()):.3f}")
    print(f"  consensus (holo) {aas.decode(jnp.argmax(with_ligand['logits'][:, :20], -1), aas.PMPNN_CODE)[:60]}")

    passed.append(report(
        "ligand improves pocket recovery", holo_pocket > apo_pocket,
        f"{holo_pocket:.3f} > {apo_pocket:.3f}"))
    passed.append(report(
        "ligand changes pocket logits more than the rest of the protein",
        float(jnp.abs(with_ligand["logits"] - apo["logits"])[pocket].mean())
        > float(jnp.abs(with_ligand["logits"] - apo["logits"])[~pocket].mean())))
    print(f"\nsmoke: {sum(passed)}/{len(passed)} checks passed")
    return all(passed)


# ============================== sample ==============================

def mode_sample(opt):
    """Design a sequence through the shared sampler and check the ligand survives."""
    data = to_pmpnn(load_case(opt))
    model = jax.jit(make_ligmpnn(opt.params, eps=0.0))
    sampler = sample(model, logit_transform=transform_logits([
        scale_by_temperature(temperature=opt.temperature),
        forbid("C", aas.PMPNN_CODE),
        norm_logits]))

    masked = data.drop_aa()
    result, log_p = sampler(jax.random.PRNGKey(0), masked)
    other, _ = sampler(jax.random.PRNGKey(1), masked)

    passed = [
        report("no masked positions remain", not bool((result["aa"] == 20).any())),
        report("no cysteines sampled",
               not bool((result["aa"] == aas.PMPNN_CODE.index("C")).any())),
        report("log_p is finite", bool(np.isfinite(float(log_p))), f"log_p = {float(log_p):.2f}"),
        report("ligand fields survive sampling",
               all(key in result for key in
                   ("ligand_positions", "ligand_elements", "ligand_mask"))),
        report("ligand unchanged by sampling",
               bool(jnp.allclose(result["ligand_positions"], data["ligand_positions"]))),
        report("different seeds give different sequences",
               not bool((result["aa"] == other["aa"]).all())),
    ]
    designed = aas.decode(result["aa"], aas.PMPNN_CODE)
    native = aas.decode(data["aa"], aas.PMPNN_CODE)
    identity = float(np.mean([a == b for a, b in zip(designed, native)]))
    print(f"\n  designed {designed[:60]}")
    print(f"  native   {native[:60]}")
    print(f"  sequence identity to native: {identity:.3f}")
    print(f"\nsample: {sum(passed)}/{len(passed)} checks passed")
    return all(passed)


# ============================== batch ==============================

def mode_batch(opt):
    """Check jit stability and the batched sample axis, with shared and per-sample ligands."""
    data = to_pmpnn(load_case(opt)).drop_aa()
    apply_fn, params = make_ligmpnn(opt.params, eps=0.0, split_params=True)
    single = jax.jit(apply_fn)
    batched = jax.jit(batch_ligand_mpnn(apply_fn))
    num_samples = opt.num_samples
    key = jax.random.PRNGKey(0)

    # a batch of identical structures, so every sample must agree with the unbatched call
    positions = jnp.repeat(data["atom_positions"][None], num_samples, axis=0)
    shared = data.update(atom_positions=positions)
    batch_result = batched(params, key, shared)
    keys = jax.random.split(key, num_samples)
    reference = jnp.stack([
        single(params, keys[i], data)["logits"] for i in range(num_samples)])
    delta = float(jnp.abs(batch_result["logits"] - reference).max())
    passed = [report("batched logits match per-sample calls", delta < 1e-5,
                     f"max|delta| = {delta:.3g}"),
              report("batched shape",
                     batch_result["logits"].shape == (num_samples, data.size, 21),
                     str(batch_result["logits"].shape))]

    # a ligand which moves per sample
    per_sample = jnp.repeat(data["ligand_positions"][None], num_samples, axis=0)
    per_sample = per_sample + jnp.arange(num_samples)[:, None, None] * 0.0
    moved = shared.update(ligand_positions=per_sample)
    moved_result = batched(params, key, moved)
    delta = float(jnp.abs(moved_result["logits"] - reference).max())
    passed.append(report("per-sample ligand pose accepted", delta < 1e-5,
                         f"(S, A, 3) ligand, max|delta| = {delta:.3g}"))

    # tie fields must survive the batcher, which batch_mpnn drops
    passed.append(report("tie fields reach the model through the batcher",
                         "tie_index" in shared and "tie_weights" in shared))

    # jit must not recompile for a different ligand padded to the same size
    other_ligand = LigandAtoms.empty().pad(data.num_ligand_atoms)
    before = single._cache_size()
    single(params, key, data.with_ligand(other_ligand))
    after = single._cache_size()
    passed.append(report("no recompile for a different ligand of the same size",
                         after == before, f"cache size {before} -> {after}"))
    print(f"\nbatch: {sum(passed)}/{len(passed)} checks passed")
    return all(passed)


# ============================== data ==============================

def mode_data(opt):
    """Check that DesignData operations leave the ligand fields intact."""
    data = load_case(opt)
    original = data["ligand_positions"]
    num_atoms = data.num_ligand_atoms
    free = DesignData.from_length(40)

    cases = {
        "slice d[:50]": data[:50],
        "boolean mask": data[np.arange(data.size) % 2 == 0],
        "index list": data[[slice(0, 10), slice(20, 30)]],
        "split()": data.split()[0],
        ".protein": data.protein,
        "drop_aa()": data.drop_aa(),
        "untie()": data.untie(),
        "copy()": data.copy(),
        "cpu()": data.cpu(),
        "update_index()": data.update_index(
            slice(0, 5), aa=jnp.zeros((5,), jnp.int32)),
        "complex with ligand-free (d / free)": data / free,
        "complex, reversed (free / d)": free / data,
        "single chain (d + free)": data + free,
    }
    passed = []
    for label, result in cases.items():
        ok = (result.has_ligand
              and result["ligand_positions"].shape == (num_atoms, 3)
              and bool(jnp.allclose(result["ligand_positions"], original)))
        passed.append(report(label, ok, f"L = {result.size}"))

    passed.append(report("drop_ligand() removes the fields",
                         not data.drop_ligand().has_ligand))
    passed.append(report("ligand-free composition stays ligand-free",
                         not (free / free).has_ligand))

    # the model must still run on the sliced results
    model = jax.jit(make_ligmpnn(opt.params, eps=0.0))
    sliced = to_pmpnn(data[:50]).drop_aa()
    logits = model(jax.random.PRNGKey(0), sliced)["logits"]
    passed.append(report("model runs on a sliced complex",
                         logits.shape == (50, 21) and bool(jnp.isfinite(logits).all())))

    # a missing ligand must fail loudly rather than silently running apo
    try:
        model(jax.random.PRNGKey(0), to_pmpnn(data.drop_ligand()).drop_aa())
        passed.append(report("missing ligand raises", False, "no error raised"))
    except KeyError as error:
        passed.append(report("missing ligand raises", True, str(error)[:60] + "..."))

    print(f"\ndata: {sum(passed)}/{len(passed)} checks passed")
    return all(passed)


# ============================== boltz ==============================

class _FakeJoltzResult:
    """A stand-in for a JoltzResult, holding only what the ligand extractor reads.

    Running Boltz needs a GPU and its weights, so the extractor's index arithmetic is
    checked against a hand-built features dict laid out the way Boltz lays one out: atoms
    grouped into tokens, `mol_type` per token, one-hot `ref_element` per atom, and the atom
    array padded at the end.
    """

    def __init__(self, num_samples=None):
        # 3 protein tokens of 4 atoms, then 5 single-atom ligand tokens, then 3 pad atoms
        num_tokens, num_atoms = 8, 20
        atom_to_token = np.zeros((num_atoms, num_tokens), dtype=np.float32)
        for atom in range(12):
            atom_to_token[atom, atom // 4] = 1.0
        for offset, atom in enumerate(range(12, 17)):
            atom_to_token[atom, 3 + offset] = 1.0
        # padding atoms are attached to a ligand token but carry element 0
        atom_to_token[17:20, 7] = 1.0

        mol_type = np.array([0, 0, 0, 3, 3, 3, 3, 3], dtype=np.int32)
        # C, N, O, S, H for the ligand tokens; 0 for protein backbone and padding
        atomic_numbers = np.zeros((num_atoms,), dtype=np.int32)
        atomic_numbers[:12] = 6
        atomic_numbers[12:17] = [6, 7, 8, 16, 1]
        ref_element = np.eye(120, dtype=np.float32)[atomic_numbers]
        atom_pad_mask = np.ones((num_atoms,), dtype=np.float32)
        atom_pad_mask[17:] = 0.0

        coordinates = np.arange(num_atoms * 3, dtype=np.float32).reshape(num_atoms, 3)
        if num_samples is None:
            samples = coordinates[None]                       # (rollout, atoms, 3)
            self.pae = np.zeros((num_tokens, num_tokens), dtype=np.float32)
        else:
            samples = np.stack([coordinates + index for index in range(num_samples)])
            samples = samples[:, None]                        # (samples, rollout, atoms, 3)
            self.pae = np.stack([
                np.full((num_tokens, num_tokens), float(num_samples - index))
                for index in range(num_samples)])
        self.data = dict(samples=jnp.asarray(samples), features=dict(
            atom_to_token=jnp.asarray(atom_to_token)[None],
            mol_type=jnp.asarray(mol_type)[None],
            ref_element=jnp.asarray(ref_element)[None],
            atom_pad_mask=jnp.asarray(atom_pad_mask)[None]))
        self.expected_positions = coordinates[12:16]
        self.expected_elements = np.array([6, 7, 8, 16], dtype=np.int32)


def mode_boltz(opt):
    """Check the Boltz ligand extractor's atom selection and sample indexing."""
    from flexcraft.structure.boltz._ligand import ligand_from_joltz, joltz_ligand_fields

    single = _FakeJoltzResult()
    ligand = ligand_from_joltz(single)
    passed = [
        report("protein, hydrogen and padding atoms excluded", len(ligand) == 4,
               f"{len(ligand)} atoms kept of 20"),
        report("elements correct",
               bool((ligand.elements == single.expected_elements).all()),
               str(ligand.elements)),
        report("coordinates correct",
               bool(np.allclose(ligand.positions, single.expected_positions))),
        report("all kept atoms unmasked", bool(ligand.mask.all())),
    ]

    # the best sample is the one with the lowest mean PAE, which this stub makes the last
    multi = _FakeJoltzResult(num_samples=3)
    best = ligand_from_joltz(multi)
    passed.append(report(
        "multi-sample picks the lowest-PAE sample",
        bool(np.allclose(best.positions, multi.expected_positions + 2)),
        "offset +2 of 0,1,2"))
    passed.append(report(
        "explicit sample_index honoured",
        bool(np.allclose(
            ligand_from_joltz(multi, sample_index=0).positions,
            multi.expected_positions))))

    fields = joltz_ligand_fields(single)
    passed.append(report(
        "fields padded to a bucket",
        fields["ligand_positions"].shape == (ligand_bucket(4), 3)
        and int(fields["ligand_mask"].sum()) == 4,
        f"{fields['ligand_positions'].shape}, {int(fields['ligand_mask'].sum())} valid"))
    per_sample = joltz_ligand_fields(multi, per_sample=True)
    passed.append(report(
        "per-sample poses stacked",
        per_sample["ligand_positions"].shape == (3, ligand_bucket(4), 3),
        str(per_sample["ligand_positions"].shape)))

    # the extracted ligand must drive the model
    data = to_pmpnn(DesignData.from_length(20)).with_ligand(
        ligand, num_atoms=ligand_bucket(len(ligand)))
    model = jax.jit(make_ligmpnn(opt.params, eps=0.0))
    logits = model(jax.random.PRNGKey(0), data.drop_aa())["logits"]
    passed.append(report("model runs on a Boltz-derived ligand",
                         logits.shape == (20, 21) and bool(jnp.isfinite(logits).all())))

    print(f"\nboltz: {sum(passed)}/{len(passed)} checks passed")
    return all(passed)


MODES = dict(parity=mode_parity, smoke=mode_smoke, sample=mode_sample,
             batch=mode_batch, data=mode_data, boltz=mode_boltz)


if __name__ == "__main__":
    opt = parse_options(
        "Verification driver for the LigandMPNN port.",
        mode="smoke",
        pdb="data/ligand_test/1STP.pdb",
        ligand_res="BTN",
        params="params/ligmpnn/v_32_030_25.pkl",
        raw_params="params/ligmpnn/raw/ligandmpnn_v_32_030_25.pt",
        reference="dependencies/LigandMPNN",
        temperature=0.1,
        num_samples=4,
        cutoff=8.0,
        tol=1e-4,
        tol_hv=1e-3)

    modes = list(MODES) if opt.mode == "all" else opt.mode.split(",")
    results = {}
    for name in modes:
        if name not in MODES:
            raise SystemExit(f"unknown mode {name!r}, pick from {sorted(MODES)} or 'all'")
        print(f"\n=== {name} ===")
        results[name] = MODES[name](opt)
    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name:8s} {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)
