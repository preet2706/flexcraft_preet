# LigandMPNN

`flexcraft.sequence.ligand_mpnn` provides a JAX/Haiku port of
[LigandMPNN](https://github.com/dauparas/LigandMPNN), ported from the reference PyTorch
implementation. It is ProteinMPNN conditioned on nearby non-protein atoms: the protein
encoder runs unchanged, a small graph network over the nearest ligand atoms around each
residue contributes a residual to the node embeddings, and the decoder is identical to
ProteinMPNN's.

Because the ligand only enters as that residual, `make_ligmpnn` is a drop-in sibling of
`make_pmpnn` — same call signature, same output dict — and works unchanged with
`flexcraft.sequence.sample.sample`, `flexcraft.protocols.af2cycler.af_cycler` and
`flexcraft.hallucination.losses.sequence.mpnn_recovery`.

## Getting started

`download_params.sh` fetches the released LigandMPNN weights and converts them into
flexcraft's format. To do it by hand:

```bash
mkdir -p params/ligmpnn/raw && cd params/ligmpnn/raw
for n in 005 010 020 030; do
    wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_${n}_25.pt
done
cd -
python scripts/convert_ligandmpnn_params.py \
    --in_path params/ligmpnn/raw/ligandmpnn_v_32_030_25.pt \
    --out_path params/ligmpnn/v_32_030_25.pkl
```

In the checkpoint names, `v_32` is the number of neighbours in the residue graph (**not**
ProteinMPNN's 48), `_030` the training noise level and `_25` the number of ligand context
atoms per residue. `make_ligmpnn` reads both numbers from the checkpoint, so do not pass
ProteinMPNN's defaults.

Then design a sequence for a protein–ligand complex:

```python
import jax
import flexcraft.sequence.aa_codes as aas
from flexcraft.files.ligand import load_complex, ligand_bucket
from flexcraft.sequence.ligand_mpnn import make_ligmpnn
from flexcraft.sequence.sample import (
    sample, transform_logits, scale_by_temperature, forbid, norm_logits)

# load_complex keeps the ligand's chemistry; utils.io.load_pdb does not (see below)
data = load_complex("data/ligand_test/1STP.pdb", res_names=("BTN",), num_atoms=32)
data = data.update(aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))

model = jax.jit(make_ligmpnn("params/ligmpnn/v_32_030_25.pkl", eps=0.05))
sampler = sample(model, logit_transform=transform_logits([
    scale_by_temperature(temperature=0.1),
    forbid("C", aas.PMPNN_CODE),
    norm_logits]))

result, log_p = sampler(jax.random.PRNGKey(0), data.drop_aa())
sequence = aas.decode(result["aa"], aas.PMPNN_CODE)
```

As with ProteinMPNN, sequences come back in ProteinMPNN's amino acid order — LigandMPNN uses
the same alphabet (`aas.PMPNN_CODE`, index 20 for the unknown token), so translate with
`aas.translate(..., aas.PMPNN_CODE, aas.AF2_CODE)` on the way out.

## Loading ligands

**Do not use `flexcraft.utils.io.load_pdb` for a structure with a ligand.** It goes through
salad's AlphaFold reader, which turns every HETATM residue into an UNK residue with an empty
atom mask: the ligand's chemistry is lost, and the resulting phantom `aa=20` rows are then
treated as designable positions by `drop_aa` and `sample`. Use `load_complex` instead, which
keeps only amino acids in the protein and returns the ligand separately.

`flexcraft.files.ligand` provides:

| function | source |
|---|---|
| `load_complex(path, ...)` | a PDB/mmCIF complex, as a `DesignData` with its ligand attached |
| `ligand_from_pdb` / `ligand_from_cif` / `ligand_from_file` | HETATM records, via biotite |
| `ligand_from_sdf` / `ligand_from_sdf_all` / `ligand_from_mol2` | small-molecule files, via rdkit |
| `ligand_from_smiles(smiles)` | a SMILES string, embedded with rdkit |
| `ligand_from_ccd(code)` | a CCD component from Boltz's local `params/boltz/mols/` cache |
| `flexcraft.structure.boltz.ligand_from_joltz(result)` | the ligand pose of a Boltz prediction |

All of them return a `LigandAtoms`: flat per-atom positions, atomic numbers and a mask.
Waters, hydrogens and unrecognised elements are excluded, matching upstream. Attach one to a
structure with `data.with_ligand(ligand, num_atoms=...)`.

`ligand_from_smiles` and `ligand_from_ccd` produce a conformer in an **arbitrary reference
frame** — it is not docked into the pocket. They are only meaningful for composition-only
experiments, or after a separate placement step. For a hallucination pipeline, take the
ligand from the predicted complex with `ligand_from_joltz`.

### Padding

Pad the ligand to a fixed atom count, with `num_atoms=` or `ligand_bucket(n)`. The atom count
is part of the traced shape, so without padding `jax.jit` recompiles for every distinct
ligand. Padded atoms are masked and inert, and padding does not change the result.

## Working with `DesignData`

Ligand atoms live in three scene-level fields — `ligand_positions`, `ligand_elements` and
`ligand_mask` — which are *not* indexed by residue. `DesignData` treats them accordingly:
slicing (`data[:50]`), `split()`, `.protein` and `drop_aa()` leave them untouched, and
composing a ligand-bearing structure with a ligand-free one (`target / binder`) carries the
ligand across rather than dropping it. `has_ligand`, `num_ligand_atoms`, `with_ligand(...)`
and `drop_ligand()` manage them.

A missing ligand raises rather than silently running without one. For a deliberate apo
baseline, attach `LigandAtoms.empty()`. Note that this is **not** the same as running
ProteinMPNN: the ligand branch still contributes its residual, so an apo comparison should
use LigandMPNN with an empty ligand, not `make_pmpnn`, or the comparison is confounded by two
different models rather than by the ligand.

## Batching

Use `batch_ligand_mpnn` rather than `batch_mpnn` for a batch of structures:

```python
from flexcraft.sequence.ligand_mpnn import make_ligmpnn, batch_ligand_mpnn

apply_fn, params = make_ligmpnn("params/ligmpnn/v_32_030_25.pkl", split_params=True)
batched = jax.jit(batch_ligand_mpnn(apply_fn))
logits = batched(params, key, data)["logits"]   # data["atom_positions"] is (S, L, 14, 3)
```

`batch_mpnn` replicates only a hard-coded list of fields and silently drops everything else,
including the ligand and `tie_index`/`tie_weights`. `batch_ligand_mpnn` maps over the sample
axis with a per-field `in_axes` instead, so nothing is dropped and shared fields are not
copied per sample. A ligand may be shared across samples, with shape `(A, 3)`, or move per
sample, with shape `(S, A, 3)`.

For hallucination, `mpnn_recovery` takes a `batcher` argument:

```python
from flexcraft.hallucination.losses.sequence import mpnn_recovery
recovery = mpnn_recovery(apply_fn, params, batcher=batch_ligand_mpnn)
```

## Extra outputs

Alongside ProteinMPNN's `aa`, `logits`, `ent` and `var_ent`, the result carries:

- `ligand_distance` — `(L,)` distance from each residue's pseudo-CB to the closest ligand atom.
- `ligand_contact_mask` — `(L,)` which residues are pocket residues, i.e. within
  `cutoff_for_score` (default 8 Å). Upstream computes this for reporting only; it is not a
  model input. It is a convenient pocket selector for interface-vs-elsewhere analyses.

## In the hallucination pipeline

`flexcraft.pipelines.experimental.ligand_bias` takes `--mpnn_model {pmpnn,ligand_mpnn}`:

```bash
python -m flexcraft.pipelines.experimental.ligand_bias \
    --out_path runs/ligand_bias_ligmpnn/ --categories A \
    --mpnn_model ligand_mpnn --ligmpnn_params params/ligmpnn/v_32_030_25.pkl
```

With `pmpnn` (the default, unchanged) the amino-acid-preference readout measures what the
binder's backbone *shape* favours, because ProteinMPNN never sees the target. With
`ligand_mpnn` the target's predicted atoms are supplied as chemical context, so the same
readout also reflects the ligand's chemistry. Running both at the same seed isolates that
one term.

## Verifying a change

There is no test suite — running it is how a change is validated.
`scripts/test_ligand_mpnn.py` covers the port:

```bash
# numerical parity against the reference implementation (needs the clone below)
git clone --depth 1 https://github.com/dauparas/LigandMPNN dependencies/LigandMPNN
JAX_PLATFORMS=cpu python scripts/test_ligand_mpnn.py --mode parity \
    --pdb data/ligand_test/1STP.pdb --ligand_res BTN

# everything else
python scripts/test_ligand_mpnn.py --mode all --pdb data/ligand_test/1STP.pdb --ligand_res BTN
```

`--mode parity` compares the neighbour graph, edge features, ligand node and graph features,
the node embeddings after the ligand context, and the logits. `--mode smoke` checks that the
ligand actually raises native sequence recovery in the pocket; the other modes cover
sampling, jit/vmap behaviour and `DesignData` integration.

## Scope

Only the `ligand_mpnn` model type is ported. The side-chain packer
(`ligandmpnn_sc_v_32_002_16`) and the membrane variants are not, and
`ligand_mpnn_use_side_chain_context` is not implemented — side-chain atoms of neighbouring
residues are never used as context.

One deliberate numerical difference from the reference: masked ligand atoms are pushed to
`MASKED_DISTANCE_SQ` in the context-atom gather, where upstream uses `1000.0`. That value is
a *squared* distance, i.e. only 31.6 Å, so once the atom list is padded to a fixed size a
masked padding atom would outrank a genuine atom further away than that and displace it from
the gather. Upstream never hits this because it pads only after gathering. For an unpadded
ligand the two agree exactly, which the parity test checks.
