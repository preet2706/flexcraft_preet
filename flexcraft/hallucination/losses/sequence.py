import jax
import jax.numpy as jnp
import numpy as np
from flexcraft.data.data import DesignData
from flexcraft.structure.boltz import JoltzResult
from flexcraft.sequence.mpnn import batch_mpnn
import flexcraft.sequence.aa_codes as aas

def mpnn_recovery(pmpnn, params, batcher=batch_mpnn):
    """Sequence recovery of a hallucinated sequence under a ProteinMPNN-like model.

    Args:
        pmpnn: a sequence design model in the `split_params=True` form.
        params: the model's parameters.
        batcher: function mapping the model to one which accepts a batched
            `atom_positions`. Pass `flexcraft.sequence.ligand_mpnn.batch_ligand_mpnn` for
            LigandMPNN - the default `batch_mpnn` drops every field outside its hard-coded
            list, including the ligand.

    Returns:
        A function mapping (key, sequence, result, selection, temperature) to the recovery
        and the centered logits.
    """
    def _inner(key, sequence: jax.Array, result: JoltzResult | DesignData, selection, temperature=0.1):
        if not isinstance(result, DesignData):
            result = result.to_data(return_samples=True)
        if isinstance(selection, int):
            selection = slice(0, selection)
        pmpnn_input = result.update(aa=aas.translate(result["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
        is_designable = jnp.zeros((pmpnn_input.aa.shape[0],), dtype=jnp.bool_).at[selection].set(True)
        logits = batcher(pmpnn)(params, key, pmpnn_input.drop_aa(where=is_designable))["logits"]
        logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
        if not result.is_single_sample:
            logits = logits.mean(axis=0)
        logits = logits[selection]
        center = logits.mean(axis=0)
        logits = logits - center
        prob = jax.lax.stop_gradient(jax.nn.softmax(
            logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / temperature, axis=-1))
        recovery = (prob * sequence).sum(axis=-1).mean()
        return recovery, logits
    return _inner
