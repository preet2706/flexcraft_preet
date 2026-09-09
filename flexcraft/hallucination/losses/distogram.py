from typing import Any

import jax
import jax.numpy as jnp

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData

def distogram_contacts(result: Any, target: DesignData,
                       contact_distance = 8.0) -> jax.Array:
    is_contact = target.contacts(
        contact_distance=contact_distance, atom_index = 4)
    entropy = result.contact_entropy(
        contact_distance = contact_distance)
    result = is_contact * entropy
    if is_contact.ndim == 3:
        result = result.mean(axis=0)
    return result

DISTOGRAM_BIN_EDGES = jnp.linspace(2.0, 22.0, 65)  # must match JoltzResult.distogram_bin_edges

def blueprint_distogram(target: DesignData) -> jax.Array:
    """One-hot distogram of a blueprint structure, binned like JoltzResult.distogram_bin_edges
    (64 bins, 2-22 A). Uses Cb for every residue except glycine, which has no real Cb and
    uses Ca instead -- the standard AlphaFold "CB, CA for glycine" contact-atom convention.
    (`target.atom5`'s CB is an idealized geometric construction from the backbone -- see
    positions_to_ncacocb -- so it's well-defined even for Gly, just not the real atom.)"""
    atom5 = target.atom5  # (L, 5, 3): N, CA, C, O, idealized-CB
    ca = atom5[:, 1]
    cb = atom5[:, 4]
    is_gly = target.aa == aas.AF2_CODE.index("G")
    contact_atom = jnp.where(is_gly[:, None], ca, cb)
    dist = jnp.linalg.norm(
        contact_atom[:, None, :] - contact_atom[None, :, :], axis=-1)
    num_bins = DISTOGRAM_BIN_EDGES.shape[0] - 1
    bins = jnp.digitize(dist, DISTOGRAM_BIN_EDGES[1:-1])
    return jax.nn.one_hot(bins, num_bins)

def distogram_nll(result: Any, target_distogram: jax.Array) -> jax.Array:
    """Cross-entropy between a precomputed blueprint distogram and the predicted
    distogram. Mirrors JoltzResult.sample_distogram_nll, but against an external
    target instead of the model's own sampled coordinates."""
    return -(target_distogram * result.log_distogram).sum(axis=-1).mean()
