from typing import Any

import jax
import jax.numpy as jnp

import flexcraft.sequence.aa_codes as aas
from flexcraft.structure.boltz._result import JoltzResult

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
def residue_helix_contacts(result: JoltzResult):
    resi = result.residue_index
    chain = result.chain_index
    mask = resi[:, None] - resi[None, :] == 3
    mask *= chain[:, None] == chain[None, :]
    entropy = result.contact_probability(contact_distance = 6.0)
    result = -jnp.log((entropy * mask + 1e-8).sum(axis=-1))
    return result

def helix_contacts(result: JoltzResult):
    return residue_helix_contacts(result).mean()

def distogram_jsd(x: jax.Array, y: jax.Array) -> jax.Array:
    kl_fw = -(jnp.exp(x) * y).sum(axis=-1)
    kl_rv = -(jnp.exp(y) * x).sum(axis=-1)
    jsd = (kl_fw + kl_rv) / 2
    return jsd

def _mask_residue_distance(x, residue_index, min_residue_distance):
    mask = jnp.ones_like(x)
    if residue_index is not None:
        distance = abs(residue_index[:, None] - residue_index[None, :])
        mask = distance >= min_residue_distance
    return mask

def mean_distogram_jsd(x: jax.Array, y: jax.Array,
                       residue_index: jax.Array = None,
                       min_residue_distance: int = 0) -> jax.Array:
    jsd = distogram_jsd(x, y)
    mask = _mask_residue_distance(
        jsd, residue_index, min_residue_distance)
    jsd *= mask
    result = jsd.sum() / jnp.maximum(mask.sum(), 1)
    return result

def max_distogram_jsd(x: jax.Array, y: jax.Array,
                      residue_index: jax.Array = None,
                      min_residue_distance: int = 0) -> jax.Array:
    jsd = distogram_jsd(x, y)
    mask = _mask_residue_distance(
        jsd, residue_index, min_residue_distance)
    jsd *= mask
    return jsd.max(axis=1).mean()

def topk_distogram_jsd(x: jax.Array, y: jax.Array,
                       k: int = 10,
                       residue_index: jax.Array = None,
                       min_residue_distance: int = 0) -> jax.Array:
    jsd = distogram_jsd(x, y)
    mask = _mask_residue_distance(
        jsd, residue_index, min_residue_distance)
    jsd *= mask
    jsd = jnp.sort(jsd, axis=1)[:, -k:].mean(axis=1).mean()
    return jsd

def differential_contact_entropy(x: JoltzResult, y: JoltzResult,
                                 x_slice=slice(None), y_slice=slice(None),
                                 k=10, contact_distance=10.0):
    x_entropy = x.contact_entropy(contact_distance=contact_distance)
    x_entropy = x_entropy[x_slice, x_slice]
    y_entropy = y.contact_entropy(contact_distance=contact_distance)
    y_entropy = y_entropy[y_slice, y_slice]
    delta = (x_entropy - y_entropy)
    sort_index = jnp.argsort(delta, axis=1)
    index = jnp.arange(sort_index.shape[0])[:, None]
    x_targets = sort_index[:, :k]
    y_targets = sort_index[:, -k:]
    return (x_entropy[index, x_targets].mean() + y_entropy[index, y_targets].mean()) / 2

def contact_jsd(x: JoltzResult, y: JoltzResult,
                contact_distance=10.0) -> jax.Array:
    edge_mask = x.distogram_bin_edges[1:] < contact_distance
    x_distogram_clipped = jax.nn.softmax(x.log_distogram - 1e9 * (1 - edge_mask), axis=-1)
    x_distogram_clipped = jnp.where(edge_mask, x_distogram_clipped, 0)
    kl_fw = -(x_distogram_clipped * y.log_distogram).sum(axis=-1)
    y_distogram_clipped = jax.nn.softmax(y.log_distogram - 1e9 * (1 - edge_mask), axis=-1)
    y_distogram_clipped = jnp.where(edge_mask, y_distogram_clipped, 0)
    kl_rv = -(y_distogram_clipped * x.log_distogram).sum(axis=-1)
    return (kl_fw + kl_rv) / 2

# TODO: radius of gyration, etc.
