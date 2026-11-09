"""This module implements a jax version of LigandMPNN, ported from the reference PyTorch
implementation at github.com/dauparas/LigandMPNN.

LigandMPNN is ProteinMPNN plus a ligand-context branch: the protein encoder runs unchanged,
then a small graph network over the nearest non-protein atoms around each residue contributes
a residual to the node embeddings, and the decoder is identical to ProteinMPNN's. Because the
ligand only enters as that residual, this module reuses `flexcraft.sequence.mpnn`'s layers and
its single-pass, order-free decoder verbatim - so `flexcraft.sequence.sample.sample`,
`flexcraft.protocols.af2cycler.af_cycler` and
`flexcraft.hallucination.losses.sequence.mpnn_recovery` all work with it unchanged.

Ligand atoms are carried on a `DesignData` in three scene-level fields - `ligand_positions`,
`ligand_elements` and `ligand_mask` - which `flexcraft.files.ligand` produces and
`DesignData.with_ligand` attaches. Pad them to a fixed atom count so `jax.jit` does not
recompile for every ligand.

LigandMPNN uses the same amino acid order as ProteinMPNN (`aa_codes.PMPNN_CODE` with index 20
for the unknown token), so logits need no new alphabet handling - translate them with
`aa_codes.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)` as for ProteinMPNN.
"""

import joblib
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp

from flexcraft.data.data import DesignData
import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import (
    Gelu,
    cat_neighbors_nodes,
    get_sep_mask,
    drop,
    EncLayer,
    DecLayer,
    PositionWiseFeedForward,
    ProteinFeatures,
    EmbedToken,
)

# Number of classes LigandMPNN one-hot encodes element identity into. Note that the model's
# group/period table only covers 119 elements, so an atomic number must be clamped to
# `< NUM_PERIODIC_ELEMENTS` before a group or period lookup even though the element one-hot
# itself is NUM_ELEMENTS wide.
NUM_ELEMENTS = 120
NUM_PERIODIC_ELEMENTS = 119
NUM_GROUPS = 19
NUM_PERIODS = 8

# Periodic table group and period per atomic number, taken verbatim from
# `ProteinFeaturesLigand.periodic_table_features` in the reference implementation. Index 0 is
# the unknown element. These are plain tensor attributes upstream rather than registered
# buffers, so they are *not* present in the released checkpoints and must live here.
ELEMENT_GROUP = (
    0, 1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
)
ELEMENT_PERIOD = (
    0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
)

LIGAND_KEYS = ("ligand_positions", "ligand_elements", "ligand_mask")

# Squared distance assigned to masked ligand atoms so that they sort behind every real atom.
# The reference implementation uses 1000.0 here, but that is a squared distance - only 31.6
# Angstrom - so a masked atom would outrank a genuine atom further away than that and displace
# it from the gather. Upstream never notices because it pads the context only *after* gathering,
# whereas padding the atom list to a fixed size (which is what keeps jax.jit from recompiling
# per ligand) puts masked atoms into the sort. A sentinel beyond any real distance makes the
# gather return the nearest valid atoms regardless of how the input was padded.
MASKED_DISTANCE_SQ = 1e6


def _normalize(x, axis=-1, eps=1e-12):
    """L2-normalize along an axis, matching `torch.nn.functional.normalize`."""
    norm = jnp.sqrt(jnp.square(x).sum(axis=axis, keepdims=True))
    return x / jnp.maximum(norm, eps)


def nearest_context_atoms(cb, mask, positions, elements, ligand_mask,
                          num_context, eps=1e-6):
    """Gather the nearest ligand atoms around each residue.

    Port of `get_nearest_neighbours` in the reference implementation's `data_utils.py`. That
    version sorts squared pseudo-CB-to-atom distances and takes the first `num_context`.
    `jax.lax.top_k` resolves ties by lowest index just as `torch.argsort` does, so this
    reproduces the reference gather, including the order of the masked tail. The one
    deliberate difference is `MASKED_DISTANCE_SQ`: see its comment for why the reference's
    1000.0 cannot be used once the atom list is padded to a fixed size.

    Args:
        cb: (L, 3) virtual CB positions.
        mask: (L,) residue mask.
        positions: (A, 3) ligand atom positions.
        elements: (A,) ligand atomic numbers.
        ligand_mask: (A,) ligand atom mask.
        num_context: static number of context atoms to gather per residue.
        eps: epsilon added under the square root of the closest distance.

    Returns:
        Tuple of (positions, elements, mask, distance):
            positions: (L, num_context, 3) context atom positions per residue.
            elements: (L, num_context) context atomic numbers per residue.
            mask: (L, num_context) float32 context atom mask per residue.
            distance: (L,) distance to the closest valid ligand atom.
    """
    num_context = int(num_context)
    dtype = positions.dtype
    pair_mask = mask.astype(dtype)[:, None] * ligand_mask.astype(dtype)[None, :]
    distance_sq = jnp.square(cb[:, None, :] - positions[None, :, :]).sum(-1)
    distance_sq = distance_sq * pair_mask + (1.0 - pair_mask) * MASKED_DISTANCE_SQ

    # pad to at least num_context atoms so top_k always has enough entries to return.
    # padded atoms are masked, so they sort behind every valid atom.
    num_atoms = positions.shape[0]
    if num_atoms < num_context:
        padding = num_context - num_atoms
        distance_sq = jnp.concatenate(
            [distance_sq,
             jnp.full((cb.shape[0], padding), MASKED_DISTANCE_SQ, dtype=distance_sq.dtype)],
            axis=1)
        positions = jnp.concatenate(
            [positions, jnp.zeros((padding, 3), dtype=dtype)], axis=0)
        elements = jnp.concatenate(
            [elements, jnp.zeros((padding,), dtype=elements.dtype)], axis=0)
        ligand_mask = jnp.concatenate(
            [ligand_mask, jnp.zeros((padding,), dtype=ligand_mask.dtype)], axis=0)

    negative, index = jax.lax.top_k(-distance_sq, num_context)
    return (positions[index], elements[index],
            ligand_mask[index].astype(jnp.float32),
            jnp.sqrt(-negative[:, 0] + eps))


class DecLayerJ(hk.Module):
    """A decoder layer over the ligand atom graph.

    Identical to `flexcraft.sequence.mpnn.DecLayer` except that it operates one rank higher:
    nodes are (L, M, C) ligand atoms and edges are (L, M, M, C) atom pairs. `DecLayer` cannot
    be reused directly because it builds its node broadcast with a rank-3 `jnp.tile`.
    """

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None):
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = dropout
        self.norm1 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm1"
        )
        self.norm2 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm2"
        )

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + "_W1")
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + "_W2")
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + "_W3")
        self.act = Gelu
        self.dense = PositionWiseFeedForward(
            num_hidden, num_hidden * 4, name=name + "_dense"
        )

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # concatenate h_V_i to h_E_ij, broadcasting over the trailing edge axis
        h_V_expand = jnp.broadcast_to(
            jnp.expand_dims(h_V, -2), h_E.shape[:-1] + (h_V.shape[-1],))
        h_EV = jnp.concatenate([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message
        dh = jnp.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + drop(dh, rate=self.dropout))

        # position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + drop(dh, rate=self.dropout))

        if mask_V is not None:
            h_V = jnp.expand_dims(mask_V, -1) * h_V
        return h_V


class ProteinFeaturesLigand(ProteinFeatures):
    """Protein features plus ligand context features.

    Inherits the backbone edge featurization - the k-nearest-neighbour graph, the 25 atom-pair
    RBFs and the relative position encoding - from `ProteinFeatures`, whose pair table is
    identical to the reference implementation's. Adds the ligand side: per-residue nearest
    context atoms, element embeddings, ligand-to-backbone node features and the ligand atom
    graph.
    """

    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=32, augment_eps=0.0, atom_context_num=25,
                 cutoff_for_score=8.0):
        # the parent is built with augment_eps=0: coordinate noise is applied here instead, so
        # that the backbone and the ligand are perturbed together rather than independently.
        super(ProteinFeaturesLigand, self).__init__(
            edge_features, node_features,
            num_positional_embeddings=num_positional_embeddings,
            num_rbf=num_rbf, top_k=top_k, augment_eps=0.0)
        self.ligand_augment_eps = augment_eps
        self.atom_context_num = atom_context_num
        self.cutoff_for_score = cutoff_for_score

        self.node_project_down = hk.Linear(
            node_features, with_bias=True, name="node_project_down")
        self.norm_nodes = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name="norm_nodes")
        self.type_linear = hk.Linear(64, with_bias=True, name="type_linear")
        self.y_nodes = hk.Linear(node_features, with_bias=False, name="y_nodes")
        self.y_edges = hk.Linear(node_features, with_bias=False, name="y_edges")
        self.norm_y_nodes = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name="norm_y_nodes")
        self.norm_y_edges = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name="norm_y_edges")

    def _get_edge_idx(self, X, mask, batch, eps=1e-6):
        """get edge index
        input: mask.shape = (...,L), X.shape = (...,L,3)
        return: (...,L,k)

        Overrides the parent's approximate nearest-neighbour search: the reference
        implementation uses an exact top-k, and `jax.lax.approx_min_k` can return a different
        neighbour set, which every downstream feature then inherits.
        """
        same_batch = batch[:, None] == batch[None, :]
        mask_2D = ((mask[..., None, :] * mask[..., :, None]) * same_batch).astype(X.dtype)
        dX = X[..., None, :, :] - X[..., :, None, :]
        D = mask_2D * jnp.sqrt(jnp.square(dX).sum(-1) + eps)
        D = D + (1.0 - mask_2D) * D.max(-1, keepdims=True)
        k = min(self.top_k, X.shape[-2])
        return jax.lax.top_k(-D, k)[1]

    def _make_angle_features(self, A, B, C, Y):
        """Orientation of each context atom in the local frame of a residue.

        Args:
            A, B, C: (L, 3) backbone N, CA and C positions.
            Y: (L, M, 3) context atom positions.

        Returns:
            (L, M, 4) array of direction cosines in the residue's local frame.
        """
        v1, v2 = A - B, C - B
        e1 = _normalize(v1)
        u2 = v2 - e1 * (e1 * v2).sum(-1, keepdims=True)
        e2 = _normalize(u2)
        e3 = jnp.cross(e1, e2)
        # (L, 3, 3), indexed [coordinate, basis vector]
        frame = jnp.stack([e1, e2, e3], axis=-1)
        local = jnp.einsum("lqp,lyq->lyp", frame, Y - B[:, None, :])

        rxy = jnp.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2 + 1e-8)
        rxyz = jnp.sqrt(jnp.square(local).sum(-1)) + 1e-8
        return jnp.stack([
            local[..., 0] / rxy,
            local[..., 1] / rxy,
            rxy / rxyz,
            local[..., 2] / rxyz,
        ], axis=-1)

    def __call__(self, data):
        for key in LIGAND_KEYS:
            if key not in data:
                raise KeyError(
                    f"LigandMPNN input is missing the {key!r} field. Attach ligand atoms "
                    "with DesignData.with_ligand(...), e.g. from "
                    "flexcraft.files.ligand.ligand_from_pdb. Use LigandAtoms.empty() for an "
                    "explicit apo run.")

        X = data["atom_positions"]
        # autoconvert atom14 and atom24 to atom4
        if X.shape[1] in (14, 24):
            X = X[:, :4]
        Y = data["ligand_positions"]

        if self.ligand_augment_eps > 0:
            X = X + self.ligand_augment_eps * jax.random.normal(
                hk.next_rng_key(), X.shape)
            Y = Y + self.ligand_augment_eps * jax.random.normal(
                hk.next_rng_key(), Y.shape)

        # backbone edge features, via the parent. A fresh dict is passed because
        # ProteinFeatures writes its "offset" cache into whatever it is handed.
        edge_data = dict(
            atom_positions=X,
            mask=data["mask"],
            batch_index=data["batch_index"],
            residue_index=data["residue_index"],
            chain_index=data["chain_index"])
        E, neighbours = super(ProteinFeaturesLigand, self).__call__(edge_data)

        # N, Ca, C, O and a virtual Cb
        N, Ca, C, O = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        b, c = Ca - N, C - Ca
        Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Ca

        # per-residue nearest context atoms
        Y, Y_t, Y_m, distance = nearest_context_atoms(
            Cb, data["mask"], Y, data["ligand_elements"], data["ligand_mask"],
            self.atom_context_num)

        # element identity: one-hot element, group and period
        Y_t = jnp.where((Y_t < 1) | (Y_t >= NUM_PERIODIC_ELEMENTS), 0, Y_t)
        group = jnp.asarray(ELEMENT_GROUP, dtype=jnp.int32)[Y_t]
        period = jnp.asarray(ELEMENT_PERIOD, dtype=jnp.int32)[Y_t]
        Y_t_one_hot = jnp.concatenate([
            jax.nn.one_hot(Y_t, NUM_ELEMENTS),
            jax.nn.one_hot(group, NUM_GROUPS),
            jax.nn.one_hot(period, NUM_PERIODS),
        ], axis=-1)

        # ligand-to-backbone node features: 5 * num_rbf distances, 64 element features
        # and 4 angle features
        def context_rbf(atom):
            distances = jnp.sqrt(
                jnp.square(atom[:, None, :] - Y).sum(-1) + 1e-6)
            return self._rbf(distances)

        node_features = jnp.concatenate([
            context_rbf(N), context_rbf(Ca), context_rbf(C),
            context_rbf(O), context_rbf(Cb),
            self.type_linear(Y_t_one_hot),
            self._make_angle_features(N, Ca, C, Y),
        ], axis=-1)
        V = self.norm_nodes(self.node_project_down(node_features))

        # ligand atom graph
        pair_distances = jnp.sqrt(
            jnp.square(Y[:, :, None, :] - Y[:, None, :, :]).sum(-1) + 1e-6)
        Y_edges = self.norm_y_edges(self.y_edges(self._rbf(pair_distances)))
        Y_nodes = self.norm_y_nodes(self.y_nodes(Y_t_one_hot))

        # which residues are close enough to the ligand to be considered pocket residues.
        # This is a reporting mask upstream, not a model input.
        contact = ((distance < self.cutoff_for_score)
                   * data["mask"].astype(jnp.float32) * Y_m[:, 0])
        return V, E, neighbours, Y_nodes, Y_edges, Y_m, distance, contact


class LigandMPNN(hk.Module):
    """LigandMPNN: ProteinMPNN conditioned on nearby non-protein atoms.

    The decoder matches `flexcraft.sequence.mpnn.ProteinMPNN`: a single pass which emits
    conditionally independent logits for every masked position at once, leaving the
    autoregression to `flexcraft.sequence.sample.sample`.
    """

    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=32,
        augment_eps=0.0,
        dropout=0.0,
        atom_context_num=25,
        num_context_encoder_layers=2,
        tie_messages=False,
        cutoff_for_score=8.0,
        return_intermediates=False,
    ):
        super(LigandMPNN, self).__init__(name="ligand_mpnn")

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.tie_messages = tie_messages
        self.return_intermediates = return_intermediates

        # Featurization layers
        self.features = ProteinFeaturesLigand(
            edge_features, node_features, top_k=k_neighbors, augment_eps=augment_eps,
            atom_context_num=atom_context_num, cutoff_for_score=cutoff_for_score)

        self.W_e = hk.Linear(hidden_dim, with_bias=True, name="W_e")
        self.W_s = EmbedToken(vocab_size=vocab, embed_dim=hidden_dim)

        # Encoder layers
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, name="enc" + str(i))
            for i in range(num_encoder_layers)
        ]

        # Decoder layers. These are built before the ligand context layers so that their
        # automatic Haiku module names (dec_layer, dec_layer_1, dec_layer_2) match
        # ProteinMPNN's; the context layers then take the names that follow. See
        # HAIKU_NAME_MAP, which is the single source of truth shared with the weight
        # converter.
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout, name="dec" + str(i))
            for i in range(num_decoder_layers)
        ]

        # Ligand context layers
        self.W_v = hk.Linear(hidden_dim, with_bias=True, name="W_v")
        self.W_c = hk.Linear(hidden_dim, with_bias=True, name="W_c")
        self.W_nodes_y = hk.Linear(hidden_dim, with_bias=True, name="W_nodes_y")
        self.W_edges_y = hk.Linear(hidden_dim, with_bias=True, name="W_edges_y")
        self.V_C = hk.Linear(hidden_dim, with_bias=False, name="V_C")
        self.V_C_norm = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name="V_C_norm")
        self.context_encoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout, name="ctx" + str(i))
            for i in range(num_context_encoder_layers)
        ]
        self.y_context_encoder_layers = [
            DecLayerJ(hidden_dim, hidden_dim, dropout=dropout, name="ydec" + str(i))
            for i in range(num_context_encoder_layers)
        ]

        self.W_out = hk.Linear(num_letters, with_bias=True, name="W_out")

    def __call__(self, data):
        L = data["atom_positions"].shape[0]

        # prepare node, edge and ligand context embeddings
        (V, E, neighbours, Y_nodes, Y_edges,
         Y_m, ligand_distance, ligand_contact) = self.features(data)
        local = jnp.zeros((E.shape[0], E.shape[-1]))
        pair = self.W_e(E)
        context = self.W_v(V)

        # tie messages
        tie_info = None
        if self.tie_messages and "tie_index" in data:
            tie_info = dict(tie_index=data["tie_index"], tie_weights=data["tie_weights"])

        # encoder layers
        mask_attend = jnp.take_along_axis(
            data["mask"][:, None] * data["mask"][None, :], neighbours, 1
        )
        for layer in self.encoder_layers:
            local, pair = layer(local, pair, neighbours,
                                data["mask"], mask_attend,
                                tie_info=tie_info)
        encoder_local = local

        # ligand context: alternate between updating the ligand atom graph and folding it
        # into the residue embeddings. Messages are not tied here - the reference
        # implementation has no analogue for tied ligand context.
        local_context = self.W_c(local)
        Y_m_edges = Y_m[..., :, None] * Y_m[..., None, :]
        projected_nodes = self.W_nodes_y(Y_nodes)
        projected_edges = self.W_edges_y(Y_edges)
        for y_layer, layer in zip(self.y_context_encoder_layers,
                                  self.context_encoder_layers):
            projected_nodes = y_layer(
                projected_nodes, projected_edges, Y_m, Y_m_edges)
            local_context = layer(
                local_context, jnp.concatenate([context, projected_nodes], -1),
                data["mask"], Y_m)
        local = local + self.V_C_norm(
            drop(self.V_C(local_context), rate=self.dropout))
        h_V_prev = local

        def step(x, order, filled):
            autoregressive_mask = get_sep_mask(order, filled)
            mask_attend = jnp.take_along_axis(autoregressive_mask, neighbours, 1)
            mask_1D = data["mask"][:, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1 - mask_attend)

            local = h_V_prev
            h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(local), pair, neighbours)
            h_EXV_encoder = cat_neighbors_nodes(local, h_EX_encoder, neighbours)
            h_EXV_encoder = mask_fw[..., None] * h_EXV_encoder
            h_ES = cat_neighbors_nodes(x["seq"], pair, neighbours)

            # decoder layers
            for l, layer in enumerate(self.decoder_layers):
                local = x["local"][l]
                h_ESV_decoder = cat_neighbors_nodes(local, h_ES, neighbours)
                h_ESV = mask_bw[..., None] * h_ESV_decoder + h_EXV_encoder
                local = layer(local, h_ESV, mask_V=data["mask"], tie_info=tie_info)
                # update
                x["local"] = x["local"].at[l + 1].set(local)

            x["logits"] = jax.nn.log_softmax(self.W_out(local), axis=-1)
            return x

        # initial values
        sequence_one_hot = jax.nn.one_hot(data["aa"], 21, axis=-1)
        filled = data["aa"] != 20
        X = {
            "seq": jnp.where(filled[:, None], self.W_s(sequence_one_hot), 0),
            "local": jnp.array(
                [local] + [jnp.zeros_like(local)] * len(self.decoder_layers)
            ),
            "aa": sequence_one_hot,
            "logits": jnp.zeros((L, 21)),
        }

        # sample a random decoding order for filled positions; unfilled positions cannot
        # attend to each other, so all masked logits are produced in a single pass
        order = jax.random.permutation(hk.next_rng_key(), data["aa"].shape[0])
        X = step(X, order, filled)
        ent = jnp.sum(-jnp.exp(X["logits"]) * X["logits"], axis=-1)
        var_ent = jnp.sum(
            jnp.exp(X["logits"]) * (X["logits"] + ent[..., None]) ** 2, axis=-1
        )

        result = {"aa": X["aa"], "logits": X["logits"], "ent": ent, "var_ent": var_ent,
                  "ligand_distance": ligand_distance,
                  "ligand_contact_mask": ligand_contact}
        if self.return_intermediates:
            # debugging aid, used by scripts/test_ligand_mpnn.py to compare against the
            # reference implementation layer by layer
            result["intermediates"] = dict(
                E=E, V=V, neighbours=neighbours, Y_m=Y_m,
                Y_nodes=Y_nodes, Y_edges=Y_edges,
                encoder_local=encoder_local, local=h_V_prev)
        return result


# ============================== parameter naming ==============================
# Haiku derives a module's scope name from its class and a per-name counter, in construction
# order. Those names are baked into the converted checkpoints, so they are generated here once
# and shared with scripts/convert_ligandmpnn_params.py rather than hard-coded in both places.

MODULE_SCOPE = "ligand_mpnn"
FEATURES_SCOPE = f"{MODULE_SCOPE}/~/protein_features_ligand"


def _auto_name(base, index):
    """Reproduce Haiku's automatic module naming: base, base_1, base_2, ..."""
    return base if index == 0 else f"{base}_{index}"


def _layer_entries(torch_stack, scope, prefix, leaves):
    """Build the name map entries for one stack of identical layers.

    Args:
        torch_stack: name of the PyTorch ModuleList, e.g. "encoder_layers".
        scope: Haiku module scope for this layer, e.g. "enc_layer_1".
        prefix: leaf name prefix used in the Haiku module, e.g. "enc1".
        leaves: leaf attribute names present on the layer.

    Returns:
        Dictionary mapping PyTorch module names to Haiku parameter module names.
    """
    entries = {}
    for leaf in leaves:
        entries[f"{torch_stack}.{leaf}"] = f"{MODULE_SCOPE}/~/{scope}/~/{prefix}_{leaf}"
    for leaf in ("W_in", "W_out"):
        entries[f"{torch_stack}.dense.{leaf}"] = (
            f"{MODULE_SCOPE}/~/{scope}/~/position_wise_feed_forward/~/"
            f"{prefix}_dense_{leaf}")
    return entries


def _build_name_map(num_encoder_layers=3, num_decoder_layers=3,
                    num_context_encoder_layers=2):
    """Map every PyTorch module name in a LigandMPNN checkpoint to its Haiku counterpart."""
    name_map = {
        "features.embeddings.linear":
            f"{FEATURES_SCOPE}/~/positional_encodings/~/embedding_linear",
        "features.edge_embedding": f"{FEATURES_SCOPE}/~/edge_embedding",
        "features.norm_edges": f"{FEATURES_SCOPE}/~/norm_edges",
        "features.node_project_down": f"{FEATURES_SCOPE}/~/node_project_down",
        "features.norm_nodes": f"{FEATURES_SCOPE}/~/norm_nodes",
        "features.type_linear": f"{FEATURES_SCOPE}/~/type_linear",
        "features.y_nodes": f"{FEATURES_SCOPE}/~/y_nodes",
        "features.y_edges": f"{FEATURES_SCOPE}/~/y_edges",
        "features.norm_y_nodes": f"{FEATURES_SCOPE}/~/norm_y_nodes",
        "features.norm_y_edges": f"{FEATURES_SCOPE}/~/norm_y_edges",
        "W_s": f"{MODULE_SCOPE}/~/embed_token",
    }
    for leaf in ("W_e", "W_v", "W_c", "W_nodes_y", "W_edges_y",
                 "V_C", "V_C_norm", "W_out"):
        name_map[leaf] = f"{MODULE_SCOPE}/~/{leaf}"

    encoder_leaves = ("W1", "W2", "W3", "W11", "W12", "W13", "norm1", "norm2", "norm3")
    decoder_leaves = ("W1", "W2", "W3", "norm1", "norm2")

    # DecLayer instances share one Haiku counter across the decoder and context stacks, in
    # the order LigandMPNN.__init__ builds them: decoder first, then context.
    for i in range(num_encoder_layers):
        name_map.update(_layer_entries(
            f"encoder_layers.{i}", _auto_name("enc_layer", i), f"enc{i}", encoder_leaves))
    for i in range(num_decoder_layers):
        name_map.update(_layer_entries(
            f"decoder_layers.{i}", _auto_name("dec_layer", i), f"dec{i}", decoder_leaves))
    for i in range(num_context_encoder_layers):
        name_map.update(_layer_entries(
            f"context_encoder_layers.{i}",
            _auto_name("dec_layer", num_decoder_layers + i), f"ctx{i}", decoder_leaves))
    for i in range(num_context_encoder_layers):
        name_map.update(_layer_entries(
            f"y_context_encoder_layers.{i}",
            _auto_name("dec_layer_j", i), f"ydec{i}", decoder_leaves))
    return name_map


HAIKU_NAME_MAP = _build_name_map()


# ============================== construction ==============================

DEFAULT_CONFIG = dict(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    vocab=21,
    num_context_encoder_layers=2,
)


def dummy_input(length=8, num_atoms=25):
    """Build a minimal valid LigandMPNN input, for parameter initialization and checks.

    Args:
        length: number of residues.
        num_atoms: number of ligand atoms.

    Returns:
        Dictionary of input arrays.
    """
    return dict(
        aa=jnp.full((length,), 20, dtype=jnp.int32),
        atom_positions=jnp.zeros((length, 14, 3), dtype=jnp.float32),
        atom_mask=jnp.ones((length, 14), dtype=jnp.bool_),
        residue_index=jnp.arange(length, dtype=jnp.int32),
        chain_index=jnp.zeros((length,), dtype=jnp.int32),
        batch_index=jnp.zeros((length,), dtype=jnp.int32),
        tie_index=jnp.arange(length, dtype=jnp.int32),
        tie_weights=jnp.ones((length,), dtype=jnp.float32),
        mask=jnp.ones((length,), dtype=jnp.bool_),
        ligand_positions=jnp.zeros((num_atoms, 3), dtype=jnp.float32),
        ligand_elements=jnp.full((num_atoms,), 6, dtype=jnp.int32),
        ligand_mask=jnp.ones((num_atoms,), dtype=jnp.bool_))


def assert_params_match(expected, actual):
    """Check that a loaded parameter dict matches a freshly initialized one.

    Haiku names its modules by construction order, so a misnamed scope or a transposed
    weight loads silently wrong. This compares key sets and shapes and raises instead.

    Args:
        expected: parameter dict from `hk.transform(...).init`.
        actual: parameter dict loaded from a checkpoint.

    Raises:
        ValueError: if any parameter is missing, unexpected or of the wrong shape.
    """
    want = {(module, leaf): tuple(value.shape)
            for module, leaves in expected.items() for leaf, value in leaves.items()}
    have = {(module, leaf): tuple(np.shape(value))
            for module, leaves in actual.items() for leaf, value in leaves.items()}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    mismatched = {key: (want[key], have[key])
                  for key in sorted(set(want) & set(have)) if want[key] != have[key]}
    if missing or extra or mismatched:
        raise ValueError(
            "LigandMPNN parameters do not match the model:\n"
            f"  missing ({len(missing)}): {missing}\n"
            f"  unexpected ({len(extra)}): {extra}\n"
            f"  wrong shape ({len(mismatched)}): {mismatched}")


def make_ligmpnn(params_path, num_neighbours=None, atom_context_num=None, eps=0.0,
                 tie_messages=False, cutoff_for_score=8.0, split_params=False,
                 standard_aa_code=False, check_params=True,
                 return_intermediates=False):
    """Build a LigandMPNN model from a converted checkpoint.

    The returned callable mirrors `flexcraft.sequence.mpnn.make_pmpnn`, so it can be used
    anywhere ProteinMPNN is - with the one additional requirement that its input carries
    ligand atom fields (see `DesignData.with_ligand`).

    Args:
        params_path: path to a converted LigandMPNN checkpoint, as produced by
            `scripts/convert_ligandmpnn_params.py`.
        num_neighbours: number of neighbours in the residue graph. Defaults to the
            checkpoint's own value (32 for the released LigandMPNN weights, which differs
            from ProteinMPNN's 48 - do not assume ProteinMPNN's default here).
        atom_context_num: number of ligand context atoms per residue. Defaults to the
            checkpoint's value (25).
        eps: standard deviation of coordinate noise added to backbone and ligand atoms.
        tie_messages: tie encoder and decoder messages across tied positions.
        cutoff_for_score: distance in Angstrom within which a residue counts as a pocket
            residue in the returned "ligand_contact_mask".
        split_params: return the apply function and parameters separately, for use with
            `batch_ligand_mpnn` or inside a larger parameter dict.
        standard_aa_code: if True, translate input sequences from the AF2 amino acid order
            into ProteinMPNN order inside the model.
        check_params: verify that the checkpoint's parameters match the model's.
        return_intermediates: also return per-layer activations under an "intermediates"
            key. A debugging aid for comparing against the reference implementation.

    Returns:
        If `split_params` is False, a function mapping (key, data) to a dictionary with
        "aa", "logits", "ent", "var_ent", "ligand_distance" and "ligand_contact_mask".
        Otherwise a tuple of that function's parameter-taking form and the parameters.
    """
    checkpoint = joblib.load(params_path)
    params = checkpoint["model_state_dict"]
    if num_neighbours is None:
        num_neighbours = int(checkpoint["num_edges"])
    if atom_context_num is None:
        atom_context_num = int(checkpoint["atom_context_num"])

    config = dict(
        DEFAULT_CONFIG,
        k_neighbors=num_neighbours,
        augment_eps=eps,
        dropout=0,
        atom_context_num=atom_context_num,
        tie_messages=tie_messages,
        cutoff_for_score=cutoff_for_score,
        return_intermediates=return_intermediates)

    def inner(x):
        if standard_aa_code:
            if isinstance(x, DesignData):
                x = x.to_dict()
            x = {key: value for key, value in x.items()}
            x["aa"] = aas.translate(x["aa"], aas.AF2_CODE, aas.PMPNN_CODE)
        return LigandMPNN(**config)(x)

    run_ligmpnn = hk.transform(inner).apply
    if check_params:
        expected = hk.transform(inner).init(
            jax.random.PRNGKey(0), dummy_input(num_atoms=atom_context_num))
        assert_params_match(expected, params)
    if split_params:
        return run_ligmpnn, params
    return lambda k, d: run_ligmpnn(params, k, d)


# Expected rank of each DesignData field for a single, unbatched design. A field whose rank
# is one higher than this is taken to carry a leading sample axis.
UNBATCHED_RANK = {
    "aa": 1,
    "atom_positions": 3,
    "atom_mask": 2,
    "mask": 1,
    "residue_index": 1,
    "chain_index": 1,
    "batch_index": 1,
    "tie_index": 1,
    "tie_weights": 1,
    "plddt": 1,
    "dssp": 1,
    "aa_condition": 1,
    "is_target": 1,
    "is_binder": 1,
    "aa_profile": 2,
    "offset": 2,
    "ligand_positions": 2,
    "ligand_elements": 1,
    "ligand_mask": 1,
}


def batch_ligand_mpnn(mpnn):
    """Map a model over a batch of structures sharing one sequence and ligand.

    Equivalent in purpose to `flexcraft.sequence.mpnn.batch_mpnn`, but it maps over the
    sample axis with a per-field `in_axes` instead of replicating every field, so shared
    fields are not materialised once per sample and nothing is silently dropped - including
    `tie_index`/`tie_weights` and the ligand fields, which `batch_mpnn` discards. A ligand
    may be shared across samples, with shape (A, 3), or vary per sample, with shape
    (S, A, 3).

    Args:
        mpnn: a model in the `split_params=True` form, i.e. (params, key, data) -> result.

    Returns:
        A function with the same signature which accepts a batched `atom_positions`.
    """
    def _mpnn_map(params, key, data):
        positions = data["atom_positions"]
        if positions.ndim != 4:
            return mpnn(params, key, data)
        num_samples = positions.shape[0]
        data = {name: value for name, value in data.items()}
        in_axes = {}
        for name, value in data.items():
            if name not in UNBATCHED_RANK:
                if jnp.ndim(value) >= 2:
                    raise KeyError(
                        f"batch_ligand_mpnn cannot tell whether the field {name!r} of rank "
                        f"{jnp.ndim(value)} carries a sample axis. Add it to "
                        "flexcraft.sequence.ligand_mpnn.UNBATCHED_RANK.")
                in_axes[name] = None
                continue
            in_axes[name] = 0 if jnp.ndim(value) == UNBATCHED_RANK[name] + 1 else None
        keys = jax.random.split(key, num_samples)
        return jax.vmap(mpnn, (None, 0, in_axes), 0)(params, keys, data)
    return _mpnn_map
