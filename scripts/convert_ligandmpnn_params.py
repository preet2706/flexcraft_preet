"""Convert an official LigandMPNN PyTorch checkpoint into flexcraft's parameter format.

The released LigandMPNN weights are PyTorch state dicts; flexcraft loads joblib pickles
holding a flat Haiku parameter dict (see `flexcraft.sequence.mpnn.load_params`). This script
bridges the two. It never imports the reference implementation - only the tensors.

Download the checkpoints first:

    mkdir -p params/ligmpnn/raw && cd params/ligmpnn/raw
    for n in 005 010 020 030; do
        wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_${n}_25.pt
    done

Then convert each one:

    python scripts/convert_ligandmpnn_params.py \
        --in_path params/ligmpnn/raw/ligandmpnn_v_32_030_25.pt \
        --out_path params/ligmpnn/v_32_030_25.pkl

Module names are taken from `flexcraft.sequence.ligand_mpnn.HAIKU_NAME_MAP` rather than spelled
out here, so the model and the converter cannot drift apart. With `--check True` (the default)
the result is validated against a freshly initialized model before it is written.
"""

import os

import joblib
import numpy as np

from flexcraft.utils.options import parse_options
from flexcraft.sequence.ligand_mpnn import HAIKU_NAME_MAP


def torch_to_haiku(state_dict):
    """Convert a LigandMPNN PyTorch state dict into a Haiku parameter dict.

    Conversion rules:
      - `Linear.weight` is stored as (out, in) by PyTorch and (in, out) by Haiku, so it is
        transposed and named "w"; `Linear.bias` becomes "b".
      - `LayerNorm.weight`/`bias` become "scale"/"offset".
      - `Embedding.weight` becomes the "W_s" parameter of `EmbedToken` and is *not*
        transposed, since `EmbedToken` contracts the one-hot against its first axis.
      - Linear and LayerNorm are told apart by the rank of the module's own weight, so no
        hard-coded list of layer names is needed.

    Args:
        state_dict: mapping of PyTorch parameter names to tensors.

    Returns:
        Tuple of (params, skipped):
            params: nested dict of {haiku module name: {leaf name: array}}.
            skipped: sorted list of state dict keys which held no parameters.
    """
    arrays = {name: np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
              for name, tensor in state_dict.items()}
    params = {}
    skipped = []
    for name, array in arrays.items():
        module, _, leaf = name.rpartition(".")
        if leaf not in ("weight", "bias"):
            # buffers and other non-parameter tensors
            skipped.append(name)
            continue
        if module not in HAIKU_NAME_MAP:
            raise KeyError(
                f"no Haiku counterpart for PyTorch module {module!r} (from {name!r}). "
                "Either the checkpoint is not a ligand_mpnn checkpoint, or "
                "flexcraft.sequence.ligand_mpnn.HAIKU_NAME_MAP is out of date.")
        haiku_module = HAIKU_NAME_MAP[module]

        if module == "W_s":
            # torch Embedding (vocab, dim) -> EmbedToken's W_s, same orientation
            key, value = "W_s", array
        elif arrays[f"{module}.weight"].ndim == 2:
            key, value = ("w", array.T) if leaf == "weight" else ("b", array)
        else:
            key, value = ("scale", array) if leaf == "weight" else ("offset", array)
        params.setdefault(haiku_module, {})[key] = value
    return params, sorted(skipped)


def main():
    opt = parse_options(
        "Convert an official LigandMPNN PyTorch checkpoint to flexcraft's format.",
        in_path="params/ligmpnn/raw/ligandmpnn_v_32_030_25.pt",
        out_path="params/ligmpnn/v_32_030_25.pkl",
        check="True")

    import torch

    checkpoint = torch.load(opt.in_path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "num_edges", "atom_context_num"):
        if key not in checkpoint:
            raise KeyError(
                f"{opt.in_path!r} has no {key!r}. Only ligand_mpnn checkpoints "
                "(ligandmpnn_v_32_*_25.pt) can be converted by this script.")
    num_edges = int(checkpoint["num_edges"])
    atom_context_num = int(checkpoint["atom_context_num"])
    print(f"{opt.in_path}: num_edges={num_edges} atom_context_num={atom_context_num} "
          f"noise_level={checkpoint.get('noise_level')}")

    params, skipped = torch_to_haiku(checkpoint["model_state_dict"])
    num_leaves = sum(len(leaves) for leaves in params.values())
    print(f"converted {num_leaves} parameters in {len(params)} modules"
          + (f", skipped {len(skipped)} non-parameter tensors: {skipped}" if skipped else ""))

    # every mapped module must have been filled, or a leaf was silently dropped
    unfilled = sorted(set(HAIKU_NAME_MAP.values()) - set(params))
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} expected Haiku modules got no parameters: {unfilled}")

    if opt.check == "True":
        import jax
        import haiku as hk
        from flexcraft.sequence.ligand_mpnn import (
            LigandMPNN, DEFAULT_CONFIG, dummy_input, assert_params_match)

        config = dict(DEFAULT_CONFIG, k_neighbors=num_edges, augment_eps=0.0, dropout=0,
                      atom_context_num=atom_context_num)
        expected = hk.transform(lambda d: LigandMPNN(**config)(d)).init(
            jax.random.PRNGKey(0), dummy_input(num_atoms=atom_context_num))
        assert_params_match(expected, params)
        print("parameter names and shapes match the model")

    directory = os.path.dirname(opt.out_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    joblib.dump(dict(model_state_dict=params,
                     num_edges=num_edges,
                     atom_context_num=atom_context_num,
                     noise_level=checkpoint.get("noise_level")),
                opt.out_path)
    print(f"wrote {opt.out_path}")


if __name__ == "__main__":
    main()
