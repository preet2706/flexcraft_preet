
from typing import Any
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

import warnings
from copy import copy

from flexcraft.files import PDBFile

from flexcraft.data.data import DesignData
from flexcraft.structure.common._data import InputSpec
from flexcraft.structure.af import AFInput, AFResult, make_af2, get_model_haiku_params, model_config, make_predict

_CHAIN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class AF2Spec(InputSpec):
    def to_input(self, **kwargs):
        """AF2 model input. Returns `(AFInput, None)` like every other spec returns
        `(input, writer)`; there is no writer because AFResult.save_pdb writes directly.

        AF2 ignores MSAs, and drops non-protein chains and constraints (with a warning).
        `to_chains` of a template names chains of the COMPLEX (as for ProtenixSpec, see
        protenix/_data.py:_load_template), NOT chains inside the template file as for Boltz:
        the template residues are scattered positionally onto exactly those chains.
        Extra kwargs (pad, cache) are accepted and ignored.
        """
        protein_chains = []
        for chain in self.chains:
            if chain["kind"] != "protein":
                warnings.warn(f"Skipping invalid chain type '{chain['kind']}' for AF2.")
                continue
            protein_chains.append(chain)
        if self.constraints:
            warnings.warn("Skipping provided constraints for AF2.")
        if self.precomputed_msas or any(c.get("use_msa") for c in protein_chains):
            warnings.warn("AF2 ignores MSAs: running single-sequence.")
        parts = []
        for chain in protein_chains:
            for _ in range(chain.get("num_repeats", 1)):
                parts.append(DesignData.from_sequence(chain["sequence"]))
        complex_data = DesignData.concatenate(parts, sep_chains=True)
        chain_index = np.asarray(complex_data["chain_index"])
        af_input = AFInput.from_data(complex_data)
        for template in self.templates:
            if "pdb" not in template:
                raise NotImplementedError("AF2 templates must be '.pdb' files.")
            tdata: DesignData = PDBFile(path=template["pdb"]).to_data()
            to_chains = template.get("template_chains")
            if to_chains:
                selected = [_CHAIN_NAMES.index(c) for c in to_chains]
            else:
                selected = list(range(len(parts)))
            where = np.isin(chain_index, np.asarray(selected))
            assert tdata.size == int(where.sum()), (
                f"AF2 template '{template['pdb']}' has {tdata.size} residues but complex "
                f"chains {to_chains or 'all'} span {int(where.sum())}. to_chains names chains "
                f"of the COMPLEX and the template is written onto them positionally.")
            index = jnp.asarray(np.flatnonzero(where))
            # complex-length template: real coordinates on the selected rows, zero atom_mask
            # elsewhere (AFInput.add_template ANDs atom_mask into `where`).
            full = complex_data.update(
                atom_positions=jnp.zeros_like(complex_data["atom_positions"]).at[index].set(
                    jnp.asarray(tdata["atom_positions"])),
                atom_mask=jnp.zeros_like(complex_data["atom_mask"]).at[index].set(
                    jnp.asarray(tdata["atom_mask"])),
                aa=jnp.asarray(complex_data["aa"]).at[index].set(jnp.asarray(tdata["aa"])))
            af_input = af_input.add_template(full, where=jnp.asarray(where))
        return af_input, None

class AFEvaluator:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def predict(self, key, input: AFInput, num_samples=1):
        """AF2 emits a single structure: `num_samples` is accepted and ignored."""
        return self.model(self.params, key, input)
    
    def __call__(self, key, input: AFInput, num_samples=1):
        return self.predict(key, input, num_samples=num_samples)

@dataclass
class AF2Prediction:
    data: AFResult
    writer: Any = None
    @property
    def result(self):
        return self.data

    def save_pdb(self, path, sample_index=0):
        """Save AF2 prediction as a PDB file."""
        self.data.save_pdb(path)

    def save_cif(self, path, sample_index=0):
        raise NotImplementedError("AF2 results do not save as CIF at the moment.")

    def save(self, path):
        """Save AF2 features as a compressed npz archive for debugging."""
        self.result.save(path)

class AFWrapper:
    def __init__(self, model: str = "af2_model_1_ptm", cache: str = "params/af/",
                 use_remat: bool = True):
        self.model_name = model[4:]    # strip the "af2_" kind prefix
        self.af2_params = jax.tree.map(jnp.asarray, get_model_haiku_params(
                model_name=self.model_name,
                data_dir=cache, fuse=True))
        self.af2_config = model_config(self.model_name)
        self.multimer = "multimer" in self.model_name
        self.af2_config.model.global_config.use_dgram = False
        # as in pipelines/bind/hal/af2.py: needed to differentiate through a binder-target complex
        self.af2_config.model.global_config.use_remat = use_remat

    def evaluator(self, num_recycle=4, **kwargs):
        af2 = jax.jit(make_predict(make_af2(self.af2_config, self.multimer), num_recycle=num_recycle))
        def _wrapper(params):
            return AFEvaluator(af2, params)
        return _wrapper, self.af2_params

    def predictor(self, num_recycle=4, **kwargs):
        evaluator, params = self.evaluator(num_recycle=num_recycle, **kwargs)
        def _wrapper(key, spec: AF2Spec, **kwargs):
            input, writer = spec.to_input()
            return AF2Prediction(data=evaluator(params)(key, input, num_samples=1), writer=writer)
        return _wrapper
