
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
        for chain in self.chains:
            if chain["kind"] != "protein":
                warnings.warn(f"Skipping invalid chain type '{chain['kind']}' for AF2.")
        if self.constraints:
            warnings.warn(f"Skipping provided constraints for AF2.")
        af_input = AFInput.from_chains(*[c for c in self.chains if c["kind"] == "protein"])
        for template in self.templates:
            if template.endswith(".pdb", ".pdb1"):
                tdata: DesignData = PDBFile(template).to_data()
            elif template.endswith(".cif"):
                raise NotImplementedError("Under construction.")
                # tdata: DesignData = CIFFile(template).to_data().aa_only()
            if "template_chains" in template:
                chains = [_CHAIN_NAMES.index(c) for c in template["template_chains"]]
            else:
                chains = np.arange(len(_CHAIN_NAMES))
            pos = tdata.data["atom_positions"]
            aa = af_input.data["aatype"]
            where = jnp.isin(af_input.chain_index, chains)
            af_input = af_input.add_template(pos=pos, aa=aa, where=where)
        return af_input

class AFEvaluator:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def predict(self, key, input: AFInput, num_samples=1):
        return self.model(self.params, key, input)
    
    def __call__(self, key, input: AFInput, num_samples=1):
        return self.predict(key, input, num_samples=num_samples)

@dataclass
class AF2Prediction:
    data: AFResult
    writer: Any
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
    def __init__(self, model: str = "af2_model_1_ptm", cache: str = "params/af/"):
        self.model_name = model[4:]
        self.af2_params = get_model_haiku_params(
                model_name=self.model_name,
                data_dir=cache, fuse=True)
        self.af2_config = model_config(self.model_name)
        self.multimer = "multimer" in self.model_name
        self.af2_config.model.global_config.use_dgram = False

    def evaluator(self, num_recycle=4, **kwargs):
        af2 = jax.jit(make_predict(make_af2(self.af2_config, self.multimer), num_recycle=num_recycle))
        def _wrapper(params):
            return AFEvaluator(af2, params)
        return _wrapper, self.af2_params

    def predictor(self, num_recycle=4, **kwargs):
        evaluator, params = self.evaluator(num_recycle=num_recycle, **kwargs)
        def _wrapper(key, spec: AF2Spec, **kwargs):
            input = spec.to_input()
            return AF2Prediction(evaluator(params)(key, input, num_samples=1))
        return _wrapper
