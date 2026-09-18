from typing import Iterable

import numpy as np

import jax
import jax.numpy as jnp

import equinox as eqx

from flexcraft.data.data import DesignData
from flexcraft.files import PDBFile

class InputSpec:
    def __init__(self, *chains, templates=None, constraints=None):
        self.chains = list(chains)
        self.templates = templates or list()
        self.constraints = constraints or list()
        self.temporaries = list()
        self.precomputed_msas = list()

    def add_chain(self, *chains):
        _chains = []
        for c in chains:
            if isinstance(c, DesignData):
                subchains = c.split(c["chain_index"])
                for s in subchains:
                    _chains.append(dict(
                        sequence=s.to_sequence_string(),
                        kind="protein",
                        use_msa=False
                    ))
            else:
                _chains.append(c)
        self.chains += _chains
        return self

    def add_polymer(self, *sequences, kind="protein", use_msa=False, num_repeats=1):
        _chains = []
        for c in sequences:
            subchains = c.split(":")
            for s in subchains:
                _chains.append(dict(
                    sequence=s,
                    kind=kind,
                    use_msa=use_msa,
                    num_repeats=num_repeats
                ))
        return self.add_chain(*_chains)

    def add_protein(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(*sequences, kind="protein", use_msa=use_msa, num_repeats=num_repeats)

    def add_dna(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(*sequences, kind="dna", use_msa=use_msa, num_repeats=num_repeats)

    def add_rna(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(*sequences, kind="rna", use_msa=use_msa, num_repeats=num_repeats)

    def add_smiles(self, smiles, num_repeats=1):
        _chains = [
            dict(kind="ligand", smiles=smiles, num_repeats=num_repeats)
        ]
        return self.add_chain(*_chains)

    def add_ccd(self, ccd, num_repeats=1):
        _chains = [
            dict(kind="ligand", ccd=ccd, num_repeats=num_repeats)
        ]
        return self.add_chain(*_chains)

    def add_template(self, path_or_object, to_chains=None):
        template_info = dict(template=path_or_object, template_chains=to_chains)
        if isinstance(path_or_object, str):
            if path_or_object.endswith((".pdb", ".pdb1")):
                template_info["pdb"] = path_or_object
            elif path_or_object.endswith(".cif"):
                template_info["cif"] = path_or_object
            else:
                raise NotImplementedError(f"Invalid template file '{path_or_object}'. "
                                          f"Has to be either '.pdb' or '.cif'.")
        elif isinstance(path_or_object, DesignData):
            tmpfile = PDBFile(data = path_or_object, temporary = True)
            self.temporaries.append(tmpfile)
            template_info["pdb"] = tmpfile.path
        else:
            raise NotImplementedError(
                "Input template has to be a path to a '.pdb' or '.cif' file.")
        self.templates.append(template_info)
        return self

    def add_msa(self, to_chains=None):
        if to_chains is None:
            to_chains = list(range(len(self.chains)))
        for c in to_chains:
            chain = self.chains[c]
            if not all(x == "X" for x in chain["sequence"]):
                self.chains[c]["use_msa"] = True
        return self

    def precomputed_msa(self, path, start_chain=0):
        offset = 0
        for chain in self.chains[:start_chain]:
            offset += len(chain["sequence"]) * chain.get("num_repeats", 1)
        self.precomputed_msas.append(dict(path=path, offset=offset))
        for chain in self.chains:
            chain["use_msa"] = False
        return self

    def add_bond(self, chain_1, residue_1, atom_1,
                 chain_2, residue_2, atom_2):
        self.constraints.append(dict(
            kind="bond",
            atom_1=dict(chain=chain_1, residue=residue_1, atom=atom_1),
            atom_2=dict(chain=chain_2, residue=residue_2, atom=atom_2)))
        return self

    def add_contact(self, chain_1, residue_or_atom_1,
                    chain_2, residue_or_atom_2, max_distance=6.0):
        self.constraints.append(dict(
            kind="contact",
            token_1=dict(chain=chain_1, target=residue_or_atom_1),
            token_2=dict(chain=chain_2, target=residue_or_atom_2),
            max_distance=max_distance
        ))
        return self

    def add_pocket(self, binder, *contacts, max_distance=6.0):
        self.constraints.append(dict(
            kind="pocket",
            binder=binder,
            contacts=contacts,
            max_distance=max_distance
        ))
        return self

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return self

    def to_features(self, pad=True, cache="./params/model/") -> dict:
        raise NotImplementedError("Implement to_features per model.")

    def to_input(self, pad=True, cache="./params/model/"):
        raise NotImplementedError("Implement to_input per model.")

    def save_msa(self, path: str, cache="./params/model/"):
        raise NotImplementedError("Implement save_msa per model.")

_AA_SLICE = slice(2, 22)
_AA_UNK = 22
_RNA_SLICE = slice(23, 27)
_RNA_UNK = 27
_DNA_SLICE = slice(28, 32)
_DNA_UNK = 32

class ModelInput(eqx.Module):
    features: dict
    @property
    def residue_index(self):
        return self.features["residue_index"][0]
    @property
    def chain_index(self):
        return self.features["asym_id"][0]
    @property
    def residue_type(self):
        return jnp.argmax(self.features["res_type"][0], axis=-1)

    def copy(self):
        result = type(self)(features={k: v for k, v in self.features.items()})
        return result

    def _set_res_type(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        result = self.copy()
        result.features["res_type"] = jnp.array(result.features["res_type"]).astype(jnp.float32)
        result.features["res_type"] = result.features["res_type"].at[0, start:start + sequence.shape[0]].set(0.0)
        result.features["res_type"] = result.features["res_type"].at[0, start:start + sequence.shape[0], seq_slice].set(sequence[:, :seq_count])
        return result

    def _set_profile(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        result = self.copy()
        num_msa = result.features["msa"].shape[1]
        result.features["profile"] = jnp.array(result.features["profile"]).astype(jnp.float32)
        result.features["profile"] = result.features["profile"].at[0, start:start + sequence.shape[0]].set(0.0)
        result.features["profile"] = result.features["profile"].at[0, start:start + sequence.shape[0], seq_slice].set(sequence[:, :seq_count] / num_msa)
        result.features["profile"] = result.features["profile"].at[0, start:start + sequence.shape[0], 1].set((num_msa - 1) / num_msa)
        return result

    def _set_msa(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        result = self.copy()
        result.features["msa"] = jnp.array(result.features["msa"]).astype(jnp.float32)
        # reset MSA for all positions we're setting:
        # setting all msa positions to zero
        result.features["msa"] = result.features["msa"].at[0, :, start:start + sequence.shape[0]].set(0.0)
        # setting all msa positions from the 2nd sequence onwards to "-"
        result.features["msa"] = result.features["msa"].at[0, 1:, start:start + sequence.shape[0], 1].set(1.0)
        # finally, setting the first sequence to the input sequence
        result.features["msa"] = result.features["msa"].at[0, 0, start:start + sequence.shape[0], seq_slice].set(sequence[:, :seq_count])
        return result

    def _set_sequence(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20,
                      reset_msa=True, reset_profile=True):
        result = self.copy()
        result = result._set_res_type(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        if reset_profile:
            result = result._set_profile(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        if reset_msa:
            result = result._set_msa(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        return result

    def set_aa(self, sequence, start=0, reset_msa=True, reset_profile=True):
        return self._set_sequence(sequence, start=start,
                                  reset_msa=reset_msa,
                                  reset_profile=reset_profile)

    def inherit_features(self, data: "ModelInput", features: Iterable[str] = None):
        for name in features:
            self.features[name] = data.features[name]
        return self

    def save_features(self, path: str, features: Iterable[str]) -> "ModelInput":
        features_dict = {
            name: self.features[name]
            for name in features
        }
        np.savez_compressed(path, **features_dict)
        return self

    def load_msa(self, path: str, start=0, msa_start=0, msa_end=None) -> "ModelInput":
        msa_features = np.load(path)
        result = self.copy()
        msa_names = ["msa", "msa_mask", "msa_paired", "has_deletion", "deletion_value"]
        # sequence_names = ["profile", "deletion_mean"]
        msa = msa_features["msa"]
        msa_length = msa[:, :, msa_start:msa_end].shape[2]
        sequence_count = msa.shape[1]
        current_sequence_count = self.features["msa"].shape[1]
        # pad current features to number of msa sequences
        for name in msa_names:
            if sequence_count > current_sequence_count:
                difference = sequence_count - current_sequence_count
                padding = jnp.zeros(
                    [1, difference] + list(result.features[name].shape[2:]),
                    dtype=result.features[name].dtype)
                # make padded msa all gaps
                if name == "msa":
                    padding = padding.at[..., 1].set(1)
                # adjust msa mask
                if name == "msa_mask":
                    padding = padding.at[0].set(msa_features["msa_mask"][0, -difference:, 0:1])
                result.features[name] = jnp.concatenate(
                    (result.features[name], padding), axis=1)
            result.features[name] = result.features[name].at[0, :, start:start + msa_length].set(msa_features[name][0, :, msa_start:msa_end])
        result.features["profile"] = result.features["msa"].mean(axis=1)
        result.features["deletion_mean"] = result.features["deletion_value"].mean(axis=1)
        return result

    def set_msa(self, msa, start=0):
        result = self.copy()
        result.features["msa"] = jnp.array(result.features["msa"]).astype(jnp.float32)
        # reset MSA for all positions we're setting:
        # setting all msa positions to zero
        result.features["msa"] = result.features["msa"].at[0, :, start:start + msa.shape[1]].set(0.0)
        # setting all msa positions from the 2nd sequence onwards to "-"
        result.features["msa"] = result.features["msa"].at[0, 1:, start:start + msa.shape[1], 1].set(1.0)
        # finally, setting the first sequence to the input sequence
        result.features["msa"] = result.features["msa"].at[0, 0:msa.shape[0], start:start + msa.shape[1], :].set(msa)
        # compute & set the profile
        result.features["profile"] = result.features["msa"].at[0, start:start + msa.shape[1]].set(msa.mean(axis=0))
        return result

    def set_aa_msa(self, msa, start=0):
        if msa.shape[-1] == 20:
            msa_size = self.features["msa"].shape[-1]
            seq_start = 2
            msa = jnp.pad(msa, ((0, 0), (0, 0), (seq_start, msa_size - 20)))
        aa = msa[0]
        aa = aa[..., 2:22]
        return self.set_aa(aa, start=start).set_msa(msa, start=start)

    def set_aa_msa_random(self, key, msa, start=0):
        index = jax.random.permutation(key, msa.shape[0])
        msa = msa[index]
        return self.set_aa_msa(msa, start=start)

    def set_rna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=_RNA_SLICE, seq_count=4)
    
    def set_dna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=_DNA_SLICE, seq_count=4)

