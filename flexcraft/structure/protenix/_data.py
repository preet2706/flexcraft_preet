import os
from copy import deepcopy
from typing import Dict, Sequence

import gemmi
import numpy as np
import jax
import jax.numpy as jnp

import tempfile
from collections import defaultdict
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform

from flexcraft.structure.protenix.msa.colab_request_parser import RequestParser
from protenix.data.template.template_featurizer import Templates
import protenix.openfold_local.np.residue_constants as rc

from flexcraft.structure.common._data import InputSpec, ModelInput
import flexcraft.sequence.aa_codes as aas
from salad.modules.utils.geometry import positions_to_ncacocb

class ProtenixSpec(InputSpec):
    def __init__(self, *chains, templates=None, constraints=None):
        super().__init__(*chains, templates=templates, constraints=constraints)

    def to_features(self) -> dict:
        return to_features(self.chains, self.templates, self.constraints)

    def to_input(self) -> "ProtenixInput":
        features, atom_array, token_array = self.to_features()
        return ProtenixInput(features=features), ProtenixWriter(atom_array, token_array)

class ProtenixWriter:
    def __init__(self, atom_array, token_array):
        self.atom_array = atom_array
        self.token_array = token_array

    # adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
    def to_gemmi(self, coords, plddt=None):
        if coords is not None:
            atom_array = deepcopy(self.atom_array)
            atom_array.coord = coords
        structure = gemmi.Structure()
        model = gemmi.Model("0")
        chains = {}
        for atom_idx, atom in enumerate(atom_array):
            chain = chains.setdefault(atom.chain_id, {})
            residue = chain.setdefault(int(atom.res_id), _new_gemmi_residue(atom))
            gemmi_atom = _biotite_atom_to_gemmi_atom(atom)
            if plddt is not None:
                gemmi_atom.b_iso = plddt[atom_idx]
            residue.add_atom(gemmi_atom)
        for k in chains:
            chain = gemmi.Chain(k)
            chain.append_residues(list(chains[k].values()))
            model.add_chain(chain)
        structure.add_model(model)
        return structure

    def result_to_data(self, result, sample_index=0):
        is_multisample = len(result.data["samples"].shape) == 4
        plddt = (jax.nn.softmax(result.data["confidence"].plddt_logits) * (jnp.arange(50) + 0.5)).sum(axis=-1) / 50
        coords = result.data["samples"]
        if is_multisample:
            plddt = plddt[sample_index, 0]
            coords = coords[sample_index, 0]
        else:
            plddt = plddt[0]
            coords = coords[0]
        return np.array(coords), np.array(plddt)

    def save_pdb(self, path, sample_atom_coords=None, plddt=None, result=None, ):
        if result is not None:
            sample_atom_coords, plddt = self.result_to_data(result)
        self.to_gemmi(sample_atom_coords, plddt=plddt).write_minimal_pdb(path)

    def save_cif(self, path, sample_atom_coords=None, plddt=None, result=None):
        if result is not None:
            sample_atom_coords, plddt = self.result_to_data(result)
        structure = self.to_gemmi(sample_atom_coords, plddt=plddt)
        document = structure.make_mmcif_document()
        document.write_file(path)

# adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
def _biotite_atom_to_gemmi_atom(atom):
    ga = gemmi.Atom()
    ga.pos = gemmi.Position(*atom.coord)
    ga.element = gemmi.Element(atom.element)
    ga.name = atom.atom_name
    return ga

# adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
def _new_gemmi_residue(atom):
    r = gemmi.Residue()
    r.name = atom.res_name
    r.seqid = gemmi.SeqId(atom.res_id, " ")
    r.entity_type = gemmi.EntityType.Polymer
    return r

# adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
def to_features(chains, templates, constraints):
    # Collect sequences needing MSA search
    seqs_needing_search = []
    for chain in chains:
        if "msa_dir" not in chain:
            chain["msa_dir"] = None
        if "num_repeats" not in chain:
            chain["num_repeats"] = 1
        if chain["use_msa"] and chain["msa_dir"] is None:
            seqs_needing_search.append(chain["sequence"])

    # Run MSA search if needed
    seq_to_msa_dir: Dict[str, str] = {}
    if seqs_needing_search:
        msa_tmpdir = tempfile.mkdtemp(prefix="protenix_msa_")
        unique_seqs = sorted(set(seqs_needing_search))
        msa_subdirs = _run_msa_search(unique_seqs, msa_tmpdir, None)
        seq_to_msa_dir = dict(zip(unique_seqs, msa_subdirs))

    # Build sequences list for sample dict
    sequences = []
    chain_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_to_entity = dict()
    chain_to_copy = dict()
    chain_id = 0
    entity_id = 0
    for i, chain in enumerate(chains):
        chain: Dict[str, str | int]
        if "num_repeats" not in chain:
            chain["num_repeats"] = 1
        for copy_id in range(chain["num_repeats"]):
            chain_to_entity[chain_names[chain_id]] = entity_id + 1
            chain_to_copy[chain_names[chain_id]] = copy_id + 1
            chain_id += 1
        entity_id += 1
        if chain["kind"] == "protein":
            protein_chain = {"sequence": chain["sequence"], "count": chain["num_repeats"]}

            if "msa_dir" in chain and chain["msa_dir"] is not None:
                msa_dir = chain["msa_dir"]
            elif chain["use_msa"]:
                msa_dir = seq_to_msa_dir[chain["sequence"]]
            else:
                # Create dummy MSA
                msa_dir = tempfile.mkdtemp(prefix=f"protenix_msa_dummy_{i}_")
                _make_dummy_msa(chain["sequence"], msa_dir)

            protein_chain["msa"] = {
                "precomputed_msa_dir": msa_dir,
                "pairing_db": "uniref100",
            }
            if "modifications" in chain:
                protein_chain["modifications"] = [
                    {
                        "ptmType": "CCD_" + mod["ccd"],
                        "ptmLocation": mod["residue"]
                    }
                    for mod in chain["modifications"]
                ]
            sequences.append({"proteinChain": protein_chain})
        elif chain["kind"] == "dna":
            dna_chain = {
                "sequence": chain["sequence"],
                "count": chain["num_repeats"],
            }
            if "modifications" in chain:
                dna_chain["modifications"] = [
                    {
                        "modificationType": "CCD_" + mod["ccd"],
                        "basePosition": mod["residue"]
                    }
                    for mod in chain["modifications"]
                ]
            sequences.append({"dnaChain": dna_chain})
        elif chain["kind"] == "rna":
            rna_chain = {
                "sequence": chain["sequence"],
                "count": chain["num_repeats"],
            }
            if "modifications" in chain:
                rna_chain["modifications"] = [
                    {
                        "modificationType": "CCD_" + mod["ccd"],
                        "basePosition": mod["residue"]
                    }
                    for mod in chain["modifications"]
                ]
            if "use_msa" in chain and chain["use_msa"] == True:
                raise NotImplementedError("RNA MSA not supported yet.")
                # pass # TODO
            sequences.append({"rnaChain": rna_chain})
        elif chain["kind"] == "smol":
            ligand = "CCD_ATP"
            if "ccd" in chain:
                ligand = "CCD_" + chain["ccd"].upper()
            elif "smiles" in chain:
                ligand = chain["smiles"]
            else:
                raise ValueError("small molecule chains should have either 'smiles' or 'ccd' set.")
            sequences.append({"ligand": {"ligand": ligand, "count": chain["num_repeats"]}})
        elif chain["kind"] == "ion":
            sequences.append({"ion": {"ion": chain["ccd"], "count": chain["num_repeats"]}})
        else:
            raise NotImplementedError(f"Unknown chain type {chain['kind']}.")

    sample = {"name": "pred", "sequences": sequences}
    if constraints:
        sample["constraint"] = dict()

    # add constraints
    contact_list = []
    bond_list = []
    for constraint in constraints:
        if constraint["kind"] == "contact":
            token_1 = constraint["token_1"]
            chain_1 = token_1["chain"]
            atom_1 = None
            target_1 = token_1["target"]

            token_2 = constraint["token_2"]
            chain_2 = token_2["chain"]
            atom_2 = None
            target_2 = token_2["target"]
            contact = dict(
                entity_1=chain_to_entity[chain_1],
                copy_1=chain_to_copy[chain_1],
                entity_2=chain_to_entity[chain_2],
                copy_2=chain_to_copy[chain_2],
                min_distance=0,
                max_distance=constraint["max_distance"]
            )
            if isinstance(target_1, tuple):
                target_1, atom_1 = target_1
                contact["position_1"] = target_1
                contact["atom_1"] = atom_1
            if isinstance(target_1, str):
                contact["atom_1"] = target_1
            if isinstance(target_2, tuple):
                target_2, atom_2 = target_2
                contact["position_2"] = target_2
                contact["atom_2"] = atom_2
            if isinstance(target_2, str):
                contact["atom_2"] = target_2
    
            contact_list.append(contact)
        if constraint["kind"] == "pocket":
            if "pocket" in sample:
                raise ValueError("ProtenixSpec may contain only a single pocket constraint.")
            pocket = dict()
            binder = constraint["binder"]
            binder_entity = chain_to_entity[binder]
            binder_copy = chain_to_copy[binder]
            pocket["binder_chain"] = [binder_entity, binder_copy]
            pocket["contact_residues"] = [
                (chain_to_entity[chain], chain_to_copy[chain], residue)
                # dict(entity=chain_to_entity[chain],
                #      copy=chain_to_copy[chain],
                #      position=residue)
                for chain, residue in constraint["contacts"]
            ]
            pocket["max_distance"] = constraint["max_distance"]
            sample["constraint"]["pocket"] = pocket
        if constraint["kind"] == "bond":
            bond_list.append(dict(
                entity1=chain_to_entity[constraint["atom_1"]["chain"]],
                copy1=chain_to_copy[constraint["atom_1"]["chain"]],
                position1=constraint["atom_1"]["residue"],
                atom1=constraint["atom_1"]["atom"],
                entity2=chain_to_entity[constraint["atom_2"]["chain"]],
                copy2=chain_to_copy[constraint["atom_2"]["chain"]],
                position2=constraint["atom_2"]["residue"],
                atom2=constraint["atom_2"]["atom"],
            ))
    if contact_list:
        sample["constraint"]["contacts"] = contact_list
    if bond_list:
        sample["constraint"]["covalent_bonds"] = bond_list

    # featurize
    sample2feat = SampleDictToFeatures(sample)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = np.asarray(
        atom_array.distogram_rep_atom_mask, dtype=np.int64
    )

    # featurize MSA
    # Build entity_to_asym_id mapping from atom_array
    entity_to_asym_id: Dict[str, set] = defaultdict(set)
    for entity_id, asym_id_int in zip(
        atom_array.label_entity_id, atom_array.asym_id_int
    ):
        entity_to_asym_id[entity_id].add(asym_id_int)
    entity_to_asym_id = dict(entity_to_asym_id)

    # Load and process MSA features
    msa_feats = InferenceMSAFeaturizer.make_msa_feature(
        bioassembly=sequences,
        entity_to_asym_id=entity_to_asym_id,
        token_array=token_array,
        atom_array=atom_array,
    )
    if msa_feats:
        for k, v in msa_feats.items():
            features_dict[k] = np.asarray(v) if isinstance(v, np.ndarray) else v

    # Apply data type transforms (still uses torch internally)
    features_dict = data_type_transform(features_dict)

    result = {}
    for k, v in features_dict.items():
        if isinstance(v, np.ndarray):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = {k2: np.asarray(v2) if not isinstance(v2, np.ndarray) else v2 for k2, v2 in v.items()}
        else:
            result[k] = v

    # Add atom_rep_atom_idx (needed by JAX model)
    result["atom_rep_atom_idx"] = result["distogram_rep_atom_mask"].nonzero()[0]

    num_res = result["restype"].shape[0]

    # Add templates: TODO / FIXME
    if templates:
        aatype = []
        atom_positions = []
        atom_mask = []
        for template in templates:
            if "template_chains" not in template:
                raise NotImplementedError("Templates require a selector.")
            _aatype, _atom_positions, _atom_mask = _load_template(chains, template)
            aatype.append(_aatype)
            atom_positions.append(_atom_positions)
            atom_mask.append(_atom_mask)
        aatype = np.stack(aatype, axis=0)
        atom_positions = np.stack(atom_positions, axis=0)
        atom_mask = np.stack(atom_mask, axis=0)
        result.update(
            Templates(aatype, atom_positions, atom_mask).as_protenix_dict())
    else:
        aatype = np.full(num_res, rc.unk_restype_index, dtype=np.int64)
        atom_positions = np.zeros((num_res, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((num_res, 37), dtype=np.float32)
        result.update(
            Templates(aatype[None], atom_positions[None], atom_mask[None]).as_protenix_dict())

    return result, atom_array, token_array

def _load_template(chains, template):
    path = template["template"]
    target_chains = template["template_chains"]
    _aatype, _atom_positions, _atom_mask = load_template_from_structure(path)
    target_chain_mask = []
    idx = 0
    full_length = 0
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for chain in chains:
        for i in range(chain["num_repeats"]):
            name = names[idx]
            is_target = name in target_chains
            length = len(chain["sequence"])
            full_length += length
            target_chain_mask += [is_target] * length
            idx += 1
    target_chain_mask = np.array(target_chain_mask, dtype=np.bool_)
    aatype = np.zeros((full_length,), dtype=np.int32)
    atom_positions = np.zeros((full_length, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((full_length, 37,), dtype=np.bool_)
    aatype[target_chain_mask] = _aatype
    atom_positions[target_chain_mask] = _atom_positions
    atom_mask[target_chain_mask] = _atom_mask
    return aatype, atom_positions, atom_mask

def load_template_from_structure(
    path: str
) -> Templates:
    # Parse structure (gemmi auto-detects PDB vs mmCIF)
    structure = gemmi.read_structure(path)
    model = structure[0]  # First model
    _aatype = []
    _atom_positions = []
    _atom_mask = []
    for chain in model:
        # Get polymer residues only (skip ligands, water, etc.)
        residues = [
            res for res in chain
            if gemmi.find_tabulated_residue(res.name).is_amino_acid()
        ]
        num_residues = len(residues)

        # Build atom37 positions and masks
        atom_positions = np.zeros((num_residues, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((num_residues, 37), dtype=np.float32)
        res_names = []

        for i, res in enumerate(residues):
            res_names.append(res.name)
            for atom in res:
                aname = atom.name
                if aname in rc.atom_order:
                    aidx = rc.atom_order[aname]
                    atom_positions[i, aidx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    atom_mask[i, aidx] = 1.0

        # Extract from template structure using gemmi's 3-to-1 conversion
        aatype = np.array(
            [
                rc.restype_order.get(
                    gemmi.find_tabulated_residue(name).one_letter_code,
                    rc.unk_restype_index,
                )
                for name in res_names
            ],
            dtype=np.int64,
        )
        _aatype.append(aatype)
        _atom_positions.append(atom_positions)
        _atom_mask.append(atom_mask)
    _aatype = np.concatenate(_aatype, axis=0)
    _atom_positions = np.concatenate(_atom_positions, axis=0)
    _atom_mask = np.concatenate(_atom_mask, axis=0)

    return _aatype, _atom_positions, _atom_mask

# adapted from escalante.bio/protenij, under Apache 2.0 license
def _make_dummy_msa(sequence: str, msa_dir: str) -> None:
    """Create dummy MSA files (just the query sequence)."""
    import os
    os.makedirs(msa_dir, exist_ok=True)
    for fname in ["pairing.a3m", "non_pairing.a3m"]:
        with open(os.path.join(msa_dir, fname), "w") as f:
            f.write(f">query\n{sequence}\n")

# adapted from escalante.bio/protenij, under Apache 2.0 license
def _run_msa_search(sequences: Sequence[str], msa_dir: str, email: str = "") -> list:
    """Run MMSeqs2 search for sequences, return list of result directories."""

    os.makedirs(msa_dir, exist_ok=True)
    seqs_sorted = sorted(set(sequences))
    tmp_fasta = os.path.join(msa_dir, "msa_input.fasta")

    return RequestParser.msa_search(
        seqs_pending_msa=seqs_sorted,
        tmp_fasta_fpath=tmp_fasta,
        msa_res_dir=msa_dir,
        mode="colabfold",
        email=email,
    )
    # return RequestParser.msa_postprocess(
    #     seqs_pending_msa=seqs_sorted,
    #     msa_res_dir=msa_dir,
    # )

class ProtenixInput(ModelInput):
    features: dict
    def set_aa(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=slice(0, 20), seq_count=20)

    def set_rna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=slice(21, 25), seq_count=4)
    
    def set_dna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=slice(26, 30), seq_count=4)

    def set_template(self, coords, start=None,
                     mask=None, restype=None, template_id=0):
        if restype is None:
            restype = self.features["restype"]
        num_template = coords.shape[0]
        num_protein = self.features["restype"].shape[0]
        if mask is None:
            mask = jnp.ones((num_protein,), dtype=jnp.bool_)
        if num_protein > num_template:
            _mask = jnp.zeros((num_protein,), dtype=jnp.bool_)
            _mask = _mask.at[start:start+num_template].set(mask)
            _coords = jnp.zeros((num_protein, 14, 3), dtype=jnp.float32)
            _coords = _coords.at[start:start+num_template].set(coords)
            mask = _mask
            coords = _coords
        ncacocb = positions_to_ncacocb(coords)
        pb_mask = mask
        bb_mask = mask
        pseudo_beta = jnp.where((restype == aas.AF2_CODE.index("G"))[:, None], ncacocb[:, 1], ncacocb[:, 4])
        dist = jnp.linalg.norm(pseudo_beta[:, None] - pseudo_beta[None, :], axis=-1)
        min_bin: float = 3.25
        max_bin: float = 50.75
        bin_edges = jnp.linspace(min_bin, max_bin, 39)
        bin_edges = jnp.concatenate((bin_edges, jnp.array([1e8])), axis=-1)
        dgram = (dist > bin_edges[:-1]) * (dist < bin_edges[1:]) > 0
        unit_vectors = compute_unit_vector(coords, mask)
        template_features = {
            "template_aatype": restype,
            "template_distogram": dgram,
            "template_pseudo_beta_mask": pb_mask,
            "template_unit_vector": unit_vectors,
            "template_backbone_frame_mask": bb_mask,
        }
        result = self.copy()
        result.features.update(template_features)
        return result

# adapted from escalante-bio/protenij under Apache 2.0 license
def compute_unit_vector(
    atom_positions: jnp.ndarray,
    mask: jnp.ndarray,
    epsilon: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    n_pos = atom_positions[:, 0]
    ca_pos = atom_positions[:, 1]
    c_pos = atom_positions[:, 2]

    # Build local frame: origin at CA, x-axis along C-CA
    v1 = c_pos - ca_pos  # C-CA direction
    v2 = n_pos - ca_pos  # N-CA direction

    # Gram-Schmidt orthogonalization
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + epsilon)
    e2 = v2 - jnp.sum(v2 * e1, axis=-1, keepdims=True) * e1
    e2 = e2 / (jnp.linalg.norm(e2, axis=-1, keepdims=True) + epsilon)
    e3 = jnp.cross(e1, e2)

    # Relative positions: diff[i, j] = CA[j] - CA[i]
    diff = ca_pos[None, :, :] - ca_pos[:, None, :]

    # Transform to local frame: project onto basis vectors
    ux = jnp.sum(e1[:, None, :] * diff, axis=-1)
    uy = jnp.sum(e2[:, None, :] * diff, axis=-1)
    uz = jnp.sum(e3[:, None, :] * diff, axis=-1)

    unit_vector = jnp.stack([ux, uy, uz], axis=-1)

    # Normalize to unit vector
    uv_norm = jnp.linalg.norm(unit_vector, axis=-1, keepdims=True)
    unit_vector = unit_vector / (uv_norm + epsilon)

    # 2D mask: valid only if both residues have backbone
    mask_2d = mask[:, None] * mask[None, :]

    return unit_vector, mask_2d