"""This module provides ligand (non-protein atom) parsing for LigandMPNN.

`flexcraft.utils.io.load_pdb` goes through salad's AlphaFold PDB reader, which converts every
non-standard residue into an UNK residue with an empty atom mask - so HETATM chemistry is lost
before it ever reaches a `DesignData`. This module provides the missing path: a `LigandAtoms`
container holding ligand atom coordinates and element identities, plus parsers for PDB/mmCIF
(via biotite) and SDF/MOL2/SMILES (via rdkit).

Element identities are stored as atomic numbers, matching LigandMPNN's `Y_t`, which indexes a
periodic-table-ordered element list from 1 (so the index equals the atomic number). Hydrogens
(Z=1) and unrecognised elements (Z=0) are masked out, as upstream does via
`Y_m = (Y_t != 1) * (Y_t != 0)`.
"""

import dataclasses
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import jax.numpy as jnp

from rdkit import Chem

from flexcraft.utils.dict_utils import pad_dict

# Element symbol (uppercase) -> atomic number. LigandMPNN's `element_list` is in
# periodic-table order starting at index 1, so its indices are atomic numbers; deriving the
# table from rdkit avoids transcribing a 120-entry list by hand.
ELEMENT_INDEX = {
    Chem.GetPeriodicTable().GetElementSymbol(z).upper(): z for z in range(1, 119)
}

# Number of element classes LigandMPNN one-hot encodes `Y_t` into. Note that the model's
# periodic table (group/period lookup) only has 119 entries, so indices must be clamped to
# `< NUM_PERIODIC_ELEMENTS` before a group/period lookup, but one-hot encoded to NUM_ELEMENTS.
NUM_ELEMENTS = 120
NUM_PERIODIC_ELEMENTS = 119


# Atom counts to pad ligands to. Padding to one of a few fixed sizes keeps `jax.jit` from
# recompiling for every distinct ligand, while keeping the context graph small.
LIGAND_BUCKETS = (32, 64, 128, 256, 512)


def ligand_bucket(num_atoms: int, buckets: Sequence[int] = LIGAND_BUCKETS) -> int:
    """Pick the smallest padding size that fits a given number of ligand atoms.

    Args:
        num_atoms: number of ligand atoms to accommodate.
        buckets: ascending candidate sizes.

    Returns:
        The smallest bucket at least as large as `num_atoms`. If `num_atoms` exceeds every
        bucket, returns `num_atoms` itself.
    """
    for bucket in buckets:
        if num_atoms <= bucket:
            return bucket
    return int(num_atoms)


def element_to_atomic_number(symbols: Iterable[str]) -> np.ndarray:
    """Convert element symbols to atomic numbers.

    Args:
        symbols: iterable of element symbols. Case and surrounding whitespace are ignored.

    Returns:
        Integer array of atomic numbers. Unrecognised symbols map to 0.
    """
    return np.array(
        [ELEMENT_INDEX.get(str(s).strip().upper(), 0) for s in symbols],
        dtype=np.int32)


def _atomic_number_from_atom_name(names: Iterable[str]) -> np.ndarray:
    """Guess atomic numbers from PDB atom names, for files with a blank element column.

    PDB atom names are element-prefixed (e.g. "FE", "C1", "N1"), so stripping digits and
    trying the two-character then one-character prefix recovers most elements.

    Args:
        names: iterable of PDB atom names.

    Returns:
        Integer array of atomic numbers. Unrecognised names map to 0.
    """
    result = []
    for name in names:
        letters = "".join(c for c in str(name).strip().upper() if c.isalpha())
        result.append(
            ELEMENT_INDEX.get(letters[:2], ELEMENT_INDEX.get(letters[:1], 0)))
    return np.array(result, dtype=np.int32)


@dataclasses.dataclass
class LigandAtoms:
    """A flat set of ligand (non-protein) atoms.

    This is the scene-level ligand context LigandMPNN conditions on. It carries no bonds and
    no residue structure: the model only ever sees per-atom coordinates and elements.

    Attributes:
        positions: (num_atoms, 3) float32 atom coordinates in Angstrom.
        elements: (num_atoms,) int32 atomic numbers. 0 means unknown/padding.
        mask: (num_atoms,) bool. True for atoms the model should see, i.e. atoms with a
              known, non-hydrogen element. Matches LigandMPNN's `Y_m`.
        names: optional (num_atoms,) atom names, kept for inspection and PDB output.
        res_names: optional (num_atoms,) residue names, kept for inspection.
    """

    positions: np.ndarray
    elements: np.ndarray
    mask: np.ndarray
    names: Optional[np.ndarray] = None
    res_names: Optional[np.ndarray] = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=np.float32).reshape(-1, 3)
        self.elements = np.asarray(self.elements, dtype=np.int32).reshape(-1)
        self.mask = np.asarray(self.mask, dtype=np.bool_).reshape(-1)
        if not (len(self.positions) == len(self.elements) == len(self.mask)):
            raise ValueError(
                f"inconsistent ligand atom counts: positions={len(self.positions)}, "
                f"elements={len(self.elements)}, mask={len(self.mask)}")

    def __len__(self) -> int:
        return self.positions.shape[0]

    @classmethod
    def empty(cls) -> "LigandAtoms":
        """A single fully-masked dummy atom, for running LigandMPNN without a ligand.

        LigandMPNN has no zero-atom code path, so the apo case is expressed as one masked
        atom. Note that this is *not* equivalent to ProteinMPNN: the ligand branch still
        contributes its `V_C` residual to the node embeddings.
        """
        return cls(positions=np.zeros((1, 3), dtype=np.float32),
                   elements=np.zeros((1,), dtype=np.int32),
                   mask=np.zeros((1,), dtype=np.bool_))

    @classmethod
    def concatenate(cls, items: Sequence["LigandAtoms"]) -> "LigandAtoms":
        """Concatenate several sets of ligand atoms into one.

        Args:
            items: sequence of LigandAtoms to merge.

        Returns:
            LigandAtoms containing all atoms of all inputs, in order.
        """
        items = [item for item in items if len(item) > 0]
        if not items:
            return cls.empty()

        def _cat(attr, dtype):
            values = [getattr(item, attr) for item in items]
            if any(v is None for v in values):
                return None
            return np.concatenate(values, axis=0).astype(dtype)

        return cls(
            positions=np.concatenate([i.positions for i in items], axis=0),
            elements=np.concatenate([i.elements for i in items], axis=0),
            mask=np.concatenate([i.mask for i in items], axis=0),
            names=_cat("names", object),
            res_names=_cat("res_names", object))

    def select(self, index) -> "LigandAtoms":
        """Select a subset of atoms.

        Args:
            index: boolean mask or integer index array over atoms.

        Returns:
            LigandAtoms restricted to the selected atoms.
        """
        return LigandAtoms(
            positions=self.positions[index],
            elements=self.elements[index],
            mask=self.mask[index],
            names=None if self.names is None else self.names[index],
            res_names=None if self.res_names is None else self.res_names[index])

    def drop_hydrogens(self) -> "LigandAtoms":
        """Remove hydrogens and unknown-element atoms outright.

        LigandMPNN masks these out anyway, so dropping them keeps the atom count - and
        therefore the padded context size - small.
        """
        return self.select((self.elements != 1) & (self.elements != 0))

    def pad(self, num_atoms: int) -> "LigandAtoms":
        """Zero-pad to a fixed number of atoms.

        A static atom count keeps `jax.jit` from recompiling for every ligand. Padded atoms
        get element 0 and mask False, so they are inert in the model.

        Args:
            num_atoms: target number of atoms. Must be at least the current count.

        Returns:
            LigandAtoms with exactly `num_atoms` atoms.
        """
        if num_atoms < len(self):
            raise ValueError(
                f"cannot pad {len(self)} ligand atoms down to {num_atoms}")
        padded = pad_dict(dict(positions=self.positions,
                               elements=self.elements,
                               mask=self.mask), num_atoms)
        return LigandAtoms(**padded)

    def centroid(self) -> np.ndarray:
        """Centroid of the unmasked atoms, or the origin if there are none."""
        if not self.mask.any():
            return np.zeros((3,), dtype=np.float32)
        return self.positions[self.mask].mean(axis=0)

    def to_data_fields(self, num_atoms: Optional[int] = None) -> dict:
        """Convert to the `ligand_*` fields of a `DesignData`.

        Args:
            num_atoms: optional static atom count to pad to.

        Returns:
            Dictionary with keys "ligand_positions", "ligand_elements" and "ligand_mask".
        """
        item = self if num_atoms is None else self.pad(num_atoms)
        return dict(
            ligand_positions=jnp.asarray(item.positions, dtype=jnp.float32),
            ligand_elements=jnp.asarray(item.elements, dtype=jnp.int32),
            ligand_mask=jnp.asarray(item.mask, dtype=jnp.bool_))

    def save_pdb(self, path: str, chain_id: str = "X"):
        """Write the ligand atoms as HETATM records, for visual inspection.

        This is a debugging aid, not a round-trippable format: bonds and residue identity
        are not preserved.

        Args:
            path: output PDB path.
            chain_id: chain identifier to write.
        """
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        table = Chem.GetPeriodicTable()
        with open(path, "wt") as f:
            for idx in range(len(self)):
                z = int(self.elements[idx])
                symbol = table.GetElementSymbol(z) if 1 <= z < 119 else "X"
                name = symbol if self.names is None else str(self.names[idx])
                res_name = "LIG" if self.res_names is None else str(self.res_names[idx])
                x, y, zc = self.positions[idx]
                f.write(
                    f"HETATM{idx + 1:5d} {name[:4]:<4s} {res_name[:3]:>3s} {chain_id}"
                    f"{1:4d}    {x:8.3f}{y:8.3f}{zc:8.3f}"
                    f"{1.0:6.2f}{0.0:6.2f}          {symbol:>2s}\n")
            f.write("END\n")


# ============================== biotite: PDB / mmCIF ==============================

def ligand_from_structure(atoms, res_names=None, chain_ids=None, hetero_only=True,
                          keep_waters=False, keep_ions=True, keep_sugars=True,
                          min_occupancy=1e-6, drop_hydrogens=True) -> LigandAtoms:
    """Extract ligand atoms from a biotite `AtomArray`.

    Mirrors LigandMPNN's own selection (`not protein and not water`, zero-occupancy atoms
    dropped), but via biotite rather than ProDy.

    Args:
        atoms: biotite AtomArray for a single model.
        res_names: optional sequence of residue names to keep (e.g. ("BTN",)).
        chain_ids: optional sequence of chain identifiers to keep.
        hetero_only: if True, only consider HETATM records.
        keep_waters: if True, keep solvent atoms. Upstream excludes them.
        keep_ions: if True, keep monoatomic ions (e.g. ZN, MG).
        keep_sugars: if True, keep carbohydrate residues.
        min_occupancy: drop atoms whose occupancy is at or below this value.
        drop_hydrogens: if True, remove hydrogens and unknown elements outright.

    Returns:
        LigandAtoms for the selected atoms. If nothing is selected, returns
        `LigandAtoms.empty()`.
    """
    import biotite.structure as struc

    selection = np.ones(atoms.array_length(), dtype=np.bool_)
    if hetero_only:
        selection &= np.asarray(atoms.hetero, dtype=np.bool_)
    selection &= ~struc.filter_amino_acids(atoms)
    selection &= ~struc.filter_nucleotides(atoms)
    if not keep_waters:
        selection &= ~struc.filter_solvent(atoms)
    if not keep_ions:
        selection &= ~struc.filter_monoatomic_ions(atoms)
    if not keep_sugars:
        selection &= ~struc.filter_carbohydrates(atoms)
    if res_names is not None:
        selection &= np.isin(atoms.res_name, np.asarray(res_names))
    if chain_ids is not None:
        selection &= np.isin(atoms.chain_id, np.asarray(chain_ids))
    if "occupancy" in atoms.get_annotation_categories():
        selection &= np.asarray(atoms.occupancy) > min_occupancy

    if not selection.any():
        return LigandAtoms.empty()

    selected = atoms[selection]
    elements = element_to_atomic_number(selected.element)
    # some PDB files leave the element column blank; fall back to the atom name
    unknown = elements == 0
    if unknown.any():
        elements[unknown] = _atomic_number_from_atom_name(
            np.asarray(selected.atom_name)[unknown])

    result = LigandAtoms(
        positions=np.asarray(selected.coord, dtype=np.float32),
        elements=elements,
        mask=(elements != 1) & (elements != 0),
        names=np.asarray(selected.atom_name, dtype=object),
        res_names=np.asarray(selected.res_name, dtype=object))
    if drop_hydrogens:
        result = result.drop_hydrogens()
    return result if len(result) > 0 else LigandAtoms.empty()


def ligand_from_pdb(path: str, model: int = 1, **kwargs) -> LigandAtoms:
    """Extract ligand atoms from a PDB file.

    Args:
        path: path to a PDB file.
        model: model number to read.
        kwargs: passed to `ligand_from_structure`.

    Returns:
        LigandAtoms for the ligand atoms in the file.
    """
    import biotite.structure.io.pdb as pdb_io

    structure = pdb_io.PDBFile.read(path)
    atoms = structure.get_structure(
        model=model, extra_fields=["occupancy", "b_factor", "charge"], altloc="first")
    return ligand_from_structure(atoms, **kwargs)


def ligand_from_cif(path: str, model: int = 1, **kwargs) -> LigandAtoms:
    """Extract ligand atoms from an mmCIF or BinaryCIF file.

    Args:
        path: path to a .cif/.mmcif/.bcif file.
        model: model number to read.
        kwargs: passed to `ligand_from_structure`.

    Returns:
        LigandAtoms for the ligand atoms in the file.
    """
    import biotite.structure.io.pdbx as pdbx

    reader = pdbx.BinaryCIFFile if path.endswith(".bcif") else pdbx.CIFFile
    atoms = pdbx.get_structure(
        reader.read(path), model=model,
        extra_fields=["occupancy", "b_factor", "charge"], altloc="first")
    return ligand_from_structure(atoms, **kwargs)


def ligand_from_file(path: str, **kwargs) -> LigandAtoms:
    """Extract ligand atoms from a structure or small-molecule file, dispatching on suffix.

    Args:
        path: path to a .pdb, .cif, .mmcif, .bcif, .sdf, .mol or .mol2 file.
        kwargs: passed to the format-specific parser.

    Returns:
        LigandAtoms for the ligand atoms in the file.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".pdb", ".ent"):
        return ligand_from_pdb(path, **kwargs)
    if suffix in (".cif", ".mmcif", ".bcif"):
        return ligand_from_cif(path, **kwargs)
    if suffix in (".sdf", ".mol"):
        return ligand_from_sdf(path, **kwargs)
    if suffix == ".mol2":
        return ligand_from_mol2(path, **kwargs)
    raise ValueError(f"unsupported ligand file type: {path!r}")


# ============================== rdkit: SDF / MOL2 / SMILES ==============================

def ligand_from_rdkit_mol(mol, conformer_id: int = -1,
                          drop_hydrogens: bool = True) -> LigandAtoms:
    """Convert an rdkit molecule with a 3D conformer to LigandAtoms.

    Args:
        mol: rdkit Mol with at least one conformer.
        conformer_id: conformer to read. -1 selects the default conformer.
        drop_hydrogens: if True, remove hydrogens outright.

    Returns:
        LigandAtoms for the molecule's atoms.
    """
    if mol is None:
        raise ValueError("received an empty molecule (rdkit returned None)")
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "molecule has no 3D conformer; LigandMPNN needs atom coordinates. "
            "Embed one with rdkit (see ligand_from_smiles) or supply a posed file.")
    positions = np.asarray(
        mol.GetConformer(conformer_id).GetPositions(), dtype=np.float32)
    elements = np.array(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
    names = np.array(
        [atom.GetSymbol() + str(atom.GetIdx() + 1) for atom in mol.GetAtoms()],
        dtype=object)
    result = LigandAtoms(
        positions=positions, elements=elements,
        mask=(elements != 1) & (elements != 0), names=names)
    if drop_hydrogens:
        result = result.drop_hydrogens()
    return result if len(result) > 0 else LigandAtoms.empty()


def ligand_from_sdf(path: str, index: int = 0, sanitize: bool = True,
                    drop_hydrogens: bool = True, **kwargs) -> LigandAtoms:
    """Extract ligand atoms from one record of an SDF/MOL file.

    Args:
        path: path to a .sdf or .mol file.
        index: which record to read, for multi-record SDF files.
        sanitize: run rdkit sanitization.
        drop_hydrogens: if True, remove hydrogens outright.

    Returns:
        LigandAtoms for the selected record.
    """
    supplier = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=drop_hydrogens)
    mols = list(supplier)
    if index >= len(mols):
        raise ValueError(
            f"{path!r} has {len(mols)} records, cannot read index {index}")
    if mols[index] is None:
        raise ValueError(f"rdkit could not parse record {index} of {path!r}")
    return ligand_from_rdkit_mol(mols[index], drop_hydrogens=drop_hydrogens, **kwargs)


def ligand_from_sdf_all(path: str, sanitize: bool = True,
                        drop_hydrogens: bool = True, **kwargs) -> List[LigandAtoms]:
    """Extract ligand atoms from every record of an SDF file.

    Args:
        path: path to a .sdf file.
        sanitize: run rdkit sanitization.
        drop_hydrogens: if True, remove hydrogens outright.

    Returns:
        List of LigandAtoms, one per parseable record. Unparseable records are skipped.
    """
    supplier = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=drop_hydrogens)
    return [ligand_from_rdkit_mol(mol, drop_hydrogens=drop_hydrogens, **kwargs)
            for mol in supplier if mol is not None]


def ligand_from_mol2(path: str, sanitize: bool = True,
                     drop_hydrogens: bool = True, **kwargs) -> LigandAtoms:
    """Extract ligand atoms from a MOL2 file.

    Args:
        path: path to a .mol2 file.
        sanitize: run rdkit sanitization.
        drop_hydrogens: if True, remove hydrogens outright.

    Returns:
        LigandAtoms for the molecule.
    """
    mol = Chem.MolFromMol2File(path, sanitize=sanitize, removeHs=drop_hydrogens)
    if mol is None:
        raise ValueError(
            f"rdkit could not parse {path!r}. Its MOL2 reader only understands "
            "Corina/Sybyl atom types - convert the file to SDF and use ligand_from_sdf.")
    return ligand_from_rdkit_mol(mol, drop_hydrogens=drop_hydrogens, **kwargs)


def ligand_from_smiles(smiles: str, seed: int = 0, optimize: bool = True,
                       max_attempts: int = 10,
                       drop_hydrogens: bool = True) -> LigandAtoms:
    """Build a 3D conformer from a SMILES string and return its atoms.

    WARNING: the resulting conformer sits in an arbitrary reference frame - it is *not*
    placed in a binding pocket. Feeding it to LigandMPNN alongside a protein backbone is
    only meaningful for composition-only experiments, or after a separate placement/docking
    step. For a hallucination pipeline, take the ligand from the predicted complex instead
    (see `flexcraft.structure.boltz.ligand_from_joltz`).

    Args:
        smiles: SMILES string.
        seed: random seed for conformer embedding.
        optimize: run MMFF (falling back to UFF) force-field optimization.
        max_attempts: embedding attempts before giving up.
        drop_hydrogens: if True, remove hydrogens outright.

    Returns:
        LigandAtoms for the embedded molecule.
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        params.maxIterations = max_attempts
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"rdkit could not embed a conformer for {smiles!r}")
    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except (ValueError, RuntimeError):
            pass  # an unoptimized conformer is still usable
    if drop_hydrogens:
        mol = Chem.RemoveHs(mol)
    return ligand_from_rdkit_mol(mol, drop_hydrogens=drop_hydrogens)


def load_complex(path: str, model: int = 1, protein_chain_ids=None,
                 num_atoms: Optional[int] = None, **ligand_kwargs):
    """Load a protein-ligand complex from a PDB or mmCIF file as a `DesignData`.

    Use this instead of `flexcraft.utils.io.load_pdb` whenever the file contains a ligand.
    `load_pdb` goes through salad's AlphaFold reader, which turns every HETATM residue into
    an UNK residue with an empty atom mask - so ligand chemistry is lost, and the resulting
    phantom `aa=20` rows are then treated as designable positions by `drop_aa` and `sample`.
    This reader keeps only amino acid residues in the protein, and returns the ligand
    separately in the `ligand_*` fields.

    Args:
        path: path to a PDB or mmCIF file.
        model: model number to read.
        protein_chain_ids: optional sequence of chain identifiers to keep in the protein.
        num_atoms: optional static number of ligand atoms to pad to.
        ligand_kwargs: passed to `ligand_from_structure` (e.g. `res_names=("BTN",)`).

    Returns:
        DesignData for the protein with the file's ligand atoms attached.
    """
    import biotite.structure as struc
    from salad.aflib.common import residue_constants as rc

    from flexcraft.data.data import DesignData

    suffix = os.path.splitext(path)[1].lower()
    extra = ["occupancy", "b_factor", "charge"]
    if suffix in (".cif", ".mmcif", ".bcif"):
        import biotite.structure.io.pdbx as pdbx
        reader = pdbx.BinaryCIFFile if suffix == ".bcif" else pdbx.CIFFile
        atoms = pdbx.get_structure(
            reader.read(path), model=model, extra_fields=extra, altloc="first")
    else:
        import biotite.structure.io.pdb as pdb_io
        atoms = pdb_io.PDBFile.read(path).get_structure(
            model=model, extra_fields=extra, altloc="first")

    ligand = ligand_from_structure(atoms, **ligand_kwargs)

    protein = atoms[struc.filter_amino_acids(atoms)]
    if protein_chain_ids is not None:
        protein = protein[np.isin(protein.chain_id, np.asarray(protein_chain_ids))]
    if protein.array_length() == 0:
        raise ValueError(f"no amino acid residues found in {path!r}")

    # one row per residue, atoms placed into their atom14 slots
    starts = struc.get_residue_starts(protein)
    length = len(starts)
    aa = np.full((length,), 20, dtype=np.int32)
    positions = np.zeros((length, 14, 3), dtype=np.float32)
    atom_mask = np.zeros((length, 14), dtype=np.bool_)
    residue_index = np.zeros((length,), dtype=np.int32)
    chain_index = np.zeros((length,), dtype=np.int32)

    chain_ids = {}
    bounds = list(starts) + [protein.array_length()]
    for position, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        res_name = str(protein.res_name[start])
        letter = rc.restype_3to1.get(res_name, "X")
        aa[position] = rc.restype_order.get(letter, rc.restype_num)
        residue_index[position] = int(protein.res_id[start])
        chain = str(protein.chain_id[start])
        chain_index[position] = chain_ids.setdefault(chain, len(chain_ids))
        slots = rc.restype_name_to_atom14_names.get(
            res_name, rc.restype_name_to_atom14_names["UNK"])
        for atom in range(start, stop):
            name = str(protein.atom_name[atom])
            if name in slots:
                slot = slots.index(name)
                positions[position, slot] = protein.coord[atom]
                atom_mask[position, slot] = True

    data = DesignData.from_dict(dict(
        aa=jnp.asarray(aa),
        atom_positions=jnp.asarray(positions),
        atom_mask=jnp.asarray(atom_mask),
        residue_index=jnp.asarray(residue_index),
        chain_index=jnp.asarray(chain_index),
        batch_index=jnp.zeros((length,), dtype=jnp.int32),
        tie_index=jnp.arange(length, dtype=jnp.int32),
        tie_weights=jnp.ones((length,), dtype=jnp.float32),
        # backbone N, CA, C must all be present for the features to be meaningful
        mask=jnp.asarray(atom_mask[:, :3].all(axis=-1))))
    return data.with_ligand(ligand, num_atoms=num_atoms)


def ligand_from_ccd(code: str, mols_dir: str = "params/boltz/mols",
                    drop_hydrogens: bool = True) -> LigandAtoms:
    """Build ligand atoms from a CCD code using Boltz's local component cache.

    `params/boltz/mols/` holds one pickled rdkit molecule per CCD code, so this resolves
    standard cofactors (ATP, HEM, NAD, ...) with their idealised geometry and without
    network access. As with SMILES, the geometry is *not* placed in a pocket.

    Args:
        code: CCD component code, e.g. "ATP".
        mols_dir: directory holding Boltz's per-code pickles.
        drop_hydrogens: if True, remove hydrogens outright.

    Returns:
        LigandAtoms for the component.
    """
    import pickle

    path = os.path.join(mols_dir, f"{code.upper()}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no CCD component {code!r} in {mols_dir!r}")
    with open(path, "rb") as f:
        mol = pickle.load(f)
    return ligand_from_rdkit_mol(mol, drop_hydrogens=drop_hydrogens)
