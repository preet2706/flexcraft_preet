"""Explorative, per-ligand analysis of a ligand_bias hallucination run.

For every target (small molecule or DNA) this writes three small figures, each answering one
question, into <out_path>/<NN>_<target>/ :

    1_metrics.png          How good are the replicates?  Every metric as a box plus one dot per
                           replicate (a KDE would draw a smooth curve through only ~10 points).
    2_interface_aa.png     Which amino-acid types are enriched at the interface?  Interface
                           residues (any heavy atom <= --contact_cutoff from the target) are
                           compared with the rest of the *same* binder, replicate by replicate.
    3_contacts_burial.png  Which target atoms touch the binder, and is the target sitting on the
                           surface or buried in a pocket?

Axis limits are fixed across targets, so flipping between folders compares them directly.
Cross-target numbers go into ligand_summary.csv rather than into one crowded plot.

Also written to <out_path>:
    per_attempt.csv     burial, contact count and atoms-in-contact for every replicate
    interface_aa.csv    interface vs. rest composition, replicate x amino-acid class
    ligand_atoms.csv    every target atom of every replicate with its distance to the binder
    ligand_summary.csv  one row per target

The PDB measurements are cached; pass --recompute to redo them.

Usage:
    python3 scripts/analyze_ligand_hallucination.py --run_path runs/ligand_hallucination_ligmpnn

Needs only matplotlib, pandas, scipy, Biopython and RDKit (the `flexcraft` environment).
"""

import io
import os
import copy
import json
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.Data.PDBData import protein_letters_3to1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDepictor, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

from plot_ligand_bias import (  # same directory
    _load_results_csv, _condition_order, _short_label,
    CATEGORY_COLOR, CATEGORY_LABEL, BLUE, ORANGE, GREY, INK)
from analyze_ligand_burial import RADII, classify, CLASS_COLOR, CLASS_LABEL

# Same classes as flexcraft.pipelines.experimental.ligand_bias.AA_CLASSES (that module runs
# option parsing at import time, so it cannot be imported here).
AA_CLASSES = dict(hydrophobic="AVLIM", aromatic="FWY", positive="KRH",
                  negative="DE", polar="STNQC", special="GP")
PASS_GREEN = "#55A868"
THRESHOLD_GREEN = "#1B8A4B"             # darker than the palette green, for the overview lines
CONTACT_CMAP = LinearSegmentedColormap.from_list("shade", ["#EDEDED", "#2F4F85"])
REPLICATES_FOR_CONSISTENT = 0.8       # a class counts as consistent if >= 80% of replicates agree


# ============================== measuring the predicted complexes ==============================

def measure_complex(path: str, blueprint_path: str = None) -> dict:
    """Everything this analysis needs from one predicted complex (chain A = binder, every other
    chain = target), from a single parse of the PDB.

    Returns a dict with
        letters      binder one-letter sequence (list)
        res_dist     (L,) distance of each binder residue to its nearest target heavy atom
        atoms        list of per-target-atom dicts (chain, resseq, resname, atom, element,
                     min_dist to the binder, sasa_alone, sasa_complex)
        binder_plddt (L,) pLDDT of every binder residue
        target_plddt pLDDT of every target token
        blueprint    only if `blueprint_path` is given: blueprint_ca_rmsd, dssp_identity and the
                     final binder's dssp string (see compare_to_blueprint)

    Boltz writes pLDDT * 100 into the B-factor column. A small molecule is tokenized one atom per
    token and a nucleotide one residue per token, so that is how the target's pLDDT is averaged.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PDBParser(QUIET=True).get_structure("x", path)[0]
    binder = [r for r in model["A"] if r.id[0] == " "]
    letters = [protein_letters_3to1.get(r.get_resname().upper(), "X") for r in binder]
    target_atoms = [a for c in model if c.id != "A" for a in c.get_atoms()]

    binder_atoms = [(i, a.coord) for i, r in enumerate(binder) for a in r.get_atoms()]
    owner = np.array([i for i, _ in binder_atoms])
    binder_xyz = np.array([c for _, c in binder_atoms])
    target_xyz = np.array([a.coord for a in target_atoms])

    atom_dist, _ = cKDTree(binder_xyz).query(target_xyz)            # target atom -> binder
    to_target, _ = cKDTree(target_xyz).query(binder_xyz)             # binder atom -> target
    res_dist = np.full(len(binder), np.inf)
    np.minimum.at(res_dist, owner, to_target)

    sr = ShrakeRupley(probe_radius=1.4, n_points=100, radii_dict=RADII)
    sr.compute(model, level="A")
    sasa_complex = np.array([a.sasa for a in target_atoms])
    alone = copy.deepcopy(model)
    alone.detach_child("A")
    sr.compute(alone, level="A")
    sasa_alone = np.array([a.sasa for c in alone for a in c.get_atoms()])

    atoms = []
    for k, a in enumerate(target_atoms):
        res = a.get_parent()
        atoms.append(dict(
            chain=res.get_parent().id, resseq=int(res.id[1]), resname=res.get_resname().strip(),
            atom=a.get_name().strip(), element=(a.element or "").strip().upper(),
            x=float(a.coord[0]), y=float(a.coord[1]), z=float(a.coord[2]),
            min_dist=float(atom_dist[k]), sasa_alone=float(sasa_alone[k]),
            sasa_complex=float(sasa_complex[k])))
    binder_plddt = np.array([np.mean([a.get_bfactor() for a in r]) for r in binder]) / 100.0
    target_plddt = []
    for chain in model:
        if chain.id == "A":
            continue
        for res in chain:
            values = [a.get_bfactor() / 100.0 for a in res]
            target_plddt += values if res.get_resname().strip() == "LIG" else [float(np.mean(values))]
    result = dict(letters=letters, res_dist=res_dist, atoms=atoms,
                  binder_plddt=binder_plddt, target_plddt=np.array(target_plddt))
    if blueprint_path is not None:
        backbone, present = backbone_of(binder)
        result["blueprint"] = compare_to_blueprint(blueprint_path, binder, backbone, present)
    return result


def measure_run(run_path: str, df: pd.DataFrame, cutoff: float, surface_max: float,
                pocket_min: float):
    """Measure every non-apo replicate. Returns (per_attempt, interface_aa, ligand_atoms)."""
    per_attempt, interface_rows, atom_rows = [], [], []
    todo = df[df["category"] != "apo"]
    for n, (_, r) in enumerate(todo.iterrows(), start=1):
        row = int(r["row_index"])
        stem = f"{row:04d}_{r['condition_name']}_a{int(r['attempt_index'])}"
        blueprint_path = f"{run_path}/blueprints/{stem}.pdb"
        m = measure_complex(f"{run_path}/predictions/{stem}.pdb",
                            blueprint_path if os.path.exists(blueprint_path) else None)
        if n % 25 == 0 or n == len(todo):
            print(f"  measured {n}/{len(todo)} predictions")

        interface = m["res_dist"] <= cutoff      # cutoff only applies here
        letters = np.array(m["letters"])
        for cls, members in AA_CLASSES.items():
            in_class = np.isin(letters, list(members))
            n_if, n_rest = int(interface.sum()), int((~interface).sum())
            interface_rows.append(dict(
                row_index=row, aa_class=cls, n_interface=n_if, n_rest=n_rest,
                interface_frac=in_class[interface].mean() if n_if else np.nan,
                rest_frac=in_class[~interface].mean() if n_rest else np.nan))

        a = pd.DataFrame(m["atoms"])
        a.insert(0, "row_index", row)
        atom_rows.append(a)
        buried = float(1.0 - a["sasa_complex"].sum() / a["sasa_alone"].sum())
        per_attempt.append(dict(
            row_index=row, n_interface=int(interface.sum()), target_atoms=len(a),
            buried_fraction=buried, atoms_in_contact=float((a["min_dist"] <= cutoff).mean()),
            burial_class=classify(buried, surface_max, pocket_min),
            binder_plddt_pdb=float(m["binder_plddt"].mean()),
            ligand_plddt=float(m["target_plddt"].mean()),
            plddt_all_tokens=float(np.concatenate([m["binder_plddt"], m["target_plddt"]]).mean())))
        if "blueprint" in m:
            b = m["blueprint"]
            per_attempt[-1].update(blueprint_ca_rmsd=b["blueprint_ca_rmsd"],
                                   dssp_identity=b["dssp_identity"])
            # the same DSSP function on the final structure should give the pipeline's own string
            csv_dssp = str(r.get("dssp_hard", ""))
            per_attempt[-1]["dssp_matches_results"] = (
                float(np.mean([x == y for x, y in zip(b["dssp_final"], csv_dssp)]))
                if len(csv_dssp) == len(b["dssp_final"]) else np.nan)

    interface_aa = pd.DataFrame(interface_rows)
    interface_aa["enrichment_pp"] = 100.0 * (interface_aa["interface_frac"]
                                             - interface_aa["rest_frac"])
    return pd.DataFrame(per_attempt), interface_aa, pd.concat(atom_rows, ignore_index=True)


# ============================== blueprint comparison ==============================

DSSP_CODE = "LHE"                        # salad's assign_dssp: loop = 0, helix = 1, strand = 2
_ASSIGN_DSSP = None


def backbone_of(residues):
    """((L, 4, 3) N/CA/C/O coordinates, (L,) all-four-present mask) of a list of residues."""
    xyz = np.zeros((len(residues), 4, 3), dtype=np.float32)
    present = np.zeros((len(residues),), dtype=np.bool_)
    for i, res in enumerate(residues):
        if all(name in res for name in ("N", "CA", "C", "O")):
            xyz[i] = [res[name].coord for name in ("N", "CA", "C", "O")]
            present[i] = True
    return xyz, present


def dssp_of(backbone, present) -> np.ndarray:
    """Secondary structure (0 loop, 1 helix, 2 strand) with the function the pipeline itself uses
    (DesignData.assign_dssp), so that its dssp_hard strings and this are directly comparable."""
    global _ASSIGN_DSSP
    if _ASSIGN_DSSP is None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        from salad.modules.utils.dssp import assign_dssp
        _ASSIGN_DSSP = jax.jit(assign_dssp)
    import jax.numpy as jnp
    dssp, *_ = _ASSIGN_DSSP(jnp.asarray(backbone), jnp.zeros((len(backbone),), dtype=jnp.int32),
                            jnp.asarray(present))
    return np.asarray(dssp)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of two (N, 3) point sets after optimal rigid superposition."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, sing, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))              # exclude reflections
    msd = ((a ** 2).sum() + (b ** 2).sum() - 2.0 * (sing[0] + sing[1] + sign * sing[2])) / len(a)
    return float(np.sqrt(max(msd, 0.0)))


def compare_to_blueprint(blueprint_path: str, binder, final_backbone, final_present) -> dict:
    """CA RMSD and DSSP identity of the final binder against its salad blueprint.

    Residues are matched by order (the blueprint and the final binder both have `num_aa`
    residues). DSSP identity is the fraction of residues with the same loop/helix/strand state."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blueprint = [r for r in PDBParser(QUIET=True).get_structure("bp", blueprint_path)[0]["A"]
                     if r.id[0] == " "]
    if len(blueprint) != len(binder):
        raise ValueError(f"{blueprint_path}: blueprint has {len(blueprint)} residues, the final "
                         f"binder {len(binder)}")
    bp_backbone, bp_present = backbone_of(blueprint)
    both = bp_present & final_present
    ca_final, ca_blueprint = final_backbone[both, 1], bp_backbone[both, 1]
    dssp_final = dssp_of(final_backbone, final_present)
    dssp_blueprint = dssp_of(bp_backbone, bp_present)
    return dict(blueprint_ca_rmsd=kabsch_rmsd(ca_final, ca_blueprint),
                dssp_identity=float((dssp_final[both] == dssp_blueprint[both]).mean()),
                dssp_final="".join(DSSP_CODE[i] for i in dssp_final))


# ============================== ligand structures for drawing ==============================

def ligand_molecule(entry: dict):
    """(heavy-atom RDKit molecule with 2D coordinates, {PDB atom name: atom index}), or None.

    Boltz names the atoms of a SMILES ligand `<ELEMENT><canonical rank + 1>`, with the rank
    taken on the molecule *including hydrogens*, and takes CCD atom names from the component
    definition. Both are reproduced here so that every predicted atom can be placed on the 2D
    structure; `check_mapping` verifies the result against the predicted coordinates."""
    kind = entry["target_kind"]
    if kind == "smiles":
        with_h = AllChem.AddHs(Chem.MolFromSmiles(entry["smiles"]))
        ranks = list(AllChem.CanonicalRankAtoms(with_h))
        mol = Chem.RemoveHs(with_h)               # heavy atoms keep their indices
        names = {f"{a.GetSymbol().upper()}{ranks[a.GetIdx()] + 1}": a.GetIdx()
                 for a in mol.GetAtoms()}
    elif kind == "ccd":
        with open(f"params/boltz/mols/{entry['ccd']}.pkl", "rb") as f:
            mol = Chem.RemoveHs(pickle.load(f), sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        Chem.SanitizeMol(mol, catchErrors=True)       # metal centres (HEM) may not sanitize
        names = {a.GetProp("name"): a.GetIdx() for a in mol.GetAtoms() if a.HasProp("name")}
    else:
        return None
    if mol.GetNumAtoms() < 2:                     # a single ion has no structure to draw
        return None
    _add_2d_coords(mol)
    return mol, names


METALS = {"FE", "ZN", "MG", "MN", "CU", "CO", "NI", "MO", "CA", "NA", "K"}


def _add_2d_coords(mol):
    """Give the molecule 2D coordinates for drawing.

    RDKit's default layout is fine for organic molecules but tangles a metal complex such as
    heme, whose four Fe-N coordination bonds it cannot place. CoordGen is built for exactly
    that, so it is used whenever a metal is bonded to the molecule. Only the drawing is affected;
    the atom set and names are untouched."""
    has_metal = any(a.GetSymbol().upper() in METALS and a.GetDegree() >= 2
                    for a in mol.GetAtoms())
    if has_metal:
        mol.RemoveAllConformers()                 # the CCD molecule carries a 3D conformer
        rdCoordGen.AddCoords(mol)
    else:
        rdDepictor.Compute2DCoords(mol)


def check_mapping(mol, names: dict, atoms: pd.DataFrame, max_bond: float = 2.3) -> bool:
    """True if the PDB atom names match the structure AND every bond of the structure is a
    plausible length in the prediction.

    The second condition is what makes the name mapping trustworthy: a wrongly permuted mapping
    pairs atoms that are not bonded, which shows up as bonds of many Angstrom. It is checked on
    every replicate."""
    if set(atoms["atom"]) != set(names):
        return False
    by_index = {i: n for n, i in names.items()}
    bonds = [(by_index[b.GetBeginAtomIdx()], by_index[b.GetEndAtomIdx()])
             for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in by_index and b.GetEndAtomIdx() in by_index]
    for _, one in atoms.groupby("row_index"):
        xyz = {r.atom: np.array([r.x, r.y, r.z]) for r in one.itertuples()}
        if any(np.linalg.norm(xyz[a] - xyz[b]) > max_bond for a, b in bonds):
            return False
    return True


def draw_shaded_molecule(ax, mol, names: dict, value: dict):
    """Draw the 2D structure with each atom shaded by `value[atom name]` (0-1).
    Atoms without a value (NaN or missing) are drawn in mid-grey."""
    colors, radii = {}, {}
    for name, idx in names.items():
        v = value.get(name, np.nan)
        colors[idx] = (0.78, 0.78, 0.78) if not np.isfinite(v) else CONTACT_CMAP(v)[:3]
        radii[idx] = 0.42
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 700)
    options = drawer.drawOptions()
    options.bondLineWidth = 2
    options.setBackgroundColour((1, 1, 1, 1))
    drawer.DrawMolecule(mol, highlightAtoms=list(colors), highlightBonds=[],
                        highlightAtomColors=colors, highlightAtomRadii=radii)
    drawer.FinishDrawing()
    image = mpimg.imread(io.BytesIO(drawer.GetDrawingText()))
    ink = np.where((image[..., :3] < 0.97).any(axis=-1))          # non-white pixels
    pad = 25
    image = image[max(ink[0].min() - pad, 0):ink[0].max() + pad,
                  max(ink[1].min() - pad, 0):ink[1].max() + pad]
    ax.imshow(image)
    ax.axis("off")


# ============================== plotting helpers ==============================

def strip_box(ax, values, color, x=0.0, width=0.55, rng=None):
    """A box (quartiles + median) with every replicate drawn on top as a dot."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    rng = rng or np.random.default_rng(0)
    box = ax.boxplot([values], positions=[x], widths=width, showfliers=False,
                     patch_artist=True, manage_ticks=False,
                     medianprops=dict(color=INK, lw=1.8), whiskerprops=dict(color=INK, lw=0.9),
                     capprops=dict(color=INK, lw=0.9), boxprops=dict(color=INK, lw=0.9))
    box["boxes"][0].set_facecolor(to_rgba(color, 0.25))
    ax.scatter(x + rng.uniform(-0.17, 0.17, len(values)), values, s=24, color=color,
               edgecolors="white", linewidths=0.6, zorder=3)


def style_axis(ax, title, ylim):
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8)


# panel = (column, title, scale, ylim, success-filter threshold or None)
# A threshold is drawn wherever the success filter (mosaic_scan.py's _passing) defines one:
# ipSAE > 0.75, binder pLDDT > 0.8, both directional PAEs < 5 A, LigandMPNN recovery > 0.6.
# The filter has no criterion for ipTM, the interface PAE, the ligand pLDDT or burial, so those
# panels carry no line rather than an invented one.
ROW_STRUCTURE = [
    ("binder_plddt_pdb", "binder pLDDT", 1.0, (0.4, 1.0), 0.8),
    ("ligand_plddt", "ligand pLDDT", 1.0, (0.0, 1.0), None),
    ("iptm", "ipTM", 1.0, (0.0, 1.0), None),
    ("interface_pae", "interface PAE (Å)\nlower = better", 32.0, (0, 20), None),
    ("n_interface", "contacting binder residues\n(≤ cutoff from target)", 1.0, None, None),
]
ROW_FILTER = [
    ("mosaic_ipsae", "ipSAE", 1.0, (0.0, 1.0), 0.75),
    ("mosaic_binder_target_pae", "binder→target PAE (Å)\nlower = better", 1.0, (0, 26), 5.0),
    ("mosaic_target_binder_pae", "target→binder PAE (Å)\nlower = better", 1.0, (0, 26), 5.0),
    ("mosaic_recovery", "LigandMPNN recovery", 1.0, (0.1, 0.9), 0.6),
    ("mosaic_plddt", "binder pLDDT\n(re-prediction)", 1.0, (0.4, 1.0), 0.8),
]
SECONDARY = [("dssp_H", "helix"), ("dssp_E", "strand"), ("dssp_L", "loop")]
# Only for runs started from a salad blueprint. The success filter has no criterion for these.
BLUEPRINT_ROW = [
    ("blueprint_ca_rmsd", "CA RMSD to blueprint (Å)\nlower = closer to the design", 1.0, None, None),
    ("dssp_identity", "DSSP identity to blueprint\n(fraction of residues)", 1.0, (0.0, 1.0), None),
]


# ============================== figure 1: metrics ==============================

def fig_metrics(label, subtitle, rep, color, ylim_contacts, ylim_rmsd, path, dpi):
    has_blueprint = ("blueprint_ca_rmsd" in rep.columns
                     and rep["blueprint_ca_rmsd"].notna().any())
    n_rows = 3 if has_blueprint else 2
    height = 2.35 * n_rows + 1.9                      # inches; a fixed header and legend band
    fig, axes = plt.subplots(n_rows, 6, figsize=(15.8, height))
    fig.subplots_adjust(left=0.06, right=0.985, top=1 - 1.75 / height, bottom=0.85 / height,
                        wspace=0.55, hspace=0.78)
    rng = np.random.default_rng(0)

    def draw(ax, key, title, scale, ylim, thr):
        strip_box(ax, rep[key].to_numpy(dtype=float) * scale, color, rng=rng)
        if thr is not None:
            ax.axhline(thr, color=PASS_GREEN, lw=1.2, ls=":", zorder=1)
        style_axis(ax, title, ylim)
        ax.set_xlim(-0.8, 0.8)

    for ax, (key, title, scale, ylim, thr) in zip(axes[0], ROW_STRUCTURE):
        draw(ax, key, title, scale, ylim or ylim_contacts, thr)
    for ax, (key, title, scale, ylim, thr) in zip(axes[1], ROW_FILTER):
        draw(ax, key, title, scale, ylim, thr)
    axes[1, 5].axis("off")
    if has_blueprint:
        for ax, (key, title, scale, ylim, thr) in zip(axes[2], BLUEPRINT_ROW):
            draw(ax, key, title, scale, ylim or ylim_rmsd, thr)
        for ax in axes[2, len(BLUEPRINT_ROW):]:
            ax.axis("off")

    ax = axes[0, 5]                                   # secondary structure: three boxes
    for i, (key, name) in enumerate(SECONDARY):
        strip_box(ax, rep[key], color, x=i, width=0.55, rng=rng)
    style_axis(ax, "secondary structure\n(fraction of binder)", (0, 1))
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in SECONDARY], fontsize=8)
    ax.set_xlim(-0.6, 2.6)

    left = axes[0, 0].get_position().x0 - 0.04
    head = lambda row: axes[row, 0].get_position().y1 + 0.56 / height     # above the 2-line titles
    fig.text(left, head(0), "Final structure  (best of the diffusion samples)",
             fontsize=10, color=GREY, fontweight="bold")
    fig.text(left, head(1), "Success-filter metrics  (separate single-sample re-prediction)",
             fontsize=10, color=GREY, fontweight="bold")
    if has_blueprint:
        fig.text(left, head(2), "Fold vs. the salad blueprint the run started from  (best-of-4 structure)",
                 fontsize=10, color=GREY, fontweight="bold")
    fig.text(left, head(0) + 0.30 / height, subtitle, fontsize=9.5, color=INK, va="bottom")
    fig.text(left, head(0) + 0.56 / height, label, fontsize=15, fontweight="bold", va="bottom")
    legend = [
        Line2D([], [], marker="o", ls="", color=color, markeredgecolor="white",
               label="one replicate"),
        Line2D([], [], marker="s", ls="", color=to_rgba(color, 0.25),
               markeredgecolor=INK, label="box = quartiles, bar = median"),
        Line2D([], [], color=PASS_GREEN, ls=":", lw=1.4,
               label="success-filter threshold (only where the filter defines one)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================== figure 2: interface amino-acid types ==============================

def fig_interface(label, subtitle, enrich, n_if, n_excluded, min_interface, lim, cutoff,
                  path, dpi):
    """enrich: DataFrame (replicates x class) of interface - rest, in percentage points."""
    classes = list(AA_CLASSES)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    rng = np.random.default_rng(0)
    n_rep = len(enrich)
    need = int(np.ceil(REPLICATES_FOR_CONSISTENT * n_rep))
    for i, cls in enumerate(classes):
        v = enrich[cls].dropna().to_numpy()
        k_pos, k_neg = int((v > 0).sum()), int((v < 0).sum())
        consistent = max(k_pos, k_neg) >= need
        mean = float(v.mean()) if len(v) else 0.0
        ax.bar(i, mean, width=0.62, color=BLUE if mean > 0 else ORANGE,
               alpha=1.0 if consistent else 0.35, zorder=2)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=17, color=INK, alpha=0.55,
                   linewidths=0, zorder=3)
        ax.text(i, lim * 0.97, f"{k_pos}/{len(v)}", ha="center", va="top", fontsize=9,
                fontweight="bold" if consistent else "normal",
                color=INK if consistent else GREY)
    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{c}\n{AA_CLASSES[c]}" for c in classes], fontsize=8.5)
    ax.set_ylabel("interface − rest of binder\n(percentage points of residues)")
    ax.text(-0.5, lim * 0.97, "replicates\nenriched →", ha="right", va="top",
            fontsize=7.5, color=GREY)
    ax.set_title(f"{label}: which residue types sit at the interface?", loc="left",
                 fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK)
    fig.text(0.01, -0.02,
             f"Interface = binder residues with a heavy atom ≤ {cutoff:g} Å from the target "
             f"(median {int(np.median(n_if))} per replicate, range {int(n_if.min())}–"
             f"{int(n_if.max())}).\nRest = all other residues of the same binder. Each dot is one "
             f"replicate, the bar is their mean. Strong colour = at least {need}/{n_rep} "
             f"replicates agree\nin direction; faded = inconsistent between replicates. "
             f"Blue = enriched, orange = depleted."
             + (f"\n{n_excluded} replicate(s) with fewer than {min_interface} interface residues "
                f"are left out (a composition from so few residues is noise)."
                if n_excluded else ""),
             fontsize=7.8, color=GREY, va="top")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_interface_pooled(label, subtitle, rows, cutoff, path, dpi, n_boot=2000):
    """Interface composition pooled over replicates, for targets whose interface is only a few
    residues per replicate (e.g. a single ion). Per-replicate compositions of 2-4 residues are
    noise, so the residues of all replicates are pooled and the uncertainty is a 95% bootstrap
    interval over *replicates* (resampling whole replicates keeps their correlation intact)."""
    classes = list(AA_CLASSES)
    by_class = {c: rows[rows["aa_class"] == c].sort_values("row_index") for c in classes}
    n_if = by_class[classes[0]]["n_interface"].to_numpy(float)
    n_rest = by_class[classes[0]]["n_rest"].to_numpy(float)
    ci = np.stack([by_class[c]["interface_frac"].to_numpy(float) * n_if for c in classes], axis=1)
    cr = np.stack([by_class[c]["rest_frac"].to_numpy(float) * n_rest for c in classes], axis=1)

    def pooled(idx):
        return 100.0 * (ci[idx].sum(0) / n_if[idx].sum() - cr[idx].sum(0) / n_rest[idx].sum())

    rng = np.random.default_rng(0)
    point = pooled(np.arange(len(n_if)))
    boot = np.array([pooled(rng.integers(0, len(n_if), len(n_if))) for _ in range(n_boot)])
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    lim = float(np.ceil(max(np.abs(low).max(), np.abs(high).max()) / 10.0) * 10.0)

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for i, c in enumerate(classes):
        sure = low[i] > 0 or high[i] < 0              # interval excludes zero
        ax.bar(i, point[i], width=0.62, color=BLUE if point[i] > 0 else ORANGE,
               alpha=1.0 if sure else 0.35, zorder=2)
        ax.errorbar(i, point[i], yerr=[[point[i] - low[i]], [high[i] - point[i]]], color=INK,
                    capsize=4, lw=1.2, zorder=3)
    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{c}\n{AA_CLASSES[c]}" for c in classes], fontsize=8.5)
    ax.set_ylabel("interface − rest of binder\n(percentage points of residues)")
    ax.set_title(f"{label}: which residue types sit at the interface?", loc="left",
                 fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK)
    fig.text(0.01, -0.02,
             f"POOLED over replicates: the interface is only ~{int(np.median(n_if))} residue(s) per "
             f"replicate, too few to compare replicate by replicate.\n{int(n_if.sum())} interface "
             f"residues pooled (≤ {cutoff:g} Å from the target). Bars = pooled difference to the rest "
             f"of the binders; whiskers = 95% bootstrap interval over\nreplicates. Strong colour = "
             f"interval excludes zero; faded = it does not. Blue = enriched, orange = depleted.",
             fontsize=7.8, color=GREY, va="top")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================== figure 3: contacts and burial ==============================

def contact_units(atoms: pd.DataFrame, cutoff: float, by_residue: bool):
    """Contact frequency per target atom (or per residue, for DNA) across replicates."""
    keys = ["chain", "resseq", "resname"] + ([] if by_residue else ["atom", "element"])
    a = atoms.assign(contact=atoms["min_dist"] <= cutoff)
    if by_residue:                        # a nucleotide contacts if any of its atoms does
        a = a.groupby(["row_index"] + keys, sort=False)["contact"].any().reset_index()
    return a.groupby(keys, sort=False)["contact"].mean().reset_index()


def atom_burial(atoms: pd.DataFrame) -> pd.Series:
    """Mean fraction of each atom's solvent-accessible surface that the binder covers.

    1 - SASA(in complex) / SASA(alone), averaged over replicates. Atoms with almost no exposed
    surface even on their own (< 0.5 A^2, e.g. inside a ring system) have no meaningful ratio and
    are left out of the mean (NaN if that is true in every replicate)."""
    ratio = np.where(atoms["sasa_alone"] > 0.5,
                     1.0 - atoms["sasa_complex"] / atoms["sasa_alone"], np.nan)
    return atoms.assign(burial=ratio).groupby("atom", sort=False)["burial"].mean()


def burial_panel(ax, rep, color, args, rng):
    """Overall burial of the target, one dot per replicate, with the three burial classes."""
    ax.axhspan(0, args.surface_max, color=CLASS_COLOR["surface"], alpha=0.55, zorder=0)
    ax.axhspan(args.surface_max, args.pocket_min, color=CLASS_COLOR["partial"], alpha=0.25, zorder=0)
    ax.axhspan(args.pocket_min, 1.0, color=CLASS_COLOR["pocket"], alpha=0.25, zorder=0)
    strip_box(ax, rep["buried_fraction"], color, rng=rng)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 2.1)                 # room on the right for the class counts
    ax.set_xticks([])
    ax.set_ylabel("fraction of target surface\ncovered by the binder")
    ax.set_title("Surface or pocket?", loc="left", fontsize=11, fontweight="bold")
    counts = rep["burial_class"].value_counts()
    for name, y in (("pocket", 0.85), ("partial", 0.55), ("surface", 0.2)):
        ax.text(0.5, y, f"{CLASS_LABEL[name]}\n{counts.get(name, 0)}/{len(rep)}",
                ha="left", va="center", fontsize=9, color=INK)


def fig_contacts(label, subtitle, entry, drawing, atoms, rep, cutoff, args, color, path, dpi):
    is_dna = entry["target_kind"] == "dna"
    single = atoms["atom"].nunique() == 1 and not is_dna
    freq = contact_units(atoms, cutoff, by_residue=is_dna)
    rng = np.random.default_rng(0)

    if single:                                       # one ion: only the burial question applies
        fig = plt.figure(figsize=(6.2, 5.2))
        ax = fig.add_axes([0.16, 0.14, 0.78, 0.62])
        burial_panel(ax, rep, color, args, rng)
        in_contact = int((atoms["min_dist"] <= cutoff).sum())
        fig.text(0.16, 0.04, f"A single atom has no orientation: it is within {cutoff:g} Å of the "
                 f"binder in {in_contact}/{len(rep)} replicates.", fontsize=8, color=GREY)
        top = 0.76
    else:
        fig = plt.figure(figsize=(13.5, 5.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 0.95, 0.95], wspace=0.3,
                              left=0.04, right=0.985, top=0.84, bottom=0.2)
        ax = fig.add_subplot(gs[0])
        if drawing is not None:
            mol, names = drawing
            draw_shaded_molecule(ax, mol, names, atom_burial(atoms).to_dict())
            fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CONTACT_CMAP), ax=ax,
                         orientation="horizontal", fraction=0.05, pad=0.02,
                         label="fraction of the atom's solvent-accessible surface covered by the "
                               "binder\n(0 = exposed, 1 = buried; grey = no exposed surface)")
            ax.set_title("Which target atoms are buried by the binder?", loc="left",
                         fontsize=11, fontweight="bold")
        else:
            labels = ([f"{r.chain}:{r.resname}{r.resseq}" for r in freq.itertuples()] if is_dna
                      else [r.atom for r in freq.itertuples()])
            ax.bar(range(len(freq)), freq["contact"], color=color, width=0.75)
            ax.set_xticks(range(len(freq)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(f"fraction of replicates in contact (≤ {cutoff:g} Å)")
            ax.set_title("Which nucleotides contact the binder?" if is_dna
                         else "Which target atoms contact the binder?",
                         loc="left", fontsize=11, fontweight="bold")
        burial_panel(fig.add_subplot(gs[1]), rep, color, args, rng)
        ax = fig.add_subplot(gs[2])
        strip_box(ax, rep["atoms_in_contact"], color, rng=rng)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([])
        ax.set_ylabel(f"fraction of target atoms\nwithin {cutoff:g} Å of the binder")
        ax.set_title("How much of it touches?", loc="left", fontsize=11, fontweight="bold")
        top = 0.84

    fig.text(0.04, top + 0.12, label, fontsize=15, fontweight="bold", va="bottom")
    fig.text(0.04, top + 0.075, subtitle, fontsize=9.5, color=INK, va="bottom")
    caption = (f"Burial = 1 − SASA(target in complex) / SASA(target alone), 1.4 Å probe. "
               f"Surface-bound < {args.surface_max:g} ≤ partially buried < {args.pocket_min:g} ≤ "
               f"pocket-buried.")
    if drawing is not None:
        caption += (f"\nAtom shading is the mean over replicates. Contact alone (≤ {cutoff:g} Å) "
                    f"saturates for most small molecules, because a ~100-residue binder wraps them; "
                    f"burial separates the buried end from the exposed end.")
    if is_dna:
        caption += (f"\nA nucleotide counts as in contact if any of its atoms is. Burial is relative "
                    f"to the target's own size, so a 12-bp DNA can hardly be 'buried' by a "
                    f"~100-residue binder: judge DNA by the contacting nucleotides.")
    fig.text(0.04, 0.01 if not single else -0.02, caption, fontsize=7.8, color=GREY, va="bottom"
             if not single else "top")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================== overview of all targets ==============================

# column = (data column, title, scale, xlim, success-filter threshold or None)
# The two PAE columns are the directional ones the success filter uses (< 5 A), instead of the
# interface PAE, so that every criterion of the filter is shown with its threshold.
OVERVIEW_COLUMNS = [
    ("binder_plddt_pdb", "binder pLDDT", 1.0, (0.4, 1.0), 0.8),
    ("ligand_plddt", "ligand pLDDT", 1.0, (0.0, 1.0), None),
    ("iptm", "ipTM", 1.0, (0.0, 1.0), None),
    ("mosaic_ipsae", "ipSAE", 1.0, (0.0, 1.0), 0.75),
    ("mosaic_binder_target_pae", "binder→target PAE (Å)\nlower = better", 1.0, (0, 26), 5.0),
    ("mosaic_target_binder_pae", "target→binder PAE (Å)\nlower = better", 1.0, (0, 26), 5.0),
    ("mosaic_recovery", "LigandMPNN\nrecovery", 1.0, (0.1, 0.9), 0.6),
    ("buried_fraction", "target burial\n(fraction of surface covered)", 1.0, (0.0, 1.0), None),
]


def overview_columns(data):
    """OVERVIEW_COLUMNS, with the blueprint metrics before the burial column when present."""
    columns = list(OVERVIEW_COLUMNS)
    if "blueprint_ca_rmsd" in data.columns and data["blueprint_ca_rmsd"].notna().any():
        top = float(np.ceil(data["blueprint_ca_rmsd"].max() / 5.0) * 5.0)
        columns[-1:-1] = [
            ("blueprint_ca_rmsd", "CA RMSD to\nblueprint (Å)", 1.0, (0, top), None),
            ("dssp_identity", "DSSP identity\nto blueprint", 1.0, (0.0, 1.0), None)]
    return columns


def fig_overview(data, ordered, args, path, dpi):
    """All targets at once: one row per target, one column per metric. Every cell is a boxplot
    (quartiles, median, whiskers) with the replicates drawn on top as small dots, so neither the
    summary nor the individual replicates are hidden. Success-filter thresholds are thick dashed
    lines, with their value written at the top of the column."""
    # rows grouped by category, with a gap between groups
    rows, y, last = [], 0.0, None
    for cond in ordered:
        entry = cond["entry"]
        if last is not None and entry["category"] != last:
            y += 0.9
        rows.append((y, cond, entry))
        y += 1.0
        last = entry["category"]

    n_replicates = int(sum(len(data[data['condition_index'] == c['condition_index']])
                          for _, c, _ in rows))
    columns = overview_columns(data)
    n_col = len(columns)
    fig, axes = plt.subplots(1, n_col, figsize=(1.95 * n_col + 3.4, 0.31 * y + 2.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=1 - 1.55 / fig.get_figheight(),
                        bottom=0.9 / fig.get_figheight(), wspace=0.2)
    rng = np.random.default_rng(0)

    for ax, (key, title, scale, xlim, thr) in zip(axes, columns):
        if key == "buried_fraction":                  # the three burial classes as bands
            ax.axvspan(0, args.surface_max, color=CLASS_COLOR["surface"], alpha=0.55, zorder=0)
            ax.axvspan(args.surface_max, args.pocket_min, color=CLASS_COLOR["partial"],
                       alpha=0.25, zorder=0)
            ax.axvspan(args.pocket_min, 1.0, color=CLASS_COLOR["pocket"], alpha=0.25, zorder=0)
        for yi, cond, entry in rows:
            rep = data[data["condition_index"] == cond["condition_index"]]
            vals = rep[key].to_numpy(dtype=float) * scale
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            color = CATEGORY_COLOR[entry["category"]]
            box = ax.boxplot([vals], positions=[yi], vert=False, widths=0.62, showfliers=False,
                             patch_artist=True, manage_ticks=False,
                             medianprops=dict(color=INK, lw=2.2),
                             whiskerprops=dict(color=INK, lw=1.0),
                             capprops=dict(color=INK, lw=1.0),
                             boxprops=dict(color=INK, lw=1.0))
            box["boxes"][0].set_facecolor(to_rgba(color, 0.32))
            box["boxes"][0].set_zorder(2.5)
            ax.scatter(vals, yi + rng.uniform(-0.2, 0.2, len(vals)), s=9, color=color,
                       alpha=0.85, linewidths=0, zorder=4)
        if thr is not None:
            # behind the translucent boxes but clearly visible: thick, dashed, saturated green
            ax.axvline(thr, color=THRESHOLD_GREEN, lw=2.8, ls=(0, (4, 2)), zorder=2.0)
            sign = "<" if "pae" in key else ">"
            ax.text(thr + 0.012 * (xlim[1] - xlim[0]), -0.68, f"{sign} {thr:g}", ha="left",
                    va="top", fontsize=8.5, fontweight="bold", color=THRESHOLD_GREEN, zorder=5)
        ax.set_xlim(xlim[0], xlim[1] + (0.03 if xlim[1] == 1.0 else 0.0))
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", color="#F0F0F0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    # target labels (coloured by category) and the success-filter pass count
    ax0 = axes[0]
    ax0.set_yticks([yi for yi, _, _ in rows])
    labels = []
    for yi, cond, entry in rows:
        rep = data[data["condition_index"] == cond["condition_index"]]
        labels.append(f"{_short_label(entry['name'], entry['category'])}   "
                      f"{int((rep['passed'] == 'True').sum())}/{len(rep)}")
    ax0.set_yticklabels(labels, fontsize=9)
    for tick, (_, _, entry) in zip(ax0.get_yticklabels(), rows):
        tick.set_color(CATEGORY_COLOR[entry["category"]])
    ax0.set_ylim(y - 0.4, -0.7)                         # first target at the top
    ax0.text(-0.02, 1.0, "target     passes filter", transform=ax0.transAxes, ha="right",
             va="bottom", fontsize=8, color=GREY)

    fig.text(0.01, 1 - 0.34 / fig.get_figheight(),
             f"All targets  ·  {n_replicates} replicates  ·  {len(rows)} targets  ·  {args.run_path}",
             fontsize=14, fontweight="bold", va="top")
    legend = [Line2D([], [], marker="o", ls="", color=CATEGORY_COLOR[c], label=CATEGORY_LABEL[c])
              for c in dict.fromkeys(e["category"] for _, _, e in rows)]
    legend += [Line2D([], [], color=INK, lw=2.2, label="median (box = quartiles)"),
               Line2D([], [], color=THRESHOLD_GREEN, ls=(0, (4, 2)), lw=2.8,
                      label="success-filter threshold (only where the filter defines one)")]
    fig.legend(handles=legend, loc="upper left", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.01, 1 - 0.62 / fig.get_figheight()))
    fig.text(0.01, 0.2 / fig.get_figheight(),
             f"Boxes show the quartiles and median of the 10 replicates; dots are the replicates. Burial bands: surface-bound < {args.surface_max:g} ≤ "
             f"partially buried < {args.pocket_min:g} ≤ pocket-buried. DNA burial is relative to the "
             f"target's own size and is not comparable with the small molecules.\n"
             f"pLDDT panels come from the best-of-4 structure; the filter's own pLDDT criterion is "
             f"scored on a separate re-prediction (shown in the per-target figures).",
             fontsize=8, color=GREY)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================== summary ==============================

def summarize_target(label, entry, rep, enrich, atoms, cutoff, pooled_mode=False):
    is_dna = entry["target_kind"] == "dna"
    if is_dna:
        freq = contact_units(atoms, cutoff, by_residue=True)
        contacted = freq[freq["contact"] >= 0.8]
        buried_atoms = exposed_atoms = ""
        contacted_units = ";".join(f"{r.chain}:{r.resname}{r.resseq}" for r in contacted.itertuples())
    else:
        # contact saturates for small molecules (many atoms touch in every replicate), so the
        # atoms are ranked by burial instead
        burial = atom_burial(atoms).dropna().sort_values()
        exposed_atoms = ";".join(f"{n}:{v:.2f}" for n, v in burial.head(3).items())
        buried_atoms = ";".join(f"{n}:{v:.2f}" for n, v in burial.tail(3)[::-1].items())
        contacted_units = ""
    n_rep = len(rep)
    k_pos = {c: int((enrich[c] > 0).sum()) for c in AA_CLASSES}
    mean = {c: float(enrich[c].mean()) for c in AA_CLASSES}
    best = max(AA_CLASSES, key=lambda c: mean[c])
    worst = min(AA_CLASSES, key=lambda c: mean[c])
    row = dict(
        target=label, category=entry["category"], n_replicates=n_rep,
        n_replicates_in_enrichment=len(enrich),
        enrichment_mode="pooled" if pooled_mode else "per-replicate",
        n_pass_filter=int((rep["passed"] == "True").sum()),
        target_atoms=int(rep["target_atoms"].iloc[0]),
        binder_plddt_median=rep["binder_plddt_pdb"].median(),
        ligand_plddt_median=rep["ligand_plddt"].median(),
        plddt_all_tokens_median=rep["plddt"].median(),
        binder_plddt_reprediction_median=rep["mosaic_plddt"].median(),
        iptm_median=rep["iptm"].median(),
        **({"blueprint_ca_rmsd_median": rep["blueprint_ca_rmsd"].median(),
            "dssp_identity_median": rep["dssp_identity"].median()}
           if "blueprint_ca_rmsd" in rep.columns else {}),
        ipsae_median=rep["mosaic_ipsae"].median(),
        interface_pae_A_median=32 * rep["interface_pae"].median(),
        binder_to_target_pae_A_median=rep["mosaic_binder_target_pae"].median(),
        target_to_binder_pae_A_median=rep["mosaic_target_binder_pae"].median(),
        ligandmpnn_recovery_median=rep["mosaic_recovery"].median(),
        contacting_residues_median=rep["n_interface"].median(),
        buried_fraction_median=rep["buried_fraction"].median(),
        atoms_in_contact_median=rep["atoms_in_contact"].median(),
        n_surface=int((rep["burial_class"] == "surface").sum()),
        n_partial=int((rep["burial_class"] == "partial").sum()),
        n_pocket=int((rep["burial_class"] == "pocket").sum()),
        majority_burial=rep["burial_class"].mode().iloc[0],
        most_enriched_class=best, most_enriched_pp=mean[best],
        most_enriched_class_enriched_in=f"{k_pos[best]}/{len(enrich)}",
        most_depleted_class=worst, most_depleted_pp=mean[worst],
        most_depleted_class_enriched_in=f"{k_pos[worst]}/{len(enrich)}",
        most_buried_atoms=buried_atoms, most_exposed_atoms=exposed_atoms,
        contacted_nucleotides=contacted_units)
    if pooled_mode:
        # a per-replicate "most enriched class" from the one or two replicates that have a large
        # enough interface is not meaningful; the pooled result is in the figure instead
        for key in ("most_enriched_class", "most_enriched_pp", "most_enriched_class_enriched_in",
                    "most_depleted_class", "most_depleted_pp", "most_depleted_class_enriched_in"):
            row[key] = np.nan
    return row


# ============================== main ==============================

def check_pldt(data: pd.DataFrame):
    """Check that the PDB B-factors really are the pLDDT the pipeline reported.

    results.csv's `plddt` is the mean over *all* tokens of the final structure - binder and
    target together - so it is reproduced here from the B-factors and compared. This validates
    reading pLDDT from the B-factor column, the per-token averaging, and which structure the
    saved PDB is. It also shows that `plddt` is not a binder-only value."""
    d = data[data["category"] != "apo"]
    diff = (d["plddt"] - d["plddt_all_tokens"]).abs()
    binder_only = (d["plddt"] - d["binder_plddt_pdb"]).abs()
    print(f"pLDDT check on {len(d)} replicates: results.csv `plddt` vs the all-token mean rebuilt "
          f"from B-factors: max |diff| = {diff.max():.4f} (median {diff.median():.4f});"
          f" vs the binder-only mean: median |diff| = {binder_only.median():.4f}")
    if diff.max() > 0.02:
        print("  [warn] the B-factors do not reproduce results.csv pLDDT; ligand pLDDT may not be "
              "on the same structure/scale. Do not trust it until this is understood.")


def check_blueprint(data: pd.DataFrame):
    """Report the DSSP re-check: the final structure's DSSP, computed here with the pipeline's own
    function, against the dssp_hard string the pipeline wrote to results.csv."""
    if "dssp_matches_results" not in data.columns:
        return
    m = data["dssp_matches_results"].dropna()
    print(f"blueprint metrics for {data['blueprint_ca_rmsd'].notna().sum()} replicates. DSSP "
          f"re-check against results.csv dssp_hard: identical for {int((m == 1.0).sum())}/{len(m)} "
          f"replicates (min agreement {m.min():.2f})")
    if m.min() < 0.95:
        print("  [warn] this DSSP does not reproduce the pipeline's; the identity to the blueprint "
              "is then not comparable with results.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", default="runs/ligand_hallucination_ligmpnn")
    ap.add_argument("--out_path", default=None, help="default: <run_path>/analysis")
    ap.add_argument("--contact_cutoff", type=float, default=4.5)
    ap.add_argument("--surface_max", type=float, default=0.40)
    ap.add_argument("--pocket_min", type=float, default=0.70)
    ap.add_argument("--min_interface", type=int, default=5,
                    help="replicates with fewer interface residues are left out of the "
                         "amino-acid enrichment (a composition from 2 residues is noise)")
    ap.add_argument("--only", default="", help="comma list of target names to plot")
    ap.add_argument("--recompute", action="store_true", help="redo the PDB measurements")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    out = args.out_path or f"{args.run_path}/analysis"
    # read the run first: creating the output folder up front would, for a mistyped --run_path,
    # leave behind an empty directory tree that looks like a run
    if not os.path.exists(f"{args.run_path}/ligands.json"):
        raise SystemExit(f"{args.run_path!r} is not a ligand_bias run (no ligands.json)")
    manifest = json.load(open(f"{args.run_path}/ligands.json"))
    os.makedirs(out, exist_ok=True)
    with open(f"{args.run_path}/results.csv") as f:
        n_cols = len(f.readline().rstrip("\n").split(","))
    df = _load_results_csv(f"{args.run_path}/results.csv", n_cols=n_cols)
    bad = df[df["status"] != "ok"]
    print(f"{len(df)} rows, {len(bad)} not ok" + (f" (dropped): {sorted(bad.row_index)}" if len(bad) else ""))
    df = df[df["status"] == "ok"].copy()

    cache = [f"{out}/{n}.csv" for n in ("per_attempt", "interface_aa", "ligand_atoms")]
    cached = None
    if all(os.path.exists(p) for p in cache) and not args.recompute:
        # Only an empty field is missing data. pandas would otherwise also read the string "NA"
        # as NaN, which is the real name of an atom in HEM (nitrogen A) and would silently drop it.
        cached = [pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in cache]
        if "ligand_plddt" not in cached[0].columns:
            print("cached measurements predate the pLDDT columns -> recomputing")
            cached = None
    if cached is not None:
        print("using cached measurements (--recompute to redo)")
        per_attempt, interface_aa, ligand_atoms = cached
    else:
        print("measuring predicted complexes ...")
        per_attempt, interface_aa, ligand_atoms = measure_run(
            args.run_path, df, args.contact_cutoff, args.surface_max, args.pocket_min)
        for table, p in zip((per_attempt, interface_aa, ligand_atoms), cache):
            table.to_csv(p, index=False)

    data = df.merge(per_attempt, on="row_index", how="left")
    check_pldt(data)
    check_blueprint(data)
    # shared axis limits, so figures of different targets are directly comparable
    ylim_contacts = (0, float(np.ceil(per_attempt["n_interface"].max() / 5.0) * 5.0))
    ylim_rmsd = ((0, float(np.ceil(per_attempt["blueprint_ca_rmsd"].max() / 5.0) * 5.0))
                 if "blueprint_ca_rmsd" in per_attempt.columns else None)
    interface_all = interface_aa.dropna(subset=["interface_frac", "rest_frac"])
    big_enough = per_attempt.loc[per_attempt["n_interface"] >= args.min_interface, "row_index"]
    interface_aa = interface_aa[interface_aa["row_index"].isin(big_enough)]
    lim_enrich = float(np.ceil(interface_aa["enrichment_pp"].abs().max() / 10.0) * 10.0)

    only = {s for s in args.only.split(",") if s}
    summary = []
    ordered = [c for c in _condition_order(manifest["conditions"]) if c["entry"] is not None]
    for position, cond in enumerate(ordered, start=1):
        entry = cond["entry"]
        label = _short_label(entry["name"], entry["category"])
        if only and label not in only and entry["name"] not in only:
            continue
        rep = data[data["condition_index"] == cond["condition_index"]].reset_index(drop=True)
        rows = rep["row_index"].tolist()
        atoms = ligand_atoms[ligand_atoms["row_index"].isin(rows)]
        enrich = (interface_aa[interface_aa["row_index"].isin(rows)]
                  .pivot(index="row_index", columns="aa_class", values="enrichment_pp"))
        n_if = per_attempt[per_attempt["row_index"].isin(rows)]["n_interface"].to_numpy()
        n_excluded = len(rep) - len(enrich)
        color = CATEGORY_COLOR[entry["category"]]

        n_target_atoms = int(rep["target_atoms"].iloc[0])
        if entry["target_kind"] == "dna":
            size = f"{atoms[['chain', 'resseq']].drop_duplicates().shape[0]} nucleotides"
        else:
            size = "single atom" if n_target_atoms == 1 else f"{n_target_atoms} heavy atoms"
        subtitle = (f"{CATEGORY_LABEL[entry['category']]}  ·  {size}  ·  "
                    f"{len(rep)} replicates  ·  {(rep['passed'] == 'True').sum()}/{len(rep)} "
                    f"pass the success filter  ·  median ligand pLDDT "
                    f"{rep['ligand_plddt'].median():.2f}")

        folder = f"{out}/{position:02d}_{label}"
        os.makedirs(folder, exist_ok=True)
        fig_metrics(label, subtitle, rep, color, ylim_contacts, ylim_rmsd,
                    f"{folder}/1_metrics.png", args.dpi)
        pooled_mode = len(enrich) < len(rep) / 2     # most replicates have a too-small interface
        if pooled_mode:
            print(f"  [note] {label}: only {len(enrich)}/{len(rep)} replicates have >= "
                  f"{args.min_interface} interface residues -> pooled interface composition")
            fig_interface_pooled(label, subtitle, interface_all[interface_all["row_index"].isin(rows)],
                                 args.contact_cutoff, f"{folder}/2_interface_aa.png", args.dpi)
        else:
            fig_interface(label, subtitle, enrich, n_if, n_excluded, args.min_interface, lim_enrich,
                          args.contact_cutoff, f"{folder}/2_interface_aa.png", args.dpi)

        drawing = None
        mapped = ligand_molecule(entry)
        if mapped is not None:
            mol, names = mapped
            if check_mapping(mol, names, atoms):
                drawing = (mol, names)
            else:
                print(f"  [warn] {label}: atom mapping failed verification -> bar chart instead")
        fig_contacts(label, subtitle, entry, drawing, atoms,
                     rep, args.contact_cutoff, args, color, f"{folder}/3_contacts_burial.png",
                     args.dpi)
        summary.append(summarize_target(label, entry, rep, enrich, atoms, args.contact_cutoff,
                                        pooled_mode))
        print(f"[ok] {position:02d} {label}")

    fig_overview(data, ordered, args, f"{out}/overview.png", args.dpi)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{out}/ligand_summary.csv", index=False)
    print(f"\nwrote figures and tables to {out}/")


if __name__ == "__main__":
    main()
