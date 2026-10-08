"""Is each target bound at the surface of its binder, or buried in a pocket?

For every attempt of a ligand_bias run (small molecule or DNA target) this reads the final
re-predicted complex (<run>/predictions/*.pdb; chain A = binder, every other chain = target) and
measures, per attempt:

  buried_fraction    fraction of the target's solvent-accessible surface that the binder covers:
                     1 - SASA(target in complex) / SASA(target alone)   (Shrake-Rupley, 1.4 A probe)
  contact_residues   binder residues with any heavy atom within --contact_cutoff (4.5 A) of the
                     target; shell_residues is the same within --shell_cutoff (6 A)
  atoms_contacted    fraction of target heavy atoms within the contact cutoff of the binder

and classes each attempt as surface-bound (buried_fraction < --surface_max, default 0.40),
partially buried, or pocket-buried (>= --pocket_min, default 0.70). The burial is relative to the
target's own size, so a 12-bp DNA duplex can never be "buried" by a 100-residue binder: use
contact_residues for size-independent comparison across target types.

Writes into --out_path (default: the run folder):
    ligand_burial_per_attempt.csv   one row per attempt
    ligand_burial_summary.csv       one row per target: medians and class fractions
    ligand_burial.png               burial fraction, burial classes and contacting residues per target
    ligand_burial_vs_contacts.png   every attempt as contacts vs. burial (the pocket quadrant)

Usage:
    python3 scripts/analyze_ligand_burial.py --run_path runs/ligand_hallucination_pmpnninit \\
        --compare_path runs/ligand_bias_hallucination
"""

import copy
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley, ATOMIC_RADII

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ligand_bias import (  # same directory
    _load_results_csv, _condition_order, _short_label,
    CATEGORY_COLOR, CATEGORY_LABEL, CATEGORY_ORDER, GREY, INK)

RADII = dict(ATOMIC_RADII, FE=1.94)       # heme iron is missing from Biopython's table
CLASS_COLOR = {"surface": "#E8E8E8", "partial": "#9DB4D8", "pocket": "#4C72B0"}
CLASS_LABEL = {"surface": "surface-bound", "partial": "partially buried", "pocket": "pocket-buried"}


def measure(path: str, contact_cutoff: float, shell_cutoff: float) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PDBParser(QUIET=True).get_structure("x", path)[0]
    binder = model["A"]
    target_chains = [c for c in model if c.id != "A"]
    target_atoms = [a for c in target_chains for a in c.get_atoms()]
    sr = ShrakeRupley(probe_radius=1.4, n_points=100, radii_dict=RADII)

    sr.compute(model, level="A")
    sasa_complex = sum(a.sasa for a in target_atoms)
    alone = copy.deepcopy(model)
    alone.detach_child("A")
    sr.compute(alone, level="A")
    sasa_alone = sum(a.sasa for c in alone for a in c.get_atoms())

    binder_atoms = [(i, a.coord) for i, r in enumerate(binder) for a in r.get_atoms()]
    owner = np.array([i for i, _ in binder_atoms])
    tree = cKDTree(np.array([c for _, c in binder_atoms]))
    target_xyz = np.array([a.coord for a in target_atoms])
    near = tree.query_ball_point(target_xyz, r=contact_cutoff)
    shell = tree.query_ball_point(target_xyz, r=shell_cutoff)
    return dict(
        target_atoms=len(target_atoms), sasa_alone=sasa_alone, sasa_in_complex=sasa_complex,
        buried_fraction=float(1.0 - sasa_complex / sasa_alone),
        contact_residues=int(len({owner[j] for hits in near for j in hits})),
        shell_residues=int(len({owner[j] for hits in shell for j in hits})),
        atoms_contacted=float(np.mean([len(h) > 0 for h in near])))


def classify(frac: float, surface_max: float, pocket_min: float) -> str:
    return "surface" if frac < surface_max else ("pocket" if frac >= pocket_min else "partial")


def analyze_run(run_path, args):
    manifest = json.load(open(f"{run_path}/ligands.json"))
    with open(f"{run_path}/results.csv") as f:
        n_cols = len(f.readline().rstrip("\n").split(","))
    df = _load_results_csv(f"{run_path}/results.csv", n_cols=n_cols)
    df = df[(df["status"] == "ok") & (df["category"] != "apo")].copy()
    rows = []
    for _, r in df.iterrows():
        stem = f"{int(r['row_index']):04d}_{r['condition_name']}_a{int(r['attempt_index'])}"
        m = measure(f"{run_path}/predictions/{stem}.pdb", args.contact_cutoff, args.shell_cutoff)
        m.update(row_index=int(r["row_index"]), condition_index=int(r["condition_index"]),
                 attempt_index=int(r["attempt_index"]), condition_name=r["condition_name"],
                 category=r["category"],
                 burial_class=classify(m["buried_fraction"], args.surface_max, args.pocket_min))
        rows.append(m)
    per_attempt = pd.DataFrame(rows)
    ordered = [c for c in _condition_order(manifest["conditions"]) if c["entry"] is not None]
    label_of = {c["condition_index"]: _short_label(c["entry"]["name"], c["entry"]["category"])
                for c in ordered}
    per_attempt["label"] = per_attempt["condition_index"].map(label_of)
    order = [label_of[c["condition_index"]] for c in ordered]
    cat_of = {label_of[c["condition_index"]]: c["entry"]["category"] for c in ordered}
    return per_attempt, order, cat_of, manifest


def summarize(per_attempt, order):
    out = []
    for label in order:
        sub = per_attempt[per_attempt["label"] == label]
        out.append(dict(
            target=label, category=sub["category"].iloc[0], n_attempts=len(sub),
            target_atoms=int(sub["target_atoms"].iloc[0]),
            buried_fraction_median=sub["buried_fraction"].median(),
            contact_residues_median=sub["contact_residues"].median(),
            shell_residues_median=sub["shell_residues"].median(),
            atoms_contacted_mean=sub["atoms_contacted"].mean(),
            **{f"frac_{c}": float((sub["burial_class"] == c).mean()) for c in CLASS_COLOR},
            majority_class=sub["burial_class"].mode().iloc[0]))
    return pd.DataFrame(out)


def fig_per_target(per_attempt, order, cat_of, args, run_label, compare=None, compare_label=""):
    n = len(order)
    y = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.33 * n + 2.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.2], wspace=0.08))
    rng = np.random.default_rng(0)

    def dots(ax, col, comp_col=None):
        for yi, label in zip(y, order):
            vals = per_attempt.loc[per_attempt["label"] == label, col].to_numpy()
            color = CATEGORY_COLOR[cat_of[label]]
            ax.scatter(vals, yi + rng.uniform(-0.18, 0.18, len(vals)), s=16, color=color,
                       alpha=0.75, linewidths=0, zorder=3)
            ax.plot([np.median(vals)] * 2, [yi - 0.34, yi + 0.34], color=INK, lw=1.8, zorder=4)
            if compare is not None:
                cv = compare.loc[compare["label"] == label, col]
                if len(cv):
                    ax.scatter([cv.median()], [yi], marker="D", s=26, facecolors="white",
                               edgecolors=INK, linewidths=1.0, zorder=5)

    ax = axes[0]
    ax.axvspan(0, args.surface_max, color=CLASS_COLOR["surface"], alpha=0.5, zorder=0)
    ax.axvspan(args.pocket_min, 1.0, color=CLASS_COLOR["pocket"], alpha=0.12, zorder=0)
    dots(ax, "buried_fraction")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of target surface buried by the binder")
    ax.set_title("Burial (dots = attempts, bar = median)", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    for tick, label in zip(ax.get_yticklabels(), order):
        tick.set_color(CATEGORY_COLOR[cat_of[label]])
    ax.invert_yaxis()

    ax = axes[1]
    left = np.zeros(n)
    for c in ("surface", "partial", "pocket"):
        frac = np.array([(per_attempt.loc[per_attempt["label"] == l, "burial_class"] == c).mean()
                         for l in order])
        ax.barh(y, frac, left=left, color=CLASS_COLOR[c], edgecolor="white", linewidth=0.8,
                height=0.78, label=CLASS_LABEL[c])
        left += frac
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of attempts")
    ax.set_title("Burial class", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              title=f"surface < {args.surface_max:g} ≤ partial < {args.pocket_min:g} ≤ pocket",
              title_fontsize=7.5)

    ax = axes[2]
    dots(ax, "contact_residues")
    ax.set_xlabel(f"binder residues within {args.contact_cutoff:g} Å of the target")
    ax.set_title("Contacting residues", fontsize=10)
    if compare is not None:
        ax.scatter([], [], marker="D", s=26, facecolors="white", edgecolors=INK,
                   label=f"{compare_label} (median)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(f"Surface-bound or buried in a pocket? ({run_label})", y=1.0)
    fig.savefig(f"{args.out_path}/ligand_burial.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(per_attempt, args, run_label):
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.axhspan(args.pocket_min, 1.0, color=CLASS_COLOR["pocket"], alpha=0.10, zorder=0)
    ax.axhspan(0, args.surface_max, color=CLASS_COLOR["surface"], alpha=0.5, zorder=0)
    for cat in CATEGORY_ORDER[1:]:
        sub = per_attempt[per_attempt["category"] == cat]
        if len(sub):
            ax.scatter(sub["contact_residues"], sub["buried_fraction"], s=24,
                       color=CATEGORY_COLOR[cat], alpha=0.8, linewidths=0, label=CATEGORY_LABEL[cat])
    med = per_attempt.groupby("label").agg(x=("contact_residues", "median"),
                                           y=("buried_fraction", "median"),
                                           cat=("category", "first"))
    for label, r in med.iterrows():
        ax.scatter([r.x], [r.y], s=70, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=4)
        ax.annotate(label, (r.x, r.y), xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                    color=INK)
    ax.set_xlabel(f"binder residues within {args.contact_cutoff:g} Å of the target")
    ax.set_ylabel("fraction of target surface buried")
    ax.set_ylim(0, 1)
    ax.set_title(f"Pocket = many contacts and high burial ({run_label}); rings = per-target medians")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(f"{args.out_path}/ligand_burial_vs_contacts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", required=True)
    ap.add_argument("--compare_path", default=None, help="a second run, overlaid as hollow diamonds")
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--contact_cutoff", type=float, default=4.5)
    ap.add_argument("--shell_cutoff", type=float, default=6.0)
    ap.add_argument("--surface_max", type=float, default=0.40)
    ap.add_argument("--pocket_min", type=float, default=0.70)
    args = ap.parse_args()
    args.out_path = args.out_path or args.run_path

    per_attempt, order, cat_of, manifest = analyze_run(args.run_path, args)
    per_attempt.to_csv(f"{args.out_path}/ligand_burial_per_attempt.csv", index=False)
    summary = summarize(per_attempt, order)
    summary.to_csv(f"{args.out_path}/ligand_burial_summary.csv", index=False)

    compare = compare_label = None
    if args.compare_path:
        compare, _, _, cmp_manifest = analyze_run(args.compare_path, args)
        compare_label = f"{cmp_manifest['options'].get('init_mode', 'other')} init"
    run_label = f"{manifest['options'].get('init_mode', '?')} init, " \
                f"{manifest['options'].get('hallucination_steps', '?')} hallucination steps"
    fig_per_target(per_attempt, order, cat_of, args, run_label, compare, compare_label)
    fig_scatter(per_attempt, args, run_label)

    pd.set_option("display.width", 200)
    print(f"{len(per_attempt)} attempts; burial class counts: "
          f"{per_attempt['burial_class'].value_counts().to_dict()}")
    print(summary[["target", "category", "target_atoms", "buried_fraction_median",
                   "contact_residues_median", "frac_surface", "frac_partial", "frac_pocket",
                   "majority_class"]].round(2).to_string(index=False))
    for cat in CATEGORY_ORDER[1:]:
        sub = per_attempt[per_attempt["category"] == cat]
        if len(sub):
            print(f"  {cat:4s} median buried {sub.buried_fraction.median():.2f}, "
                  f"median contacting residues {sub.contact_residues.median():.0f}")
    print(f"Wrote ligand_burial_*.csv/png to {args.out_path}")


if __name__ == "__main__":
    main()
