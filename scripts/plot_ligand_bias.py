"""Analyze and visualize a flexcraft.pipelines.experimental.ligand_bias run.

Reads <run_path>/{results.csv, ligands.json, contact_frequency.npy, aa_preference.npy}
and writes, into <run_path> by default:

    ligand_bias_summary.csv        one row per condition: attempt-averaged stats
    ligand_bias_engagement.png     6-panel per-ligand summary (contact, confidence,
                                    fold vs. apo, sequence uncertainty)
    ligand_bias_interface_map.png  heatmap: per-residue contact frequency, ligand x position
    ligand_bias_aa_enrichment.png  heatmap: interface-minus-bulk amino-acid class enrichment

Usage:
    python3 scripts/plot_ligand_bias.py --run_path runs/ligand_bias

Known data quirk this script works around: a handful of Category-C (Enamine) SMILES carry
a CXSMILES extended-stereo annotation suffix (e.g. "...C1C |&1:3,4,6|") whose embedded commas
break results.csv's naive comma-joined rows (flexcraft.files.csv.ScoreCSV does not quote
fields). `_load_results_csv` below reconstructs those rows instead of dropping them; the
underlying predictions are unaffected (RDKit/Boltz both parse the CXSMILES suffix correctly,
only the CSV serialization of the raw string breaks).
"""

import os
import re
import json
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# House palette (flexcraft's own, reused from scripts/plot_mosaic_scan.py /
# plot_landscape_distogram_sweep.py -- validated CVD dE ~20, normal dE ~27).
BLUE, ORANGE, GREEN, PURPLE, GREY, INK = (
    "#4C72B0", "#DD8452", "#55A868", "#8172B2", "#8C8C8C", "#2B2B2B")
CATEGORY_ORDER = ["apo", "A", "B", "C", "DNA"]
CATEGORY_COLOR = {"apo": GREY, "A": BLUE, "B": ORANGE, "C": GREEN, "DNA": PURPLE}
CATEGORY_LABEL = {"apo": "apo (no target)", "A": "A: PDB/CCD cofactor",
                   "B": "B: ChEBI biological (not in PDB)", "C": "C: Enamine REAL (synthetic)",
                   "DNA": "DNA"}
SEQ_CMAP = LinearSegmentedColormap.from_list("house_seq", ["#FFFFFF", BLUE])
DIV_CMAP = LinearSegmentedColormap.from_list("house_div", [ORANGE, "#F2F2F2", BLUE])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "font.size": 10,
})

AA_CLASSES = ["hydrophobic", "aromatic", "positive", "negative", "polar", "special"]


def _load_results_csv(path: str, n_cols: int) -> pd.DataFrame:
    """Line-level CSV loader tolerant of extra commas embedded in the `target_repr`
    column (index 6) -- see module docstring. Any row with more than n_cols fields has
    its surplus fields folded back into column 6; anything else is a genuine parse
    error and raises rather than silently mis-aligning columns."""
    with open(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
    assert len(header) == n_cols, f"expected {n_cols} columns, header has {len(header)}"
    target_repr_idx = header.index("target_repr")
    rows = []
    with open(path, "rt") as f:
        f.readline()
        for line_no, line in enumerate(f, start=2):
            fields = line.rstrip("\n").split(",")
            if len(fields) == n_cols:
                rows.append(fields)
                continue
            extra = len(fields) - n_cols
            if extra <= 0:
                raise ValueError(f"{path}:{line_no}: {len(fields)} fields, expected {n_cols}")
            merged = ",".join(fields[target_repr_idx:target_repr_idx + extra + 1])
            fixed = fields[:target_repr_idx] + [merged] + fields[target_repr_idx + extra + 1:]
            assert len(fixed) == n_cols, f"{path}:{line_no}: repair failed"
            rows.append(fixed)
    df = pd.DataFrame(rows, columns=header)
    text_cols = {
        "condition_name", "category", "target_kind", "target_repr", "pairs_with", "catalog_id",
        "init_mode", "binder_sequence", "init_sequence", "passed", "contact_topk_residues", "dssp_hard",
        "aa_consensus_sequence", "status", "error"}
    for c in df.columns:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _short_label(name: str, category: str) -> str:
    if category == "C":
        m = re.match(r"enamine_[a-z]_(\d+)", name)
        return f"enamine_{m.group(1)}" if m else name[:14]
    if category == "B":
        return name.replace("CHEBI_", "")
    return name


def _condition_order(conditions: list) -> list:
    """apo first, then the categories in CATEGORY_ORDER, each in manifest order."""
    order = [c for c in conditions if c["entry"] is None]
    for cat in CATEGORY_ORDER[1:]:
        order += [c for c in conditions if c["entry"] and c["entry"]["category"] == cat]
    assert len(order) == len(conditions), "condition_order lost or duplicated an entry"
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_path", default="runs/ligand_bias")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()
    out_path = args.out_path or args.run_path

    with open(f"{args.run_path}/ligands.json") as f:
        manifest = json.load(f)
    num_aa = manifest["num_aa"]
    conditions = manifest["conditions"]
    with open(f"{args.run_path}/results.csv") as f:
        n_cols = len(f.readline().rstrip("\n").split(","))
    df = _load_results_csv(f"{args.run_path}/results.csv", n_cols=n_cols)
    run_label = (f"{manifest['options'].get('init_mode', 'init-only')} init, "
                 f"{manifest['options'].get('hallucination_steps', 0)} hallucination steps"
                 if "init_mode" in manifest["options"] else "init-only forward pass")
    contact = np.load(f"{args.run_path}/contact_frequency.npy")     # (N_ROWS, num_aa)
    aa_pref = np.load(f"{args.run_path}/aa_preference.npy")         # (N_ROWS, num_aa, 20)

    n_ok = (df["status"] == "ok").sum()
    print(f"Loaded {len(df)} rows ({n_ok} ok, {len(df) - n_ok} failed) across "
          f"{len(conditions)} conditions.")
    if n_ok < len(df):
        print(df.loc[df["status"] != "ok", ["row_index", "condition_name", "error"]]
              .to_string(index=False))
    df = df[df["status"] == "ok"].copy()

    order = _condition_order(conditions)
    labels, categories, cond_indices = [], [], []
    for c in order:
        cond_indices.append(c["condition_index"])
        if c["entry"] is None:
            labels.append("apo")
            categories.append("apo")
        else:
            labels.append(_short_label(c["entry"]["name"], c["entry"]["category"]))
            categories.append(c["entry"]["category"])
    colors = [CATEGORY_COLOR[cat] for cat in categories]

    # ---- per-condition aggregate summary (mean +/- SEM across attempts) ----
    agg_cols = ["plddt", "ptm", "iptm", "interface_pae", "contact_total", "contact_any_frac",
                "contact_mean_pos", "contact_std_pos", "rmsd_to_apo", "dssp_L", "dssp_H", "dssp_E",
                "aa_entropy_mean"] + [f"frac_{c}" for c in AA_CLASSES] + [f"interface_frac_{c}" for c in AA_CLASSES]
    if "halluc_recovery" in df.columns:
        agg_cols += ["halluc_recovery", "halluc_loss"]
    grouped = df.groupby("condition_index")
    summary = grouped[agg_cols].mean().add_suffix("_mean")
    summary = summary.join(grouped[agg_cols].sem().add_suffix("_sem"))
    summary["n_attempts"] = grouped.size()
    summary = summary.reindex(cond_indices)
    summary.insert(0, "condition_name", labels)
    summary.insert(1, "category", categories)
    summary.to_csv(f"{out_path}/ligand_bias_summary.csv")
    print(f"Wrote {out_path}/ligand_bias_summary.csv")

    apo_idx = cond_indices[labels.index("apo")]
    ligand_mask = [i for i, c in enumerate(categories) if c != "apo"]

    # ============================== Figure 1: engagement/confidence/fold ==============================
    has_halluc = "halluc_recovery" in df.columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.2))
    x = np.arange(len(cond_indices))
    rng = np.random.default_rng(0)

    def _bar(ax, col, title, ylabel, ref_apo=False):
        means = summary[f"{col}_mean"].to_numpy()
        sems = summary[f"{col}_sem"].to_numpy()
        ax.bar(x, means, yerr=sems, color=colors, width=0.75, alpha=0.75,
               error_kw=dict(elinewidth=1.0, ecolor=INK, capsize=2))
        # with only a handful of attempts per target, show every attempt, not just mean +/- SEM
        for xi, ci in zip(x, cond_indices):
            vals = df.loc[df.condition_index == ci, col].dropna().to_numpy()
            ax.scatter(xi + rng.uniform(-0.2, 0.2, len(vals)), vals, s=9, color=INK,
                       alpha=0.55, linewidths=0, zorder=3)
        if ref_apo:
            ax.axhline(means[cond_indices.index(apo_idx)], ls="--", lw=1.0, color=INK, zorder=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5)
        ax.set_xlim(-0.6, len(x) - 0.4)

    _bar(axes[0, 0], "contact_total", "Interface size", "# binder residues contacting target")
    _bar(axes[0, 1], "interface_pae", "Interface PAE", "mean interface PAE, normalized /32 (apo: n/a)")
    _bar(axes[0, 2], "plddt", "Binder confidence", "mean pLDDT", ref_apo=True)
    _bar(axes[0, 3], "iptm", "Binder-target confidence", "ipTM (apo: n/a)")
    _bar(axes[1, 0], "dssp_H", "Helix content", "fraction helix", ref_apo=True)
    _bar(axes[1, 1], "rmsd_to_apo", "Fold shift vs. apo", "CA RMSD to apo (Å) (apo: n/a)")
    _bar(axes[1, 2], "aa_entropy_mean", "Sequence uncertainty", "mean ProteinMPNN entropy (nats)")
    if has_halluc:
        _bar(axes[1, 3], "halluc_recovery", "Design self-consistency",
             "ProteinMPNN recovery (hallucination end)", ref_apo=True)
    else:
        axes[1, 3].axis("off")

    present = [c for c in CATEGORY_ORDER if c in categories]
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[c]) for c in present]
    fig.legend(handles, [CATEGORY_LABEL[c] for c in present],
               loc="lower center", ncol=len(present), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Boltz2 small-molecule / DNA biasing ({run_label}): per-target engagement "
                 f"(bars mean ± SEM, dots = attempts, n={int(summary['n_attempts'].max())} "
                 f"attempts/target)", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{out_path}/ligand_bias_engagement.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}/ligand_bias_engagement.png")

    # ============================== Figure 2: interface position heatmap ==============================
    per_cond_contact = np.stack([contact[df.loc[df.condition_index == ci, "row_index"].to_numpy()].mean(axis=0)
                                  if ci != apo_idx else np.full(num_aa, np.nan)
                                  for ci in cond_indices])
    fig, ax = plt.subplots(figsize=(13, 0.34 * len(cond_indices) + 1.6))
    plot_rows = [i for i in range(len(cond_indices)) if categories[i] != "apo"]
    im = ax.imshow(per_cond_contact[plot_rows], aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1,
                    interpolation="nearest")
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels([labels[i] for i in plot_rows], fontsize=8)
    for tick, i in zip(ax.get_yticklabels(), plot_rows):
        tick.set_color(colors[i])
    ax.set_xlabel(f"binder residue index (1-{num_aa})")
    ax.set_title(f"Where does the target dock along the binder chain? ({run_label})\n"
                 "(fraction of attempts in which each residue contacts the target, "
                 "best-of-N structure per attempt)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("contact frequency")
    fig.tight_layout()
    fig.savefig(f"{out_path}/ligand_bias_interface_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}/ligand_bias_interface_map.png")

    # ============================== Figure 3: AA-class enrichment at interface ==============================
    enrich = np.array([[summary.loc[ci, f"interface_frac_{c}_mean"] - summary.loc[ci, f"frac_{c}_mean"]
                         for c in AA_CLASSES] for ci in cond_indices])
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(ligand_mask) + 1.6))
    vmax = np.nanmax(np.abs(enrich[ligand_mask])) or 1.0
    im = ax.imshow(enrich[ligand_mask], aspect="auto", cmap=DIV_CMAP,
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), interpolation="nearest")
    ax.set_xticks(range(len(AA_CLASSES)))
    ax.set_xticklabels(AA_CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(ligand_mask)))
    ax.set_yticklabels([labels[i] for i in ligand_mask], fontsize=8)
    for tick, i in zip(ax.get_yticklabels(), ligand_mask):
        tick.set_color(colors[i])
    ax.set_title("Amino-acid class enrichment at the interface vs. the rest of the binder\n"
                 "(interface fraction minus whole-binder fraction, ProteinMPNN-read)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("interface − bulk fraction")
    fig.tight_layout()
    fig.savefig(f"{out_path}/ligand_bias_aa_enrichment.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}/ligand_bias_aa_enrichment.png")

    # ====== Figure 4 (hallucinated runs only): enrichment in the DESIGNED sequence itself ======
    # Hallucinated runs record the actual binder sequence, so the interface chemistry can be read
    # directly from it instead of via ProteinMPNN's opinion of the backbone (Figure 3).
    if "binder_sequence" in df.columns:
        class_letters = dict(zip(AA_CLASSES, ["AVLIM", "FWY", "KRH", "DE", "STNQC", "GP"]))
        per_attempt = []
        for _, r in df[df["category"] != "apo"].iterrows():
            seq = np.array(list(r["binder_sequence"]))
            iface = contact[int(r["row_index"])] > 0
            if iface.sum() < 3 or (~iface).sum() < 3:
                continue
            row = dict(condition_index=int(r["condition_index"]))
            for c, letters in class_letters.items():
                in_class = np.isin(seq, list(letters))
                row[c] = in_class[iface].mean() - in_class[~iface].mean()
            per_attempt.append(row)
        pa = pd.DataFrame(per_attempt)
        mean_by_target = pa.groupby("condition_index")[AA_CLASSES].mean().reindex(cond_indices)
        print(f"\nDesigned-sequence interface-minus-rest class fraction "
              f"(pooled over {len(pa)} attempts with an interface):")
        from scipy.stats import wilcoxon
        for c in AA_CLASSES:
            print(f"  {c:12s} {pa[c].mean():+.3f}  (Wilcoxon vs 0: p = {wilcoxon(pa[c]).pvalue:.2g})")
        fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(ligand_mask) + 1.6))
        data = mean_by_target.to_numpy()[ligand_mask]
        vmax = np.nanmax(np.abs(data)) or 1.0
        im = ax.imshow(data, aspect="auto", cmap=DIV_CMAP,
                       norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), interpolation="nearest")
        ax.set_xticks(range(len(AA_CLASSES)))
        ax.set_xticklabels(AA_CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(len(ligand_mask)))
        ax.set_yticklabels([labels[i] for i in ligand_mask], fontsize=8)
        for tick, i in zip(ax.get_yticklabels(), ligand_mask):
            tick.set_color(colors[i])
        ax.set_title(f"Designed-sequence class enrichment at the interface ({run_label})\n"
                     "(fraction at contacting residues minus fraction elsewhere, mean of attempts)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("interface − rest fraction")
        fig.tight_layout()
        fig.savefig(f"{out_path}/ligand_bias_aa_enrichment_designed.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}/ligand_bias_aa_enrichment_designed.png")


if __name__ == "__main__":
    main()
