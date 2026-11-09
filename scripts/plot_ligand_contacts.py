"""Contacts per ligand and attempt, with the mean and the attempts that pass the success filter.

For every ligand/DNA attempt of a ligand_bias run, counts the binder residues with any heavy atom
within --contact_cutoff (default 4.5 A) of the target in the final re-predicted complex, and plots
every attempt (dot), the per-target mean (black diamond), and marks the attempts that pass the
mosaic_scan.py success filter with a star. Which filter is shown (--filter auto picks the first
available):

  run     the `passed` column written by new ligand_bias runs (exact, scored on the fly)
  exact   <run>/mosaic_filter_exact.csv from scripts/mosaic_filter_ligand_bias.py --mode exact
  approx  a necessary-condition screen from results.csv (pLDDT > 0.8, hallucination-end recovery
          > 0.6, mean interface PAE < 5 A); ipSAE and the directional PAEs are not available
          for runs that predate the on-the-fly filter
  none    no marking

Writes <out_path>/ligand_contacts.png, ligand_contacts_per_attempt.csv and
ligand_contacts_summary.csv.

Usage:
    python3 scripts/plot_ligand_contacts.py --run_path runs/ligand_hallucination_pmpnninit
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_ligand_bias import (  # same directory
    _load_results_csv, _condition_order, _short_label,
    CATEGORY_COLOR, CATEGORY_LABEL, CATEGORY_ORDER, INK)
from analyze_ligand_burial import measure
from mosaic_filter_ligand_bias import approx as approx_screen


def pass_flags(run_path: str, df: pd.DataFrame, mode: str):
    """(boolean Series indexed like df, short description of the filter used)."""
    if mode in ("auto", "run") and "passed" in df.columns and (df["passed"] != "none").any():
        return df["passed"] == "True", "mosaic filter (scored on the fly)"
    if mode == "run":
        raise SystemExit(f"{run_path}/results.csv has no `passed` column")
    exact_csv = f"{run_path}/mosaic_filter_exact.csv"
    if mode in ("auto", "exact") and os.path.exists(exact_csv):
        ex = pd.read_csv(exact_csv).set_index("row_index")["exact_pass"]
        return df["row_index"].map(ex).fillna(False).astype(bool), "mosaic filter (exact rescoring)"
    if mode == "exact":
        raise SystemExit(f"{exact_csv} not found; run scripts/mosaic_filter_ligand_bias.py --mode exact")
    if mode in ("auto", "approx"):
        ap = approx_screen(df).set_index("row_index")["approx_pass"]
        return (df["row_index"].map(ap).fillna(False).astype(bool),
                "approximate mosaic screen (pLDDT > 0.8, recovery > 0.6, interface PAE < 5 Å)")
    return pd.Series(False, index=df.index), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", required=True)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--contact_cutoff", type=float, default=4.5)
    ap.add_argument("--filter", choices=["auto", "run", "exact", "approx", "none"], default="auto")
    args = ap.parse_args()
    out_path = args.out_path or args.run_path

    import json
    manifest = json.load(open(f"{args.run_path}/ligands.json"))
    with open(f"{args.run_path}/results.csv") as f:
        n_cols = len(f.readline().rstrip("\n").split(","))
    df = _load_results_csv(f"{args.run_path}/results.csv", n_cols=n_cols)
    df = df[(df["status"] == "ok") & (df["category"] != "apo")].reset_index(drop=True)

    df["contacts"] = [
        measure(f"{args.run_path}/predictions/{int(r.row_index):04d}_{r.condition_name}_a"
                f"{int(r.attempt_index)}.pdb", args.contact_cutoff, 6.0)["contact_residues"]
        for r in df.itertuples()]
    df["passes"], filter_label = pass_flags(args.run_path, df, args.filter)

    ordered = [c for c in _condition_order(manifest["conditions"]) if c["entry"] is not None]
    label_of = {c["condition_index"]: _short_label(c["entry"]["name"], c["entry"]["category"])
                for c in ordered}
    df["label"] = df["condition_index"].map(label_of)
    order = [label_of[c["condition_index"]] for c in ordered]
    cat_of = {label_of[c["condition_index"]]: c["entry"]["category"] for c in ordered}

    summary = df.groupby("label").agg(
        category=("category", "first"), n_attempts=("contacts", "size"),
        contacts_mean=("contacts", "mean"), contacts_std=("contacts", "std"),
        contacts_min=("contacts", "min"), contacts_max=("contacts", "max"),
        n_pass=("passes", "sum")).loc[order]
    df[["row_index", "condition_name", "category", "attempt_index", "contacts", "passes"]].to_csv(
        f"{out_path}/ligand_contacts_per_attempt.csv", index=False)
    summary.to_csv(f"{out_path}/ligand_contacts_summary.csv")

    n = len(order)
    y = np.arange(n)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.8, 0.34 * n + 2.6))
    for yi, label in zip(y, order):
        sub = df[df["label"] == label]
        color = CATEGORY_COLOR[cat_of[label]]
        jitter = yi + rng.uniform(-0.17, 0.17, len(sub))
        fail = ~sub["passes"].to_numpy()
        ax.scatter(sub["contacts"].to_numpy()[fail], jitter[fail], s=22, color=color, alpha=0.8,
                   linewidths=0, zorder=3)
        ax.scatter(sub["contacts"].to_numpy()[~fail], jitter[~fail], s=150, marker="*",
                   color=color, edgecolors=INK, linewidths=0.9, zorder=5)
        mean = sub["contacts"].mean()
        ax.scatter([mean], [yi], marker="D", s=34, color="black", zorder=6)
        ax.text(1.01, yi, f"{mean:.1f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=7.5, color=INK)
        if filter_label:
            ax.text(1.09, yi, f"{int(sub['passes'].sum())}/{len(sub)}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK)
    ax.text(1.01, -0.9, "mean", transform=ax.get_yaxis_transform(), fontsize=7.5, fontweight="bold")
    if filter_label:
        ax.text(1.09, -0.9, "pass", transform=ax.get_yaxis_transform(), fontsize=7.5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    for tick, label in zip(ax.get_yticklabels(), order):
        tick.set_color(CATEGORY_COLOR[cat_of[label]])
    ax.set_ylim(n - 0.4, -1.2)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"binder residues within {args.contact_cutoff:g} Å of the target")
    ax.grid(axis="x", color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], marker="o", ls="", color=CATEGORY_COLOR[c], markersize=6,
                      label=CATEGORY_LABEL[c]) for c in CATEGORY_ORDER[1:] if c in set(cat_of.values())]
    handles.append(Line2D([], [], marker="D", ls="", color="black", markersize=5, label="mean of the attempts"))
    if filter_label:
        handles.append(Line2D([], [], marker="*", ls="", color="white", markeredgecolor=INK,
                              markersize=11, label=f"passes: {filter_label}"))
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.45, -0.1 - 0.0 * n), ncol=2)
    ax.set_title(f"Contacts per ligand and attempt ({manifest['options'].get('init_mode', '?')} init)")
    fig.savefig(f"{out_path}/ligand_contacts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"{len(df)} attempts; filter shown: {filter_label or 'none'}"
          + (f" ({int(df['passes'].sum())} pass)" if filter_label else ""))
    print(summary[["category", "contacts_mean", "contacts_std", "n_pass"]].round(1).to_string())
    print(f"Wrote ligand_contacts.png/csv to {out_path}")


if __name__ == "__main__":
    main()
