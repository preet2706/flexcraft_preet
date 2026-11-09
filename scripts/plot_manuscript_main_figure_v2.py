"""Main manuscript figure (v2): model biases of ProteinMPNN-distribution initialisation.

Second draft. What changed against `plot_manuscript_main_figure.py` (v1, writes to
runs/manuscript_plots/v1/):

    b  KDE distributions instead of box plots, matching the house style of
       scripts/plot_mosaic_scan.py; the binder column is now mosaic_scan_n30 (13 targets x 30
       attempts, no distogram loss) instead of a weight-0 slice of the sweep; the init column is
       the SOFT ProteinMPNN distribution again, not its argmax
    c  grouped bars with one dot per target instead of a line plot over all weights; only two
       weight settings are shown (0 and the strongest available: 5, or 2 for AF2)
    f  an ABSOLUTE mean-DSSP-identity threshold instead of a relative fraction of each curve's
       own plateau; every threshold in THRESHOLD_SWEEP also gets its own standalone plot plus a
       small-multiples overview

Panels:

    a  schematic -- blueprint -> ProteinMPNN logits -> hallucination -> blueprint comparison
    b  pMPNN init vs monomer hallucination vs binder hallucination (Boltz-2)
    c  fold vs placement recovery, per model, per target, at two distogram loss weights
    d  one example: fold recovered, placement lost
    e  schematic -- averaging several blueprints, and the temperature that controls softness
    f  collapse-threshold temperature vs sheet bias, per structure model

Run from the repo root with the `flexcraft` env (matplotlib/scipy/biotite live there, not in
`flexcraft_new`):

    python3 scripts/plot_manuscript_main_figure_v2.py

Everything a panel plots is also written to runs/manuscript_plots/v2/panel_data/*.csv, and every
panel is written on its own to runs/manuscript_plots/v2/panels/ as PDF, SVG and PNG.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import biotite.structure.io.pdb as pdbio

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs")
OUT = os.path.join(RUNS, "manuscript_plots", "v2")

# Categorical slots 1-4 of the validated default palette (blue, orange, aqua, yellow), in that
# fixed order. The four-slot set clears every gate on the ADJACENT pairlist (lines and bars),
# which is what panels c and f are; slots 3 and 4 sit below 3:1 on a white surface, so both
# panels carry a per-model marker shape or an axis label as secondary encoding.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8982"
GRID = "#E6E6E6"
SURFACE = "#ffffff"
TARGET_GREY = "#b4b3ad"

MODELS = [  # (key, display label, short label, colour slot, marker)
    ("boltz2", "Boltz-2", "Boltz-2", SERIES[0], "o"),
    ("af2", "AlphaFold2", "AF2", SERIES[1], "s"),
    ("protenix_v1", "Protenix v1", "Px v1", SERIES[2], "^"),
    ("protenix_v2", "Protenix v2", "Px v2", SERIES[3], "D"),
]

# mosaic weight sweeps: base run (weights 0-2) + the extension run (3-5) where one exists
MOSAIC_RUNS = {
    "boltz2": ["mosaic/mosaic_distogram_sweep_joltz", "mosaic/mosaic_distogram_sweep_boltz_ext"],
    "af2": ["mosaic/mosaic_distogram_sweep_af2"],
    "protenix_v1": ["mosaic/mosaic_distogram_sweep_protenixV1",
                    "mosaic/mosaic_distogram_sweep_protenixV1_ext"],
    "protenix_v2": ["mosaic/mosaic_distogram_sweep_protenixV2",
                    "mosaic/mosaic_distogram_sweep_protenixV2_ext"],
}
TITRATION_RUNS = {
    "boltz2": "averaging/titration_boltz2",
    "af2": "averaging/titration_af2",
    "protenix_v1": "averaging/titration_protenixV1",
    "protenix_v2": "averaging/titration_protenixV2",
}

# panel b conditions (all Boltz-2). PANEL_B_LEGEND is the wording that goes in the legend, in the
# same order as panel_b_data(); the dict keys below stay as the internal condition names used in
# panel_data/panel_b_conditions.csv.
LANDSCAPE_DIRECT = "landscape/test_run"
LANDSCAPE_HAL = "landscape/test_run_hallucinate"
MOSAIC_BINDER = "mosaic/mosaic_scan_n30"
PANEL_B_LEGEND = ["Initial monomer prediction (n=1000)",
                  "Hallucinated monomer design (n=100)",
                  "Hallucinated binder design (n=390)"]

# panel d example: a *passing* Boltz-2 binder design whose fold matches the blueprint to 0.66 A
# but which sits on the opposite face of the target (epitope recall 0, centroid drift 39 A).
EXAMPLE = dict(run="mosaic/mosaic_distogram_sweep_joltz", target="ubiquitin", attempt=0, weight=1.0,
               blueprint="blueprints/ubiquitin_0_complex.pdb",
               design="attempts/ubiquitin_0_w1.0_stage_2.pdb")

# panel f: collapse criterion. T* is the largest averaging temperature at which the mean DSSP
# identity to the blueprint batch is still at or above COLLAPSE_THRESHOLD. 0.30 and 0.25 are the
# only cuts that resolve every (model, sheet bias) cell -- from 0.35 upwards the criterion is
# already violated at the sharpest temperature for some cells, which leaves holes in the curves.
# 0.30 separates the models more cleanly at high sheet bias than 0.25, so it is the one plotted
# in the main figure; the whole sweep is written out as its own figure.
COLLAPSE_THRESHOLD = 0.30
THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

PANEL_FORMATS = (("pdf", {}), ("svg", dict(transparent=True)), ("png", dict(dpi=400)))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.3,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",   # keep text editable in Illustrator / Inkscape
})


def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def panel_letter(fig, x, y, letter):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="top", ha="left", color=INK)


def bolden(ax, fs=None, lw=1.0):
    """Bold tick labels, axis labels and title, and a slightly heavier axis frame.

    matplotlib has no `xtick.labelweight` rcParam, so the tick labels have to be set one by one
    after they exist -- call this at the END of a panel, once the ticks are final.
    """
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")
        text.set_color(INK)
        if fs is not None:
            text.set_fontsize(fs)
    for text in (ax.xaxis.label, ax.yaxis.label, ax.title):
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_color(INK)
    ax.tick_params(width=lw, color=INK, labelcolor=INK)


def save_all(fig, path_stem):
    for ext, kw in PANEL_FORMATS:
        fig.savefig(f"{path_stem}.{ext}", **kw)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def _dedup(df):
    keys = [c for c in ("target", "attempt", "stage", "distogram_weight") if c in df]
    return df.drop_duplicates(subset=keys, keep="last")


def load_mosaic_sweep(model):
    """stage_2 (hardened final design) rows of one model's weight sweep, base + extension."""
    frames = []
    for rel in MOSAIC_RUNS[model]:
        path = os.path.join(RUNS, rel, "results.csv")
        if not os.path.isfile(path):
            print(f"  WARNING: missing {path} -- skipped")
            continue
        frames.append(_dedup(pd.read_csv(path)).query("stage == 'stage_2'"))
    return pd.concat(frames, ignore_index=True)


def load_titration(model):
    """`bias` mode rows: B independent blueprints under one soft L/H/E bias, averaged."""
    df = pd.read_csv(os.path.join(RUNS, TITRATION_RUNS[model], "results.csv"))
    b = df[df.avg_mode == "bias"].copy()
    b["e"] = b.e_bias.round(2)
    b["T"] = b.init_temperature.round(2)
    return b


def panel_b_data():
    """Three Boltz-2 settings. All three are unpaired -- different blueprints, different runs."""
    direct = pd.read_csv(os.path.join(RUNS, LANDSCAPE_DIRECT, "results.csv"))
    direct = direct[(direct.stage == "direct") & (direct.init == "soft")]
    hal = pd.read_csv(os.path.join(RUNS, LANDSCAPE_HAL, "results.csv"))
    hal = hal[hal.stage == "stage_1"]
    binder = pd.read_csv(os.path.join(RUNS, MOSAIC_BINDER, "results.csv"))
    binder = binder[binder.stage == "stage_2"]
    return [
        ("ProteinMPNN init, no hallucination", direct, SERIES[0]),
        ("Hallucination, monomer", hal, SERIES[1]),
        ("Hallucination, binder", binder, SERIES[2]),
    ]


def panel_c_data():
    """Per (model, weight, target) means of fold and placement recovery, over the whole sweep.

    Recoveries are stored as percentages, which is how every panel c variant plots them.
    """
    rows = []
    for key, label, short, color, marker in MODELS:
        df = load_mosaic_sweep(key)
        weights = sorted(df.distogram_weight.unique())
        for (w, target), t in df.groupby(["distogram_weight", "target"]):
            rows.append(dict(model=key, label=label, short=short, distogram_weight=float(w),
                             setting=("w_min" if w == weights[0] else
                                      "w_max" if w == weights[-1] else "mid"),
                             target=target, n=len(t),
                             fold_recovery_pct=100 * t.dssp_identity.mean(),
                             placement_recovery_pct=100 * t.epitope_recall_vs_blueprint.mean(),
                             placement_recovery_10a_pct=
                             100 * t.epitope_recall_vs_blueprint_10a.mean(),
                             iptm=t.iptm.mean(), pass_rate=t.passed.mean()))
    return pd.DataFrame(rows)


def panel_c_raw_data():
    """One row per finished design (all weights pooled), for the scatter variant."""
    frames = []
    for key, label, short, color, marker in MODELS:
        df = load_mosaic_sweep(key)
        frames.append(pd.DataFrame(dict(
            model=key, label=label, target=df.target.values,
            attempt=df.attempt.values, distogram_weight=df.distogram_weight.values,
            fold_recovery_pct=100 * df.dssp_identity.values,
            placement_recovery_pct=100 * df.epitope_recall_vs_blueprint.values)))
    return pd.concat(frames, ignore_index=True)


def _c_model_means(table, key):
    return (table[table.model == key]
            .groupby("distogram_weight")[["fold_recovery_pct", "placement_recovery_pct"]]
            .mean().reset_index().sort_values("distogram_weight"))


def collapse_temperature(temps, values, threshold):
    """Largest temperature, linearly interpolated, at which `values` still clears `threshold`.

    Scans upward from the sharpest temperature and stops at the first crossing, so a late
    recovery above the collapse does not resurrect the threshold. Returns (T*, status):
      'crossed'  -- a real measurement
      'censored' -- never fell below the threshold inside the grid, so T* = max(temps) is only a
                    lower bound (drawn with an open marker)
      'never'    -- already below the threshold at the sharpest temperature, T* undefined (NaN)
    """
    temps = np.asarray(temps, float)
    values = np.asarray(values, float)
    if values[0] < threshold:
        return np.nan, "never"
    for i in range(1, len(temps)):
        if values[i] < threshold:
            f = (values[i - 1] - threshold) / (values[i - 1] - values[i])
            return temps[i - 1] + f * (temps[i] - temps[i - 1]), "crossed"
    return temps[-1], "censored"


def panel_f_data(thresholds):
    """Threshold temperature vs sheet bias, per model, for every threshold in `thresholds`."""
    rows, reps = [], []
    for key, label, short, color, marker in MODELS:
        b = load_titration(key)
        pooled = b.pivot_table(index="e", columns="T", values="dssp_identity_mean", aggfunc="mean")
        per_rep = b.pivot_table(index=["e", "replicate"], columns="T",
                                values="dssp_identity_mean", aggfunc="mean")
        for threshold in thresholds:
            for e in pooled.index:
                t, status = collapse_temperature(pooled.columns.values, pooled.loc[e].values,
                                                 threshold)
                rows.append(dict(model=key, label=label, threshold=threshold, sheet_bias=e,
                                 plateau=pooled.loc[e, [0.1, 0.2, 0.3]].mean(),
                                 threshold_T=t, status=status, censored=status == "censored"))
                for r in sorted(b.replicate.unique()):
                    tr, sr = collapse_temperature(per_rep.columns.values,
                                                  per_rep.loc[(e, r)].values, threshold)
                    reps.append(dict(model=key, threshold=threshold, sheet_bias=e, replicate=r,
                                     threshold_T=tr, status=sr))
    return pd.DataFrame(rows), pd.DataFrame(reps)


def panel_f_summary(replicates):
    """Mean +- SEM of the per-replicate threshold temperature, per (model, threshold, sheet bias).

    A replicate whose curve is already below the cut at the sharpest temperature has no defined
    threshold and is left out, so `n` can be 2 instead of 3 in a few cells.
    """
    g = (replicates.dropna(subset=["threshold_T"])
         .groupby(["model", "threshold", "sheet_bias"]).threshold_T
         .agg(mean="mean", std="std", n="count").reset_index())
    g["sem"] = (g["std"] / np.sqrt(g["n"])).fillna(0.0)
    return g


def averaged_distributions(e_bias=0.5, replicate=0, n_residues=40):
    """One sharp and one soft averaged distribution from the Boltz-2 titration, as actually used."""
    run = os.path.join(RUNS, TITRATION_RUNS["boltz2"])
    dists = np.load(os.path.join(run, "distributions.npy"), mmap_mode="r")
    grid = json.load(open(os.path.join(run, "grid.json")))["grid"]

    def pick(temp):
        for g in grid:
            if (g["avg_mode"] == "bias" and abs(g["init_temperature"] - temp) < 1e-6
                    and abs(g["e_bias"] - e_bias) < 1e-6 and g["replicate"] == replicate):
                return np.asarray(dists[g["dist_index"]])[:n_residues]
        raise SystemExit(f"no bias grid point at T={temp}, E={e_bias}")

    return pick(0.2), pick(1.0)


def load_all():
    print("loading data ...")
    f_rows, f_reps = panel_f_data(sorted(set(THRESHOLD_SWEEP) | {COLLAPSE_THRESHOLD}))
    return dict(b=panel_b_data(), c=panel_c_data(), c_raw=panel_c_raw_data(), f=(f_rows, f_reps),
                e=averaged_distributions())


# --------------------------------------------------------------------------------------
# schematic drawing helpers (panels a and e)
# --------------------------------------------------------------------------------------

def box(ax, x, y, w, h, text, fc="#F4F6F9", ec=INK2, lw=0.8, fs=6.2, weight="normal",
        tc=INK, ls="-", rounding=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                linewidth=lw, linestyle=ls, facecolor=fc, edgecolor=ec,
                                mutation_aspect=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, zorder=3, linespacing=1.4)


def arrow(ax, p0, p1, color=INK2, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_width=0.22,head_length=0.42",
                                 mutation_scale=9, linewidth=lw, linestyle=ls, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4, joinstyle="round",
                                 capstyle="round"))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def mini_logits(ax, x, y, w, h, probs, vmax=None):
    """Draw a residue x amino-acid probability patch as a small heat strip."""
    ax.imshow(probs.T, extent=(x, x + w, y, y + h), aspect="auto", cmap="Blues",
              vmin=0.0, vmax=vmax if vmax is not None else probs.max(), zorder=3,
              interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK2, lw=0.6, zorder=4))


# --------------------------------------------------------------------------------------
# structure rendering (panel d)
# --------------------------------------------------------------------------------------

def ca_trace(path, chain):
    atoms = pdbio.PDBFile.read(path).get_structure(model=1)
    m = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
    return m.coord[np.argsort(m.res_id)].astype(float)


def kabsch(mobile_ref, fixed_ref, to_move):
    mc, fc = mobile_ref.mean(0), fixed_ref.mean(0)
    u, _, vt = np.linalg.svd((mobile_ref - mc).T @ (fixed_ref - fc))
    rot = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    fitted = (rot @ (mobile_ref - mc).T).T + fc
    rmsd = float(np.sqrt(((fitted - fixed_ref) ** 2).sum(1).mean()))
    return [(rot @ (x - mc).T).T + fc for x in to_move], rmsd


def _smooth(points, n=6):
    tck, _ = splprep(points.T, s=0, k=3)
    return np.array(splev(np.linspace(0, 1, len(points) * n), tck)).T


def view_basis(x_dir, cloud):
    """Right-handed view basis: first axis along `x_dir`, second along the widest remaining spread."""
    e1 = np.asarray(x_dir, float)
    e1 = e1 / np.linalg.norm(e1)
    centred = cloud - cloud.mean(0)
    perp = centred - np.outer(centred @ e1, e1)
    e2 = np.linalg.svd(perp, full_matrices=False)[2][0]
    e2 = e2 / np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)])


def _runs(points, basis, color, lw, run=10, overlap=2):
    projected = _smooth(points) @ basis.T
    out = []
    for i in range(0, len(projected) - 1, run):
        j = min(i + run + overlap, len(projected))
        out.append((projected[i:j, 2].mean(), projected[i:j, :2], color, lw))
    return out


def paint_chains(ax, chains, halo_lw=1.9, halo="white"):
    """Painter's algorithm: chains back to front, segments within a chain back to front.

    The halo is drawn once per chain rather than once per segment -- a per-segment halo erases the
    neighbouring segment's stroke and breaks each trace into dashes.
    """
    for chain in sorted(chains, key=lambda c: np.mean([r[0] for r in c])):
        xy = np.concatenate([r[1] for r in chain])
        lw = chain[0][3]
        ax.plot(xy[:, 0], xy[:, 1], color=halo, lw=lw + halo_lw, solid_capstyle="round",
                solid_joinstyle="round")
        for _, seg, color, lw in sorted(chain, key=lambda t: t[0]):
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, solid_capstyle="round",
                    solid_joinstyle="round")


# --------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------

def panel_a(ax):
    blank(ax)
    y0, h, w = 0.26, 0.34, 0.228
    xs = [0.008, 0.256, 0.504, 0.752]
    box(ax, xs[0], y0, w, h, "salad\nstructure diffusion\n\n$\\bf{blueprint}$ backbone",
        fc="#EEF4FC", ec=SERIES[0])
    box(ax, xs[1], y0, w, h, "ProteinMPNN\n\nper-residue\nlogit distribution", fc="#FDF0EA",
        ec=SERIES[1])
    box(ax, xs[2], y0, w, h, "hallucination\ninitialised from\nthat distribution", fc="#ECF7F2",
        ec=SERIES[2])
    box(ax, xs[3], y0, w, h, "success metrics\n+ comparison to\nthe blueprint", fc="#F4F4F2",
        ec=MUTED)
    for i in range(3):
        arrow(ax, (xs[i] + w + 0.003, y0 + h / 2), (xs[i + 1] - 0.003, y0 + h / 2))

    notes = ["one backbone\nper attempt", "a soft sequence,\nnot a sequence",
             "AF2 · Boltz-2\nProtenix v1 / v2", "C$\\alpha$-RMSD ·\nDSSP identity"]
    for x, text in zip(xs, notes):
        ax.text(x + w / 2, y0 - 0.055, text, ha="center", va="top", fontsize=5.8, color=MUTED,
                style="italic", linespacing=1.4)

    cw, ch = w, 0.26
    box(ax, xs[0], y0 + h + 0.13, cw, ch,
        "target structure (binder design)\nhelix / sheet bias  (L, H, E)\n"
        "hotspot / coldspot placement",
        fc=SURFACE, ec=MUTED, lw=0.7, fs=5.9, ls=(0, (2.4, 1.8)), tc=INK2)
    arrow(ax, (xs[0] + cw / 2, y0 + h + 0.125), (xs[0] + cw / 2, y0 + h + 0.008), color=MUTED,
          lw=0.8, ls=(0, (2.4, 1.8)))
    ax.text(xs[0], y0 + h + 0.405, "optional conditioning", fontsize=5.9, color=MUTED, va="bottom",
            style="italic")


def kde_curve(values, xlim, scale=1.0):
    """Silverman-bandwidth KDE on a fixed grid, matching scripts/plot_mosaic_scan.py's `_kde`.

    Returns (grid, density) rather than drawing, so the caller can normalise a whole sub-panel's
    curves by one common factor before plotting -- that keeps the three settings comparable in
    height while giving the y axis round numbers instead of raw 1/unit densities.
    """
    v = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float) * scale
    v = v[np.isfinite(v)]
    if v.size < 3 or np.isclose(v.std(), 0.0):
        return None, None
    lo, hi = xlim
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 512)
    return grid, gaussian_kde(v, bw_method="silverman")(grid)


PANEL_B_METRICS = [("dssp_identity", "Fold recovery [%]", (30.0, 100.0), 100.0),
                   ("rmsd_to_blueprint", "C$\\alpha$-RMSD to blueprint [Å]", (0.0, 20.0), 1.0),
                   # pLDDT is a confidence score, not a percentage: 0-100, no unit in brackets
                   ("plddt", "pLDDT", (40.0, 100.0), 100.0),
                   ("recovery", "ProteinMPNN recovery [%]", (0.0, 100.0), 100.0)]


def panel_b(fig, rect, data, gap_x=0.075, gap_y=0.115, legend_dy=0.030, legend_fs=6.2,
            fs=6.3, lw=1.4, ylabel_every=False, compact_labels=False, layout="row",
            legend_ncol=1, minimal_yticks=None, legend_top=False):
    """KDE small multiples -- one metric each, three settings, Boltz-2 throughout.

    The three fraction-valued metrics are shown as percentages so every axis can carry a unit in
    square brackets; Cal-RMSD is already in Angstrom. Densities are normalised per sub-panel
    (one common factor for all three curves), so the y axis is 'Density [a.u.]'.

    Every sub-panel carries the SAME normalised density axis, so it is drawn once: only the
    left-most sub-panel of each row keeps the label, the ticks and the left spine. `layout`
    picks "row" (1x4, one shared axis for the whole block -- the most compact) or "grid" (2x2,
    one per row), since a 1x4 row needs about 5 in of width to stay legible.
    """
    metrics = list(PANEL_B_METRICS)
    if compact_labels:
        # In the assembled figure each sub-panel is ~0.75 in wide and the full labels run off the
        # page; the standalone panel keeps the long form.
        short = ["Fold recovery [%]", "C$\\alpha$-RMSD [Å]", "pLDDT", "pMPNN recovery [%]"]
        metrics = [(m[0], s, m[2], m[3]) for m, s in zip(metrics, short)]
    left, bottom, width, height = rect
    ncol = len(metrics) if layout == "row" else 2
    nrow = 1 if layout == "row" else 2
    cw = (width - gap_x * (ncol - 1)) / ncol
    chh = (height - gap_y * (nrow - 1)) / nrow
    axes = []
    for i, (col, label, xlim, scale) in enumerate(metrics):
        ax = fig.add_axes([left + (i % ncol) * (cw + gap_x),
                           bottom + (nrow - 1 - i // ncol) * (chh + gap_y), cw, chh])
        axes.append(ax)
        curves = [(kde_curve(df[col], xlim, scale), color) for _, df, color in data]
        peak = max((d.max() for (g, d), _ in curves if d is not None), default=1.0)
        for (grid, dens), color in curves:
            if dens is None:
                continue
            dens = dens / peak
            ax.plot(grid, dens, color=color, lw=lw, zorder=3)
            ax.fill_between(grid, dens, color=color, alpha=0.15, lw=0, zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.12)
        # the density axis is identical in all four sub-panels, so it is drawn once per row
        first_in_row = ylabel_every or i % ncol == 0
        if first_in_row:
            # at main-figure size a sub-panel is <0.5 in tall and three y ticks collide
            minimal = compact_labels if minimal_yticks is None else minimal_yticks
            ax.set_yticks([0, 1.0] if minimal else [0, 0.5, 1.0])
            ax.set_yticklabels(["0", "1"] if minimal else ["0", "0.5", "1"])
            ax.set_ylabel("Density [a.u.]", fontsize=fs, labelpad=1.5)
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        ax.set_xlabel(label, fontsize=fs, labelpad=1.5)
        tidy(ax, grid=None)
        bolden(ax, fs=fs - 0.3)
    handles = [plt.Line2D([], [], color=c, lw=2.0, label=lab) for lab, (_, _, c) in
               zip(PANEL_B_LEGEND, data)]
    # legend_dy is the gap between the legend and the nearest edge of the plot block
    loc, anchor_y = (("lower center", bottom + height + legend_dy) if legend_top
                     else ("upper center", bottom - legend_dy))
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=(left + width / 2, anchor_y),
               ncol=legend_ncol, frameon=False, handlelength=1.4, handletextpad=0.6,
               labelspacing=0.5, columnspacing=1.6, borderpad=0,
               prop=dict(size=legend_fs, weight="bold"))
    return axes


def panel_b_ridge(fig, rect, data, fs=8.0, legend_fs=7.0, legend_dy=0.055, rmsd_frac=0.27,
                  gap=0.045, lw=1.4):
    """Alternative panel b: the three percentage metrics stacked in ONE axes.

    Fold recovery, pLDDT and ProteinMPNN recovery are all 0-100 on the same scale, so they do not
    need three separate x axes -- they become three offset baselines of one ridgeline, which
    replaces the repeated 'Density [a.u.]' axis with a metric name per row. Ca-RMSD is in
    Angstrom and cannot share that axis, so it keeps a narrow panel of its own. Two axes instead
    of four, one x label each.
    """
    pct = [m for m in PANEL_B_METRICS if m[0] != "rmsd_to_blueprint"]
    rmsd = next(m for m in PANEL_B_METRICS if m[0] == "rmsd_to_blueprint")
    left, bottom, width, height = rect
    w_rmsd = width * rmsd_frac
    w_main = width - w_rmsd - gap
    ax = fig.add_axes([left, bottom, w_main, height])
    ax_r = fig.add_axes([left + w_main + gap, bottom, w_rmsd, height])

    xlim = (0.0, 100.0)
    row_h = 1.0          # spacing between baselines
    amp = 0.82           # peak height of a curve, in row_h units
    for row, (col, label, _ignore, scale) in enumerate(pct[::-1]):
        base = row * row_h
        curves = [(kde_curve(df[col], xlim, scale), color) for _, df, color in data]
        peak = max((d.max() for (g, d), _ in curves if d is not None), default=1.0)
        ax.axhline(base, color=GRID, lw=0.7, zorder=1)
        for (grid, dens), color in curves:
            if dens is None:
                continue
            y = base + amp * dens / peak
            ax.fill_between(grid, base, y, color=color, alpha=0.18, lw=0, zorder=2 + row)
            ax.plot(grid, y, color=color, lw=lw, zorder=6 + row)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.06, (len(pct) - 1) * row_h + amp + 0.16)
    ax.set_yticks([r * row_h for r in range(len(pct))])
    ax.set_yticklabels([m[1].replace(" [%]", "").replace("ProteinMPNN", "pMPNN")
                        for m in pct[::-1]], fontsize=fs)
    ax.set_xlabel("Recovery / confidence [%]", fontsize=fs, labelpad=1.5)
    tidy(ax, grid=None)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    bolden(ax, fs=fs - 0.5)

    curves = [(kde_curve(df[rmsd[0]], rmsd[2], rmsd[3]), color) for _, df, color in data]
    peak = max((d.max() for (g, d), _ in curves if d is not None), default=1.0)
    for (grid, dens), color in curves:
        if dens is None:
            continue
        ax_r.fill_between(grid, dens / peak, color=color, alpha=0.18, lw=0, zorder=2)
        ax_r.plot(grid, dens / peak, color=color, lw=lw, zorder=3)
    ax_r.set_xlim(*rmsd[2])
    ax_r.set_ylim(0, 1.12)
    ax_r.set_yticks([])
    ax_r.set_xlabel("C$\\alpha$-RMSD [Å]", fontsize=fs, labelpad=1.5)
    tidy(ax_r, grid=None)
    ax_r.spines["left"].set_visible(False)
    bolden(ax_r, fs=fs - 0.5)

    handles = [plt.Line2D([], [], color=c, lw=2.0, label=lab) for lab, (_, _, c) in
               zip(PANEL_B_LEGEND, data)]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(left + width / 2, bottom - legend_dy), ncol=1, frameon=False,
               handlelength=1.4, handletextpad=0.6, labelspacing=0.5, borderpad=0,
               prop=dict(size=legend_fs, weight="bold"))
    return [ax, ax_r]


def _c_legend(fig, x, y, ncol, fs):
    handles = [plt.Line2D([], [], color=c, lw=1.5, marker=m, markersize=4, markerfacecolor=c,
                          markeredgecolor="white", label=lab) for _, lab, _, c, m in MODELS]
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x, y), ncol=ncol,
                      frameon=False, handlelength=1.5, handletextpad=0.5, columnspacing=1.4,
                      labelspacing=0.4, borderpad=0, prop=dict(size=fs, weight="bold"))


def panel_c(fig, rect, table, legend=True, fs=6.3, legend_fs=6.3, legend_ncol=4, lw=1.5,
            top_frac=0.26, strip_gap=0.022, legend_dy=0.012, legend_x=None):
    """Two strips over a shared weight axis: a short one for fold, a tall one for placement.

    Fold recovery is the quantity that does NOT move -- it stays in a narrow band whatever the
    model, the weight or the target -- so it gets a quarter of the height and a zoomed axis, and
    placement gets the rest. Two stacked single-axis strips rather than one squashed axis or a
    dual axis: on a common 0-100 scale the placement lines would be flattened into the bottom
    quarter and the per-model differences would be unreadable.
    """
    left, bottom, width, height = rect
    h_top = height * top_frac
    h_bot = height - h_top - strip_gap
    ax_place = fig.add_axes([left, bottom, width, h_bot])
    ax_fold = fig.add_axes([left + 0, bottom + h_bot + strip_gap, width, h_top])

    for key, label, short, color, marker in MODELS:
        m = _c_model_means(table, key)
        for ax, col in ((ax_fold, "fold_recovery_pct"), (ax_place, "placement_recovery_pct")):
            ax.plot(m.distogram_weight, m[col], color=color, lw=lw, marker=marker,
                    markersize=4.0, markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=4, clip_on=False)

    for ax in (ax_fold, ax_place):
        ax.set_xlim(-0.15, 5.15)
        ax.set_xticks(range(6))
        tidy(ax, grid=None)
    ax_fold.set_xticklabels([])
    ax_fold.set_ylim(86, 100)
    ax_fold.set_yticks([90, 100])
    ax_fold.set_ylabel("Fold\nrecovery [%]", fontsize=fs, labelpad=1.5, linespacing=1.3)
    ax_place.set_ylim(0, 30)
    ax_place.set_yticks([0, 10, 20, 30])
    ax_place.set_ylabel("Placement recovery [%]", fontsize=fs, labelpad=1.5)
    ax_place.set_xlabel("Distogram loss weight", fontsize=fs, labelpad=1.5)
    bolden(ax_fold, fs=fs - 0.3)
    bolden(ax_place, fs=fs - 0.3)
    if legend:
        _c_legend(fig, left + width / 2 if legend_x is None else legend_x,
                  bottom + height + legend_dy, legend_ncol, legend_fs)
    return [ax_fold, ax_place]


def panel_c_dumbbell(fig, rect, table, fs=7.5, legend_fs=7.0, fold_frac=0.26, gap=0.035):
    """Alternative layout: what the distogram loss moves, per model, in one row each.

    Left (wide): placement recovery, one row per model, an arrow from the no-loss weight to the
    strongest weight with every target drawn as a faint dot at both ends. Right (narrow): fold
    recovery, the same rows, shown once -- mean plus the across-target range.
    """
    left, bottom, width, height = rect
    w_fold = width * fold_frac
    w_place = width - w_fold - gap
    ax_p = fig.add_axes([left, bottom, w_place, height])
    ax_f = fig.add_axes([left + w_place + gap, bottom, w_fold, height])
    rng = np.random.default_rng(1)

    for row, (key, label, short, color, marker) in enumerate(MODELS):
        y = len(MODELS) - 1 - row
        sub = table[table.model == key]
        w0, wmax = sub.distogram_weight.min(), sub.distogram_weight.max()
        lo = sub[sub.distogram_weight == w0]
        hi = sub[sub.distogram_weight == wmax]
        for frame, dy, filled in ((lo, 0.22, False), (hi, -0.22, True)):
            ax_p.plot(frame.placement_recovery_pct, y + dy + rng.uniform(-0.07, 0.07, len(frame)),
                      ".", color=color, markersize=3.0, alpha=0.40, zorder=2)
        x0, x1 = lo.placement_recovery_pct.mean(), hi.placement_recovery_pct.mean()
        # an arrowhead on a 0.3-point shift (AF2) renders as a blob -- draw the two dots only
        if abs(x1 - x0) > 1.5:
            ax_p.annotate("", xy=(x1, y), xytext=(x0, y),
                          arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                          color=color, lw=2.0, shrinkA=0, shrinkB=0), zorder=3)
        ax_p.plot([x0], [y], "o", color=color, markersize=5.5, markerfacecolor="white",
                  markeredgecolor=color, markeredgewidth=1.4, zorder=4)
        ax_p.plot([x1], [y], "o", color=color, markersize=5.0, zorder=5)
        ax_p.annotate(f"w = {wmax:g}", (max(x0, x1), y), textcoords="offset points",
                      xytext=(8, 0), ha="left", va="center", fontsize=fs - 1.3, color=color,
                      fontweight="bold")
        fold = sub.groupby("distogram_weight").fold_recovery_pct.mean()
        ax_f.plot([sub.fold_recovery_pct.min(), sub.fold_recovery_pct.max()], [y, y],
                  color=color, lw=1.2, alpha=0.35, solid_capstyle="round", zorder=2)
        ax_f.plot([fold.mean()], [y], marker, color=color, markersize=5.0,
                  markeredgecolor="white", markeredgewidth=0.7, zorder=4)

    ax_p.set_ylim(-0.6, len(MODELS) - 0.4)
    ax_p.set_yticks(range(len(MODELS)))
    ax_p.set_yticklabels([m[1] for m in MODELS][::-1], fontsize=fs)
    ax_p.set_xlim(0, 62)
    ax_p.set_xlabel("Placement recovery [%]", fontsize=fs, labelpad=1.5)
    ax_p.set_title("open → filled: w = 0 → strongest weight · dots = targets",
                   fontsize=fs - 1.3, pad=4, color=INK2, loc="left")
    ax_f.set_ylim(*ax_p.get_ylim())
    ax_f.set_yticks([])
    ax_f.spines["left"].set_visible(False)
    ax_f.set_xlim(75, 101)
    ax_f.set_xticks([80, 100])
    ax_f.set_xlabel("Fold recovery [%]", fontsize=fs, labelpad=1.5)
    ax_f.set_title("mean · target range", fontsize=fs - 1.3, pad=4, color=INK2)
    for ax in (ax_p, ax_f):
        tidy(ax, grid=None)
        bolden(ax, fs=fs - 0.5)
    return [ax_p, ax_f]


def _c_endpoint_rows(raw):
    """Per-design rows at each model's lowest and highest distogram weight."""
    out = []
    for key, label, short, color, marker in MODELS:
        g = raw[raw.model == key]
        lo, hi = g.distogram_weight.min(), g.distogram_weight.max()
        out.append(g[g.distogram_weight.isin([lo, hi])].assign(
            endpoint=np.where(g[g.distogram_weight.isin([lo, hi])].distogram_weight == lo,
                              "w_min", "w_max")))
    return pd.concat(out, ignore_index=True)


# wrapped onto two lines: rotated 90 deg, one long line would need ~1.1 in of vertical space,
# while the two stacked lines need ~0.6 in and still fit the width of one bar slot
DISTOGRAM_SUFFIX = "+ Distogram\nplacement loss"


# the three cases panel c distinguishes, in the figure-wide palette: blue / orange / yellow
# (categorical slots 1, 2, 4). NB: orange vs yellow is the weakest pair of the palette (normal-
# vision dE 13.7, just under the 15 floor), so position and the legend carry identity as well.
CASE_COLORS = {"fold": SERIES[0], "placement": SERIES[1], "placement_loss": SERIES[3]}
CASE_LABELS = {"fold": "Fold recovery", "placement": "Placement recovery",
               "placement_loss": "Placement recovery + Distogram placement loss"}


def _c_bar_stats(v):
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def _c_legend_ax(ax, handles, fs, ncol, y=1.005):
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, handlelength=1.3, handleheight=0.95, handletextpad=0.5,
              columnspacing=1.4, borderpad=0, prop=dict(size=fs, weight="bold"))


def panel_c_box(fig, rect, raw, fs=8.0, legend_fs=7.0, fold_once=False, legend=True,
                model_label_dy=0.09, short_labels=False, tick_fs=None, legend_ncol=3):
    """Bars (mean +- SEM) grouped by model: fold and placement recovery at the two weight ends.

    One bar per (model, metric, weight) over that cell's 55 designs. Colour is the model, fill is
    the metric (solid = fold, outline = placement), and the distogram-loss case is a hatch --
    a colour- and position-independent encoding that repeats across all four groups. All three
    keys live in the legend and nothing is annotated in the plot; the x axis has no tick labels and
    the model names are black.

    `fold_once` collapses the two fold bars per model into one (both weights pooled, n = 110),
    giving 12 bars instead of 16 -- fold barely moves with the weight.
    """
    ax = fig.add_axes(rect)
    rows = _c_endpoint_rows(raw)
    specs = ([("fold_recovery_pct", None, True),
              ("placement_recovery_pct", "w_min", False),
              ("placement_recovery_pct", "w_max", False)] if fold_once else
             [("fold_recovery_pct", "w_min", True), ("fold_recovery_pct", "w_max", True),
              ("placement_recovery_pct", "w_min", False),
              ("placement_recovery_pct", "w_max", False)])
    # a wider gap between the fold block and the placement block than between the two weights;
    # both stay clearly smaller than the gap BETWEEN groups
    offsets = ([-0.26, 0.0, 0.26] if fold_once else [-0.27, -0.12, 0.12, 0.27])
    width = 0.19 if fold_once else 0.13
    plt.rcParams["hatch.linewidth"] = 0.55

    for i, (key, label, short, color, marker) in enumerate(MODELS):
        g = rows[rows.model == key]
        for (col, endpoint, filled), off in zip(specs, offsets):
            v = (g if endpoint is None else g[g.endpoint == endpoint])[col].dropna().values
            x = i + off
            mean, sem = _c_bar_stats(v)
            ax.bar(x, mean, width=width, zorder=3,
                   color=color if filled else "white", alpha=0.85 if filled else 1.0,
                   edgecolor="white" if filled else color, linewidth=0.0 if filled else 1.2,
                   hatch="////" if endpoint == "w_max" else None)
            ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor=INK, elinewidth=1.0,
                        capsize=width * 14, capthick=1.0, zorder=5)
        ax.text(i, -model_label_dy, short if short_labels else label,
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fs,
                fontweight="bold", color=INK)

    ax.set_xticks([])
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Recovery of the blueprint [%]", fontsize=fs, labelpad=1.5)
    tidy(ax, grid=None)
    bolden(ax, fs=tick_fs or fs - 1.2)
    ax.yaxis.label.set_fontsize(fs)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=0.85, edgecolor="none",
                                 label="Fold recovery"),
                   plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, linewidth=1.2,
                                 label="Placement recovery"),
                   plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, linewidth=1.2,
                                 hatch="////", label="+ Distogram placement loss")]
        _c_legend_ax(ax, handles, legend_fs, legend_ncol)
    return ax


def panel_c_case(fig, rect, raw, fs=8.0, legend_fs=7.0, legend=True, model_label_dy=0.09,
                 short_labels=False, tick_fs=None, legend_ncol=3):
    """Alternative: bars coloured by CASE rather than by model.

    Three bars per model -- fold, placement, placement + distogram loss -- in the same three
    colours in every group, so the legend is the whole key and the models sit on the x axis in
    black. 12 bars. Fold is pooled over the two weight endpoints (n = 110): it barely moves.
    """
    ax = fig.add_axes(rect)
    rows = _c_endpoint_rows(raw)
    cases = [("fold", "fold_recovery_pct", None), ("placement", "placement_recovery_pct", "w_min"),
             ("placement_loss", "placement_recovery_pct", "w_max")]
    offsets, width = [-0.26, 0.0, 0.26], 0.2
    for i, (key, label, short, _c, _m) in enumerate(MODELS):
        g = rows[rows.model == key]
        for (case, col, endpoint), off in zip(cases, offsets):
            v = (g if endpoint is None else g[g.endpoint == endpoint])[col].dropna().values
            mean, sem = _c_bar_stats(v)
            ax.bar(i + off, mean, width=width, color=CASE_COLORS[case], zorder=3)
            ax.errorbar(i + off, mean, yerr=sem, fmt="none", ecolor=INK, elinewidth=1.0,
                        capsize=width * 14, capthick=1.0, zorder=5)
        ax.text(i, -model_label_dy, short if short_labels else label,
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fs,
                fontweight="bold", color=INK)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Recovery of the blueprint [%]", fontsize=fs, labelpad=1.5)
    tidy(ax, grid=None)
    bolden(ax, fs=tick_fs or fs - 1.2)
    ax.yaxis.label.set_fontsize(fs)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CASE_COLORS[c], edgecolor="none",
                                 label=CASE_LABELS[c]) for c, *_ in cases]
        _c_legend_ax(ax, handles, legend_fs, legend_ncol)
    return ax


# Worked examples for the distogram loss: the SAME blueprint designed at two weights.
# The weight sweeps were run as two separate jobs (weights 0-2 and 3-5) and a separate job
# re-diffuses the blueprint, so a given (target, attempt) has a DIFFERENT blueprint backbone in
# the two runs (7-11 A apart) even though it docks to the same site (centroids within 2 A).
# A matched w = 0 vs w = 5 pair therefore does not exist; each example below stays inside one run.
EXAMPLES_C = {
    "w0_w2": dict(run="mosaic/mosaic_distogram_sweep_joltz", target="bromodomain", attempt=3,
                  w_lo=0.0, w_hi=2.0),
    "w3_w5": dict(run="mosaic/mosaic_distogram_sweep_boltz_ext", target="proteinG", attempt=0,
                  w_lo=3.0, w_hi=5.0),
}


def panel_c_example(fig, rect, spec, raw, fs=8.0, gap=0.030, note_fs=None):
    """Two views of one blueprint designed at two distogram weights, in a shared orientation.

    Both design complexes are superimposed on the blueprint complex through the TARGET chain, so
    the grey target and the blue blueprint binder are in the same place in both views and only
    the orange design moves. One view basis for both, so the two are directly comparable.
    """
    run = os.path.join(RUNS, spec["run"])
    stem = f"{spec['target']}_{spec['attempt']}"
    bp = os.path.join(run, "blueprints", f"{stem}_complex.pdb")
    target_bp, binder_bp = ca_trace(bp, "B"), ca_trace(bp, "A")

    fitted, stats = [], []
    for w in (spec["w_lo"], spec["w_hi"]):
        des = os.path.join(run, "attempts", f"{stem}_w{w:.1f}_stage_2.pdb")
        target_de, binder_de = ca_trace(des, "B"), ca_trace(des, "A")
        (_tf, binder_fit), _ = kabsch(target_de, target_bp, [target_de, binder_de])
        fitted.append(binder_fit)
        row = raw[(raw.model == "boltz2") & (raw.target == spec["target"]) &
                  (raw.attempt == spec["attempt"]) & (raw.distogram_weight == w)]
        stats.append(dict(weight=w,
                          drift=float(np.linalg.norm(binder_fit.mean(0) - binder_bp.mean(0))),
                          placement=float(row.placement_recovery_pct.iloc[0]),
                          fold=float(row.fold_recovery_pct.iloc[0])))

    cloud = np.concatenate([target_bp, binder_bp] + fitted)
    basis = view_basis(fitted[0].mean(0) - binder_bp.mean(0), cloud)
    p = cloud @ basis.T
    left, bottom, width, height = rect
    cw = (width - gap) / 2
    axes = []
    titles = (["no distogram loss", "+ Distogram placement loss"] if spec["w_lo"] == 0 else
              ["weaker distogram loss", "stronger distogram loss"])
    for i, (binder_fit, st, title) in enumerate(zip(fitted, stats, titles)):
        ax = fig.add_axes([left + i * (cw + gap), bottom, cw, height])
        axes.append(ax)
        paint_chains(ax, [_runs(target_bp, basis, TARGET_GREY, 2.1),
                          _runs(binder_bp, basis, SERIES[0], 2.1),
                          _runs(binder_fit, basis, SERIES[1], 2.1)])
        ax.set_xlim(p[:, 0].min() - 4, p[:, 0].max() + 4)
        ax.set_ylim(p[:, 1].min() - 4, p[:, 1].max() + 4)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{title}   (w = {st['weight']:g})", fontsize=fs, pad=3, fontweight="bold",
                     color=INK)
        ax.text(0.5, -0.02, f"{st['drift']:.0f} Å centroid shift\n"
                            f"{st['placement']:.0f} % placement recovery",
                transform=ax.transAxes, ha="center", va="top", linespacing=1.35,
                fontsize=note_fs or fs - 1.2, color=INK2, fontweight="bold")
    for x, text, color in ((0.16, "target", "#8f8e88"), (0.44, "salad blueprint", SERIES[0]),
                           (0.80, "hallucinated design", SERIES[1])):
        fig.text(left + x * width, bottom - 0.125, text, ha="center", va="top",
                 fontsize=note_fs or fs - 1.2, color=color, fontweight="bold")
    return axes, stats


def panel_c_scatter(fig, rect, raw, fs=8.0, legend_fs=7.0, seed=0, y_floor=None):
    """Alternative layout: every design as one point, fold against placement, weights pooled.

    Both axes are percentages of the same blueprint, so they share a scale conceptually but not
    a range -- the axes are fitted to the data rather than forced to 0-100, which is what makes
    the shape visible: the cloud hugs the top edge (fold is recovered) and spreads along the
    bottom-left (placement is not). Points are jittered by +-1 % for display only, because both
    axes are k/n over a short DSSP string / a small epitope and hundreds of designs land on
    exactly the same pair.
    """
    ax = fig.add_axes(rect)
    rng = np.random.default_rng(seed)
    x_all = raw.placement_recovery_pct.values
    y_all = raw.fold_recovery_pct.values
    for key, label, short, color, marker in MODELS:
        sub = raw[raw.model == key]
        jx = np.clip(sub.placement_recovery_pct + rng.uniform(-1, 1, len(sub)), 0, 100)
        jy = np.clip(sub.fold_recovery_pct + rng.uniform(-1, 1, len(sub)), 0, 100)
        ax.scatter(jx, jy, s=7, color=color, marker=marker, linewidths=0, alpha=0.22, zorder=2)
    for key, label, short, color, marker in MODELS:
        sub = raw[raw.model == key]
        mx, my = sub.placement_recovery_pct.mean(), sub.fold_recovery_pct.mean()
        # white underlay first: the Boltz-2 and Protenix v2 means are 2 % apart and would
        # otherwise merge into one blob
        ax.scatter(mx, my, s=150, color="white", marker=marker, linewidths=0, zorder=4)
        ax.scatter(mx, my, s=70, color=color, marker=marker, edgecolor=INK, linewidths=1.2,
                   zorder=5)
    pad_x = 0.03 * (x_all.max() - x_all.min())
    y_lo = y_all.min() if y_floor is None else y_floor
    pad_y = 0.05 * (y_all.max() - y_lo)
    ax.set_xlim(x_all.min() - pad_x, x_all.max() + pad_x)
    ax.set_ylim(y_lo - pad_y, min(100 + pad_y, y_all.max() + pad_y))
    hidden = 0 if y_floor is None else int((y_all < y_floor).sum())
    ax.set_xlabel("Placement recovery [%]", fontsize=fs, labelpad=1.5)
    ax.set_ylabel("Fold recovery [%]", fontsize=fs, labelpad=1.5)
    tidy(ax, grid=None)
    bolden(ax, fs=fs - 0.5)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=5,
                          markeredgecolor=INK, markeredgewidth=1.0, label=lab)
               for _, lab, _, c, m in MODELS]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=False,
                    handletextpad=0.4, labelspacing=0.4, borderpad=0,
                    prop=dict(size=legend_fs, weight="bold"))
    ax.add_artist(leg)
    note = [f"small dot = one design (n = {len(raw)})", "large marker = model mean"]
    if hidden:
        note.append(f"{hidden} designs below {y_floor:g} % not shown")
    ax.text(0.985, 0.30, "\n".join(note), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=legend_fs - 0.6, color=INK2, linespacing=1.4)
    return ax


def panel_c_heatmap(fig, rect, table, fs=7.5, gap=0.035, fold_frac=0.22):
    """Alternative layout: the whole sweep as a grid -- every model x every weight, no lines.

    Placement recovery is the only thing on a colour ramp (one sequential hue, light = low);
    fold recovery is a single numeric column on the right, because it does not vary enough to
    deserve a second ramp.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    ramp = LinearSegmentedColormap.from_list(
        "blueseq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    left, bottom, width, height = rect
    w_fold = width * fold_frac
    w_grid = width - w_fold - gap
    ax = fig.add_axes([left, bottom, w_grid, height])
    ax_f = fig.add_axes([left + w_grid + gap, bottom, w_fold, height])

    weights = sorted(table.distogram_weight.unique())
    grid = np.full((len(MODELS), len(weights)), np.nan)
    for row, (key, *_rest) in enumerate(MODELS):
        m = _c_model_means(table, key).set_index("distogram_weight")
        for col, w in enumerate(weights):
            if w in m.index:
                grid[row, col] = m.loc[w, "placement_recovery_pct"]
    norm = Normalize(0, np.nanmax(grid))
    ramp = ramp.copy()
    ramp.set_bad("#F0EFEC")   # weights AF2 has no extension run for; a cell, not a hole
    ax.imshow(np.ma.masked_invalid(grid), cmap=ramp, norm=norm, aspect="auto",
              interpolation="nearest")
    for row in range(len(MODELS)):
        for col in range(len(weights)):
            v = grid[row, col]
            if np.isnan(v):
                ax.text(col, row, "–", ha="center", va="center", fontsize=fs - 0.5, color=MUTED)
                continue
            ax.text(col, row, f"{v:.0f}", ha="center", va="center", fontsize=fs - 0.5,
                    fontweight="bold", color="white" if norm(v) > 0.55 else INK)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"{w:g}" for w in weights], fontsize=fs)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[1] for m in MODELS], fontsize=fs)
    ax.set_xlabel("Distogram loss weight", fontsize=fs, labelpad=1.5)
    ax.set_title("Placement recovery [%]", fontsize=fs, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for row, (key, *_rest) in enumerate(MODELS):
        m = _c_model_means(table, key)
        ax_f.text(0.5, row, f"{m.fold_recovery_pct.mean():.0f}", ha="center", va="center",
                  fontsize=fs, fontweight="bold", color=INK)
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(ax.get_ylim())
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    ax_f.set_title("Fold\nrecovery [%]", fontsize=fs, pad=4, linespacing=1.3, fontweight="bold")
    for spine in ax_f.spines.values():
        spine.set_visible(False)
    bolden(ax, fs=fs)
    return [ax, ax_f]


def panel_d(ax, ax_inset, info):
    run = os.path.join(RUNS, EXAMPLE["run"])
    bp_file = os.path.join(run, EXAMPLE["blueprint"])
    des_file = os.path.join(run, EXAMPLE["design"])
    target_bp, binder_bp = ca_trace(bp_file, "B"), ca_trace(bp_file, "A")
    target_de, binder_de = ca_trace(des_file, "B"), ca_trace(des_file, "A")
    # put the design complex in the blueprint's target frame: everything below is then measured
    # against the same target, which is what makes the placement comparison meaningful
    (target_fit, binder_fit), target_rmsd = kabsch(target_de, target_bp, [target_de, binder_de])
    drift = float(np.linalg.norm(binder_fit.mean(0) - binder_bp.mean(0)))
    (binder_on_binder,), fold_rmsd = kabsch(binder_de, binder_bp, [binder_de])

    cloud = np.concatenate([target_bp, binder_bp, binder_fit])
    basis = view_basis(binder_fit.mean(0) - binder_bp.mean(0), cloud)
    paint_chains(ax, [_runs(target_bp, basis, TARGET_GREY, 2.2),
                      _runs(binder_bp, basis, SERIES[0], 2.2),
                      _runs(binder_fit, basis, SERIES[1], 2.2)])
    p = cloud @ basis.T
    ax.set_xlim(p[:, 0].min() - 5, p[:, 0].max() + 5)
    ax.set_ylim(p[:, 1].min() - 13, p[:, 1].max() + 20)
    ax.set_aspect("equal")
    ax.set_axis_off()

    c_bp = (binder_bp.mean(0) @ basis.T)[:2]
    c_de = (binder_fit.mean(0) @ basis.T)[:2]
    y_arrow = p[:, 1].max() + 8
    ax.annotate("", xy=(c_de[0], y_arrow), xytext=(c_bp[0], y_arrow),
                arrowprops=dict(arrowstyle="<|-|>,head_width=0.24,head_length=0.48", lw=0.9,
                                color=INK2, shrinkA=0, shrinkB=0))
    ax.text((c_bp[0] + c_de[0]) / 2, y_arrow + 1.8, f"{drift:.0f} Å centroid shift", ha="center",
            va="bottom", fontsize=6.3, color=INK2)
    y_label = p[:, 1].min() - 2.5
    ax.text(p[:, 0].min(), y_label, "salad blueprint", ha="left", va="top", fontsize=6.4,
            color=SERIES[0], fontweight="bold")
    ax.text(p[:, 0].max(), y_label, "hallucinated design", ha="right", va="top", fontsize=6.4,
            color=SERIES[1], fontweight="bold")
    # one line lower, so a long target name cannot run into the two binder labels
    ax.text((c_bp[0] + c_de[0]) / 2, y_label - 6.5, "target (ubiquitin)", ha="center", va="top",
            fontsize=6.4, color="#8f8e88", fontweight="bold")

    # inset: the same two binders superimposed on each other -- the fold is the same object
    ax_inset.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax_inset.transAxes,
                                      boxstyle="round,pad=0,rounding_size=0.04", linewidth=0.6,
                                      facecolor="#FAFAF8", edgecolor="#DCDBD4", zorder=0))
    both = np.concatenate([binder_bp, binder_on_binder])
    basis_i = view_basis(np.linalg.svd(both - both.mean(0), full_matrices=False)[2][0], both)
    paint_chains(ax_inset, [_runs(binder_bp, basis_i, SERIES[0], 1.5),
                            _runs(binder_on_binder, basis_i, SERIES[1], 1.5)], halo_lw=1.1)
    q = both @ basis_i.T
    ax_inset.set_xlim(q[:, 0].min() - 2, q[:, 0].max() + 2)
    ax_inset.set_ylim(q[:, 1].min() - 2, q[:, 1].max() + 2)
    ax_inset.set_aspect("equal")
    ax_inset.set_axis_off()
    ax_inset.set_title(f"same fold:\nC$\\alpha$-RMSD {fold_rmsd:.2f} Å", fontsize=6.1, color=INK2,
                       pad=1.5, linespacing=1.3)

    info.update(target_fit_rmsd=target_rmsd, centroid_drift=drift, fold_rmsd=fold_rmsd)
    return info


def panel_e(ax, dists):
    dist_sharp, dist_soft = dists
    blank(ax)
    y0, h = 0.60, 0.30
    xs = [0.008, 0.232, 0.456, 0.680]
    w = 0.205
    box(ax, xs[0], y0, w, h, "$B$ = 10 salad\nblueprints under\none L/H/E bias", fc="#EEF4FC",
        ec=SERIES[0], fs=6.0)
    box(ax, xs[1], y0, w, h, "ProteinMPNN\non each\nblueprint", fc="#FDF0EA", ec=SERIES[1], fs=6.0)
    box(ax, xs[2], y0, w, h, "average into\none distribution\n$\\bar{p}$", fc="#ECF7F2",
        ec=SERIES[2], fs=6.0)
    box(ax, xs[3], y0, w, h, "sharpen $\\bar{p}^{\\,1/T}$,\ninitialise\nhallucination",
        fc="#FDF8E8", ec=SERIES[3], fs=6.0)
    for i in range(3):
        arrow(ax, (xs[i] + w + 0.003, y0 + h / 2), (xs[i + 1] - 0.003, y0 + h / 2))

    vmax = max(dist_sharp.max(), dist_soft.max())
    hm_y, hm_h, hm_w = 0.175, 0.215, 0.325
    for x, probs, label, sub in [(0.055, dist_sharp, "low $T$ — sharp", "blueprint fold is kept"),
                                 (0.560, dist_soft, "high $T$ — soft", "collapses into a helix")]:
        mini_logits(ax, x, hm_y, hm_w, hm_h, probs, vmax=vmax)
        ax.text(x + hm_w / 2, hm_y + hm_h + 0.075, label, ha="center", va="bottom", fontsize=6.2,
                color=INK)
        ax.text(x + hm_w / 2, hm_y + hm_h + 0.015, sub, ha="center", va="bottom", fontsize=5.9,
                color=MUTED, style="italic")
    ax.text(0.467, hm_y + hm_h / 2 + 0.035, "$T$", ha="center", va="bottom", fontsize=8, color=INK2)
    arrow(ax, (0.408, hm_y + hm_h / 2), (0.540, hm_y + hm_h / 2))
    ax.text(0.055, hm_y - 0.015, "residue  1 → 40", fontsize=5.4, color=MUTED, va="top", ha="left")
    ax.text(0.040, hm_y + hm_h / 2, "20 aa", fontsize=5.4, color=MUTED, rotation=90, va="center",
            ha="center")
    ax.text(0.5, 0.020,
            "each predictor tolerates a different $T$, and the limit moves with the sheet bias",
            fontsize=6.0, color=INK2, va="bottom", ha="center", style="italic")


def panel_f(ax, data, threshold=COLLAPSE_THRESHOLD, legend=True, title=None, ylabel=True,
            xlabel=True, fs=8.0, legend_fs=7.0, ms=5.0, dodge=0.016, errorbars=True):
    """Mean +- SEM (over the 3 replicates) of the collapse-threshold temperature: one dot per
    model and sheet bias joined by a line, models told apart by colour only, with translucent
    error bars drawn behind the line. The four models are nudged sideways by `dodge` so their
    error bars do not sit on top of each other."""
    _thresholds, replicates = data
    summary = panel_f_summary(replicates[np.isclose(replicates.threshold, threshold)])
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * (dodge if errorbars else 0.0)
    for (key, label, short, color, marker), off in zip(MODELS, offsets):
        sub = summary[summary.model == key].sort_values("sheet_bias")
        x = sub.sheet_bias + off
        # error bars first and translucent, so the connecting line stays readable through them
        if errorbars:
            ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="none", ecolor=color,
                        elinewidth=1.2, capsize=2.0, capthick=1.2, alpha=0.38, zorder=2,
                        clip_on=False)
        ax.plot(x, sub["mean"], "-", color=color, lw=1.5, zorder=3)
        ax.plot(x, sub["mean"], "o", color=color, markersize=ms, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4, clip_on=False)
    ax.set_xlim(-0.06, 0.86)
    ax.set_ylim(0.2, 1.05)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    if xlabel:
        ax.set_xlabel("β-sheet bias", fontsize=fs, labelpad=1.5)
    if ylabel:
        ax.set_ylabel("Collapse-threshold temperature $\\mathbf{T^{*}}$", fontsize=fs, labelpad=1.5)
    if title:
        ax.set_title(title, fontsize=fs - 0.5, pad=3)
    tidy(ax, grid=None)
    bolden(ax, fs=fs - 0.8)
    if legend:
        handles = [plt.Line2D([], [], color=c, lw=1.5, marker="o", markersize=ms,
                              markeredgecolor="white", label=lab) for _, lab, _, c, _m in MODELS]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False,
                  handlelength=1.5, handletextpad=0.4, labelspacing=0.4, borderpad=0.2,
                  prop=dict(size=legend_fs, weight="bold"))


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------

def write_panel_data(data_dir, data, example):
    os.makedirs(data_dir, exist_ok=True)
    data["c"].to_csv(os.path.join(data_dir, "panel_c_per_target.csv"), index=False)
    data["c_raw"].to_csv(os.path.join(data_dir, "panel_c_per_design.csv"), index=False)
    (data["c"].groupby(["model", "label", "distogram_weight"])
     .agg(n_targets=("target", "nunique"), fold_recovery_pct=("fold_recovery_pct", "mean"),
          placement_recovery_pct=("placement_recovery_pct", "mean"),
          placement_recovery_10a_pct=("placement_recovery_10a_pct", "mean"),
          iptm=("iptm", "mean"), pass_rate=("pass_rate", "mean"))
     .reset_index().to_csv(os.path.join(data_dir, "panel_c_summary.csv"), index=False))
    data["f"][0].to_csv(os.path.join(data_dir, "panel_f_threshold_temperature.csv"), index=False)
    data["f"][1].to_csv(os.path.join(data_dir, "panel_f_threshold_per_replicate.csv"), index=False)
    panel_f_summary(data["f"][1]).to_csv(
        os.path.join(data_dir, "panel_f_threshold_mean_sem.csv"), index=False)
    rows = []
    for name, df, _ in data["b"]:
        for metric in ("dssp_identity", "rmsd_to_blueprint", "plddt", "recovery"):
            v = pd.to_numeric(df[metric], errors="coerce")
            rows.append(dict(condition=name, metric=metric, n=int(v.notna().sum()),
                             mean=v.mean(), median=v.median(),
                             q05=v.quantile(0.05), q95=v.quantile(0.95)))
    pd.DataFrame(rows).to_csv(os.path.join(data_dir, "panel_b_conditions.csv"), index=False)
    pd.DataFrame([example]).to_csv(os.path.join(data_dir, "panel_d_example.csv"), index=False)


def build(out_dir, data, dpi=400):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 8.4))

    # row 1 ------------------------------------------------------------------
    panel_a(fig.add_axes([0.040, 0.760, 0.530, 0.205]))
    panel_letter(fig, 0.008, 0.990, "a")
    panel_b(fig, (0.665, 0.812, 0.315, 0.145), data["b"], gap_x=0.090, gap_y=0.100,
            legend_dy=0.050, fs=5.8, compact_labels=True, layout="grid")
    panel_letter(fig, 0.592, 0.990, "b")
    fig.text(0.700, 0.9885, "Boltz-2", fontsize=7, color=INK2, va="top")

    # row 2 ------------------------------------------------------------------
    panel_c_box(fig, (0.085, 0.445, 0.325, 0.150), data["c_raw"], fs=6.3, legend_fs=5.6,
                model_label_dy=0.085, short_labels=True, legend_ncol=2)
    panel_letter(fig, 0.008, 0.700, "c")
    ax_d = fig.add_axes([0.465, 0.390, 0.525, 0.280])
    ax_d_inset = fig.add_axes([0.855, 0.565, 0.125, 0.095])
    example = panel_d(ax_d, ax_d_inset, dict(EXAMPLE))
    panel_letter(fig, 0.430, 0.700, "d")

    # row 3 ------------------------------------------------------------------
    panel_e(fig.add_axes([0.045, 0.040, 0.500, 0.265]), data["e"])
    panel_letter(fig, 0.008, 0.340, "e")
    panel_f(fig.add_axes([0.670, 0.085, 0.300, 0.210]), data["f"], fs=6.3, legend_fs=5.6,
            ms=3.8, dodge=0.018)
    panel_letter(fig, 0.600, 0.340, "f")

    for ext, kw in (("pdf", {}), ("svg", dict(transparent=True)), ("png", dict(dpi=dpi))):
        path = os.path.join(out_dir, f"main_figure.{ext}")
        fig.savefig(path, **kw)
        print("wrote", path)
    plt.close(fig)
    return example


def build_single_panels(out_dir, data):
    """Each panel on its own canvas (pdf + svg + png), for re-arranging in a vector editor."""
    single = os.path.join(out_dir, "panels")
    os.makedirs(single, exist_ok=True)
    made = []

    fig = plt.figure(figsize=(6.6, 2.3))
    panel_a(fig.add_axes([0.01, 0.03, 0.98, 0.94]))
    made.append(("a_schematic_workflow", fig))

    fig = plt.figure(figsize=(7.0, 2.5))
    panel_b(fig, (0.072, 0.200, 0.898, 0.610), data["b"], gap_x=0.032, gap_y=0.0,
            legend_dy=0.040, legend_fs=6.3, fs=8.0, lw=1.7, layout="row", legend_ncol=3,
            compact_labels=True, minimal_yticks=False, legend_top=True)
    made.append(("b_init_vs_hallucination", fig))

    fig = plt.figure(figsize=(4.8, 4.4))
    panel_b(fig, (0.105, 0.265, 0.855, 0.695), data["b"], gap_x=0.145, gap_y=0.145,
            legend_dy=0.105, legend_fs=7.5, fs=8.0, lw=1.7, layout="grid")
    made.append(("b_alt1_grid", fig))

    fig = plt.figure(figsize=(5.8, 3.0))
    panel_b_ridge(fig, (0.200, 0.400, 0.780, 0.565), data["b"], legend_dy=0.240)
    made.append(("b_alt2_ridgeline", fig))

    fig = plt.figure(figsize=(5.2, 3.0))
    panel_c_box(fig, (0.115, 0.115, 0.865, 0.790), data["c_raw"])
    made.append(("c_fold_vs_placement", fig))

    fig = plt.figure(figsize=(5.2, 3.0))
    panel_c_box(fig, (0.115, 0.115, 0.865, 0.790), data["c_raw"], fold_once=True)
    made.append(("c_alt4_bar_fold_once", fig))

    fig = plt.figure(figsize=(5.2, 3.4))
    panel_c_case(fig, (0.115, 0.105, 0.865, 0.640), data["c_raw"], legend_ncol=1)
    made.append(("c_alt5_case_colours", fig))

    for tag, spec in EXAMPLES_C.items():
        fig = plt.figure(figsize=(5.0, 2.9))
        panel_c_example(fig, (0.015, 0.305, 0.970, 0.640), spec, data["c_raw"])
        made.append((f"c_example_structures_{tag}", fig))

    fig = plt.figure(figsize=(4.4, 3.2))
    panel_c(fig, (0.180, 0.135, 0.785, 0.745), data["c"], fs=8.0, legend_fs=7.0, legend_ncol=4,
            lw=1.7, strip_gap=0.050, legend_dy=0.025, legend_x=0.52)
    made.append(("c_alt0_weight_sweep", fig))

    fig = plt.figure(figsize=(5.2, 2.5))
    panel_c_dumbbell(fig, (0.165, 0.170, 0.805, 0.700), data["c"])
    made.append(("c_alt1_dumbbell", fig))

    fig = plt.figure(figsize=(4.6, 2.3))
    panel_c_heatmap(fig, (0.175, 0.195, 0.770, 0.605), data["c"])
    made.append(("c_alt2_heatmap", fig))

    fig = plt.figure(figsize=(3.6, 3.2))
    panel_c_scatter(fig, (0.165, 0.135, 0.805, 0.845), data["c_raw"])
    made.append(("c_alt3_scatter", fig))

    # same scatter with the y axis fitted to the bulk instead of the full range: 60 % of the
    # plot area is otherwise empty below a handful of failed AF2 designs
    fig = plt.figure(figsize=(3.6, 3.2))
    panel_c_scatter(fig, (0.165, 0.135, 0.805, 0.845), data["c_raw"], y_floor=70)
    made.append(("c_alt3_scatter_zoom", fig))

    fig = plt.figure(figsize=(4.6, 2.9))
    panel_d(fig.add_axes([0.01, 0.02, 0.98, 0.96]), fig.add_axes([0.78, 0.66, 0.21, 0.28]),
            dict(EXAMPLE))
    made.append(("d_example_structures", fig))

    fig = plt.figure(figsize=(5.6, 2.6))
    panel_e(fig.add_axes([0.01, 0.02, 0.98, 0.96]), data["e"])
    made.append(("e_schematic_averaging", fig))

    fig = plt.figure(figsize=(3.6, 3.0))
    panel_f(fig.add_axes([0.19, 0.16, 0.78, 0.80]), data["f"])
    made.append(("f_threshold_temperature", fig))

    fig = plt.figure(figsize=(3.6, 3.0))
    panel_f(fig.add_axes([0.19, 0.16, 0.78, 0.80]), data["f"], errorbars=False)
    made.append(("f_threshold_temperature_no_errorbars", fig))

    # one standalone panel f per threshold, so the choice of cut can be inspected on its own
    for threshold in THRESHOLD_SWEEP:
        fig = plt.figure(figsize=(3.2, 2.7))
        panel_f(fig.add_axes([0.17, 0.15, 0.79, 0.80]), data["f"], threshold=threshold,
                title=f"mean DSSP identity $\\geq$ {threshold:.2f}")
        made.append((f"f_threshold_temperature_dssp{threshold:.2f}", fig))

    for name, f in made:
        save_all(f, os.path.join(single, f"panel_{name}"))
        plt.close(f)
    print(f"wrote {len(made)} single panels to {single}")


def build_threshold_sweep(out_dir, data):
    """Small multiples: the same panel f under every threshold in THRESHOLD_SWEEP."""
    n = len(THRESHOLD_SWEEP)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.75 * cols, 2.75 * rows), squeeze=False)
    for i, (ax, threshold) in enumerate(zip(axes.ravel(), THRESHOLD_SWEEP)):
        missing = data["f"][0].query("threshold == @threshold and status == 'never'")
        note = f"\n({len(missing)} of 36 cells undefined)" if len(missing) else "\n "
        panel_f(ax, data["f"], threshold=threshold, legend=threshold == THRESHOLD_SWEEP[0],
                title=f"mean DSSP identity $\\geq$ {threshold:.2f}{note}",
                ylabel=i % cols == 0, xlabel=i >= n - cols)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Collapse-threshold temperature under different absolute DSSP-identity cuts",
                 fontsize=8.5, y=0.997)
    fig.text(0.5, 0.004, "A cell is undefined when the averaged initialisation is already below "
             "the cut at the sharpest temperature (T = 0.1); the curve is broken there.",
             ha="center", va="bottom", fontsize=6.2, color=MUTED)
    fig.tight_layout(rect=[0, 0.022, 1, 0.972], h_pad=3.2, w_pad=1.4)
    stem = os.path.join(out_dir, "panels", "panel_f_threshold_sweep")
    save_all(fig, stem)
    plt.close(fig)
    print("wrote", stem + ".{pdf,svg,png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_path", default=OUT)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--panels", default="True", help="'True' also writes each panel on its own")
    args = ap.parse_args()

    data = load_all()
    example = build(args.out_path, data, args.dpi)
    write_panel_data(os.path.join(args.out_path, "panel_data"), data, example)
    if args.panels == "True":
        build_single_panels(args.out_path, data)
        build_threshold_sweep(args.out_path, data)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\npanel b -- Boltz-2 settings")
        print(pd.read_csv(os.path.join(args.out_path, "panel_data", "panel_b_conditions.csv"))
              .to_string(index=False))
        print("\npanel c -- fold vs placement [%], per model and weight")
        print(data["c"].groupby(["label", "distogram_weight"])
              [["fold_recovery_pct", "placement_recovery_pct", "iptm", "pass_rate"]].mean()
              .to_string())
        print("\npanel d -- example")
        print(example)
        for threshold in sorted(set(THRESHOLD_SWEEP) | {COLLAPSE_THRESHOLD}):
            th = data["f"][0].query("threshold == @threshold")
            print(f"\npanel f -- T* at mean DSSP identity >= {threshold:.2f} "
                  f"({(th.status == 'never').sum()} undefined, "
                  f"{(th.status == 'censored').sum()} censored)")
            print(th.pivot_table(index="sheet_bias", columns="label",
                                 values="threshold_T").to_string())


if __name__ == "__main__":
    main()
