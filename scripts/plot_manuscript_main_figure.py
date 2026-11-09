"""Main manuscript figure: model biases of ProteinMPNN-distribution initialisation.

Assembles the six panels of the main figure into one page-width draft, and also writes each
panel on its own so the layout can be re-arranged in a vector editor afterwards:

    a  schematic -- blueprint -> ProteinMPNN logits -> hallucination -> blueprint comparison
    b  pMPNN init vs monomer hallucination vs binder hallucination (Boltz-2)
    c  fold vs placement recovery across structure models and distogram loss weights
    d  one example: fold recovered, placement lost
    e  schematic -- averaging several blueprints, and the temperature that controls softness
    f  collapse-threshold temperature vs sheet bias, per structure model

Run from the repo root with the `flexcraft` env (matplotlib/scipy/biotite live there, not in
`flexcraft_new`):

    python3 scripts/plot_manuscript_main_figure.py

Everything a panel plots is also written to runs/manuscript_plots/panel_data/*.csv, so the
numbers quoted in the text can be read off without re-running this script.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

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
OUT = os.path.join(RUNS, "manuscript_plots", "v1")

# Categorical slots 1-4 of the validated default palette (blue, orange, aqua, yellow), in that
# fixed order. The four-slot set clears every gate on the ADJACENT pairlist (lines), which is what
# panels c and f are; slots 3 and 4 sit below 3:1 on a white surface, so both panels carry a
# per-model marker shape as secondary encoding and a legend rather than colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8982"
GRID = "#E6E6E6"
SURFACE = "#ffffff"
TARGET_GREY = "#b4b3ad"

MODELS = [  # (key, display label, colour slot, marker)
    ("boltz2", "Boltz-2", SERIES[0], "o"),
    ("af2", "AlphaFold2", SERIES[1], "s"),
    ("protenix_v1", "Protenix v1", SERIES[2], "^"),
    ("protenix_v2", "Protenix v2", SERIES[3], "D"),
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

# panel b conditions (all Boltz-2)
LANDSCAPE_DIRECT = "landscape/test_run"
LANDSCAPE_HAL = "landscape/test_run_hallucinate"

# panel d example: a *passing* Boltz-2 binder design whose fold matches the blueprint to 0.66 A
# but which sits on the opposite face of the target (epitope recall 0, centroid drift 39 A).
EXAMPLE = dict(run="mosaic/mosaic_distogram_sweep_joltz", target="ubiquitin", attempt=0, weight=1.0,
               blueprint="blueprints/ubiquitin_0_complex.pdb",
               design="attempts/ubiquitin_0_w1.0_stage_2.pdb")

# panel f: the collapse criterion. The averaged-initialisation DSSP identity is compared against
# its own sharp-initialisation plateau rather than an absolute number, because the plateau itself
# falls with sheet bias (a batch of all-helix blueprints agrees with itself trivially, a batch of
# diverse sheets does not), so one absolute cut would not mean the same thing at E = 0 and E = 0.8.
PLATEAU_T = [0.1, 0.2, 0.3]
COLLAPSE_FRAC = 0.80

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.3,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.color": INK2,
    "ytick.color": INK2,
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


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def _dedup(df):
    keys = [c for c in ("target", "attempt", "stage", "distogram_weight") if c in df]
    return df.drop_duplicates(subset=keys, keep="last")


def load_mosaic(model):
    """stage_2 (hardened final design) rows of one model's weight sweep, base + extension."""
    frames = []
    for rel in MOSAIC_RUNS[model]:
        path = os.path.join(RUNS, rel, "results.csv")
        if not os.path.isfile(path):
            print(f"  WARNING: missing {path} -- skipped")
            continue
        df = _dedup(pd.read_csv(path))
        frames.append(df[df.stage == "stage_2"])
    return pd.concat(frames, ignore_index=True)


def load_titration(model):
    """`bias` mode rows: B independent blueprints under one soft L/H/E bias, averaged."""
    df = pd.read_csv(os.path.join(RUNS, TITRATION_RUNS[model], "results.csv"))
    b = df[df.avg_mode == "bias"].copy()
    b["e"] = b.e_bias.round(2)
    b["T"] = b.init_temperature.round(2)
    return b


def panel_b_data():
    # argmax, not the soft distribution: `recovery` is (sequence . pMPNN prob), so a soft sequence
    # scores low for being soft rather than for being wrong, and would not be comparable with the
    # two hardened hallucination columns. On DSSP identity, RMSD and pLDDT the two init variants
    # are within 0.01 / 0.5 A / 0.01 of each other -- soft vs hard belongs in the supplement.
    direct = pd.read_csv(os.path.join(RUNS, LANDSCAPE_DIRECT, "results.csv"))
    direct = direct[(direct.stage == "direct") & (direct.init == "hard")]
    hal = pd.read_csv(os.path.join(RUNS, LANDSCAPE_HAL, "results.csv"))
    hal = hal[hal.stage == "stage_1"]
    # weight 0 only: the monomer hallucination ran without a blueprint-distogram term, so the
    # binder column has to be the no-distogram arm of the sweep for the three to be comparable
    binder = load_mosaic("boltz2")
    binder = binder[binder.distogram_weight == 0.0]
    return [
        ("ProteinMPNN init, no hallucination", direct, SERIES[0]),
        ("Hallucination, monomer", hal, SERIES[1]),
        ("Hallucination, binder", binder, SERIES[2]),
    ]


def collapse_temperature(temps, values, crit):
    """Largest temperature, linearly interpolated, at which `values` still clears `crit`.

    Scans upward from the sharpest temperature and stops at the first crossing, so a late
    recovery above the collapse does not resurrect the threshold. Returns (T*, censored):
    `censored` is True when the curve never crosses inside the grid -- the model did not collapse
    at any temperature tested, and T* is only a lower bound.
    """
    temps = np.asarray(temps, float)
    values = np.asarray(values, float)
    if values[0] < crit:
        return np.nan, False
    for i in range(1, len(temps)):
        if values[i] < crit:
            f = (values[i - 1] - crit) / (values[i - 1] - values[i])
            return temps[i - 1] + f * (temps[i] - temps[i - 1]), False
    return temps[-1], True


def panel_f_data():
    """Per model: threshold temperature vs sheet bias, plus the per-replicate estimates."""
    rows, reps = [], []
    for key, label, color, marker in MODELS:
        b = load_titration(key)
        pooled = b.pivot_table(index="e", columns="T", values="dssp_identity_mean", aggfunc="mean")
        per_rep = b.pivot_table(index=["e", "replicate"], columns="T",
                                values="dssp_identity_mean", aggfunc="mean")
        for e in pooled.index:
            crit = COLLAPSE_FRAC * pooled.loc[e, PLATEAU_T].mean()
            t, censored = collapse_temperature(pooled.columns.values, pooled.loc[e].values, crit)
            rows.append(dict(model=key, label=label, sheet_bias=e,
                             plateau=pooled.loc[e, PLATEAU_T].mean(), criterion=crit,
                             threshold_T=t, censored=censored))
            for r in sorted(b.replicate.unique()):
                tr, cr = collapse_temperature(per_rep.columns.values, per_rep.loc[(e, r)].values,
                                              crit)
                reps.append(dict(model=key, sheet_bias=e, replicate=r, threshold_T=tr, censored=cr))
    return pd.DataFrame(rows), pd.DataFrame(reps)


def mosaic_table():
    rows = []
    for key, label, _, _ in MODELS:
        df = load_mosaic(key)
        for w, sub in df.groupby("distogram_weight"):
            rows.append(dict(
                model=key, label=label, distogram_weight=float(w), n=len(sub),
                n_targets=sub.target.nunique(),
                fold_recovery=sub.dssp_identity.mean(),
                fold_recovery_median=sub.dssp_identity.median(),
                placement_recovery=sub.epitope_recall_vs_blueprint.mean(),
                placement_recovery_median=sub.epitope_recall_vs_blueprint.median(),
                placement_recovery_10a=sub.epitope_recall_vs_blueprint_10a.mean(),
                rmsd_to_blueprint=sub.rmsd_to_blueprint.mean(),
                binder_centroid_drift=sub.binder_centroid_drift.mean(),
                iptm=sub.iptm.mean(), plddt=sub.plddt.mean(), pass_rate=sub.passed.mean()))
    return pd.DataFrame(rows)


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
    return dict(b=panel_b_data(), c=mosaic_table(), f=panel_f_data(),
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

    # optional conditioning, feeding the diffusion box
    cw, ch = w, 0.26
    box(ax, xs[0], y0 + h + 0.13, cw, ch,
        "target structure (binder design)\nhelix / sheet bias  (L, H, E)\n"
        "hotspot / coldspot placement",
        fc=SURFACE, ec=MUTED, lw=0.7, fs=5.9, ls=(0, (2.4, 1.8)), tc=INK2)
    arrow(ax, (xs[0] + cw / 2, y0 + h + 0.125), (xs[0] + cw / 2, y0 + h + 0.008), color=MUTED,
          lw=0.8, ls=(0, (2.4, 1.8)))
    ax.text(xs[0], y0 + h + 0.405, "optional conditioning", fontsize=5.9, color=MUTED, va="bottom",
            style="italic")


def panel_b(fig, rect, data, gap_x=0.075, gap_y=0.115, legend_dy=0.055):
    """2x2 small multiples -- one metric each, three conditions, Boltz-2 throughout."""
    metrics = [("dssp_identity", "DSSP identity", None, (0.3, 1.02)),
               ("rmsd_to_blueprint", "C$\\alpha$-RMSD (Å)", "log", None),
               ("plddt", "pLDDT", None, (0.3, 1.0)),
               ("recovery", "pMPNN recovery", None, (0.0, 1.0))]
    left, bottom, width, height = rect
    cw = (width - gap_x) / 2
    chh = (height - gap_y) / 2
    axes = []
    for i, (col, label, scale, ylim) in enumerate(metrics):
        ax = fig.add_axes([left + (i % 2) * (cw + gap_x),
                           bottom + (1 - i // 2) * (chh + gap_y), cw, chh])
        axes.append(ax)
        for j, (name, df, color) in enumerate(data):
            v = pd.to_numeric(df[col], errors="coerce").dropna().values
            bp = ax.boxplot([v], positions=[j], widths=0.58, showfliers=False, whis=(5, 95),
                            patch_artist=True, zorder=3)
            bp["boxes"][0].set(facecolor=color, alpha=0.30, edgecolor=color, linewidth=0.9)
            for part in ("whiskers", "caps"):
                for a in bp[part]:
                    a.set(color=color, linewidth=0.9)
            bp["medians"][0].set(color=INK, linewidth=1.3)
        ax.set_xlim(-0.62, len(data) - 0.38)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([])
        if scale:
            ax.set_yscale(scale)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=6.3, pad=3, color=INK)
        tidy(ax)
    # one shared legend under the block, so the four mini panels carry no condition tick labels
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=4.6, markerfacecolor=c,
                          markeredgecolor=c, alpha=0.85, label=f"{n}  (n = {len(d)})")
               for n, d, c in data]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(left - 0.012, bottom - legend_dy), ncol=1, frameon=False,
               handlelength=0.9, handletextpad=0.5, labelspacing=0.5, fontsize=6.2, borderpad=0)
    return axes


def panel_c(ax, table, legend=True):
    """Fold and placement recovery on one axis -- both are fractions in [0, 1]."""
    for key, label, color, marker in MODELS:
        sub = table[table.model == key].sort_values("distogram_weight")
        ax.plot(sub.distogram_weight, sub.fold_recovery, color=color, lw=1.5, marker=marker,
                markersize=4.0, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7,
                zorder=4, clip_on=False)
        ax.plot(sub.distogram_weight, sub.placement_recovery, color=color, lw=1.5,
                linestyle=(0, (3.2, 1.8)), marker=marker, markersize=4.0, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.1, zorder=4, clip_on=False)
    ax.set_xlim(-0.15, 5.15)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(6))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("blueprint-distogram loss weight")
    ax.set_ylabel("recovery of the salad blueprint")
    tidy(ax)
    ax.text(2.5, 1.10, "fold  (DSSP identity to blueprint)", ha="center", va="top", fontsize=6.3,
            color=INK2)
    ax.text(2.5, 0.325, "placement  (blueprint epitope recall)", ha="center", va="bottom",
            fontsize=6.3, color=INK2)
    if legend:
        models = [plt.Line2D([], [], color=c, lw=1.5, marker=m, markersize=4,
                             markerfacecolor=c, markeredgecolor="white", label=lab)
                  for _, lab, c, m in MODELS]
        style = [plt.Line2D([], [], color=INK2, lw=1.4, marker="o", markersize=4,
                            markerfacecolor=INK2, markeredgecolor="white", label="fold"),
                 plt.Line2D([], [], color=INK2, lw=1.4, linestyle=(0, (3.2, 1.8)), marker="o",
                            markersize=4, markerfacecolor="white", markeredgecolor=INK2,
                            label="placement")]
        leg = ax.legend(handles=models, loc="upper left", bbox_to_anchor=(0.03, 0.78),
                        frameon=False, handlelength=1.8, fontsize=6.3, labelspacing=0.42)
        ax.add_artist(leg)
        ax.legend(handles=style, loc="upper right", bbox_to_anchor=(0.99, 0.78), frameon=False,
                  handlelength=2.4, fontsize=6.3, labelspacing=0.42)


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

    # the two real averaged distributions this run actually used, at the two ends of the grid
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


def panel_f(ax, data, legend=True):
    thresholds, replicates = data
    for key, label, color, marker in MODELS:
        sub = thresholds[thresholds.model == key].sort_values("sheet_bias")
        rep = replicates[replicates.model == key].sort_values(["sheet_bias", "replicate"])
        jitter = np.tile([-0.013, 0.0, 0.013], len(rep) // 3 + 1)[:len(rep)]
        ax.plot(rep.sheet_bias + jitter, rep.threshold_T, ".", color=color, markersize=2.3,
                alpha=0.35, zorder=2)
        ax.plot(sub.sheet_bias, sub.threshold_T, color=color, lw=1.5, zorder=4)
        measured, censored = sub[~sub.censored], sub[sub.censored]
        ax.plot(measured.sheet_bias, measured.threshold_T, marker, color=color, markersize=4.0,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7, zorder=5,
                clip_on=False)
        ax.plot(censored.sheet_bias, censored.threshold_T, marker, color=color, markersize=4.0,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1, zorder=5,
                clip_on=False)
    ax.axhline(1.0, color=MUTED, lw=0.7, linestyle=(0, (2.5, 2.0)), zorder=1)
    ax.text(0.80, 1.015, "no collapse up to $T$ = 1  (open marker)", fontsize=5.8, color=MUTED,
            va="bottom", ha="right")
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(0.1, 1.10)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_xlabel("sheet bias  E    (L = 0.2,  H = 0.8 − E)")
    ax.set_ylabel("collapse-threshold temperature  $T^{*}$")
    tidy(ax)
    if legend:
        handles = [plt.Line2D([], [], color=c, lw=1.5, marker=m, markersize=4, markerfacecolor=c,
                              markeredgecolor="white", label=lab) for _, lab, c, m in MODELS]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.015), frameon=False,
                  handlelength=1.8, fontsize=6.3, labelspacing=0.42)


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------

def write_panel_data(data_dir, data, example):
    os.makedirs(data_dir, exist_ok=True)
    data["c"].to_csv(os.path.join(data_dir, "panel_c_fold_vs_placement.csv"), index=False)
    data["f"][0].to_csv(os.path.join(data_dir, "panel_f_threshold_temperature.csv"), index=False)
    data["f"][1].to_csv(os.path.join(data_dir, "panel_f_threshold_per_replicate.csv"), index=False)
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
    panel_a(fig.add_axes([0.045, 0.760, 0.565, 0.205]))
    panel_letter(fig, 0.008, 0.990, "a")
    panel_b(fig, (0.705, 0.785, 0.275, 0.160), data["b"], gap_x=0.085, gap_y=0.085,
            legend_dy=0.030)
    panel_letter(fig, 0.655, 0.990, "b")
    fig.text(0.688, 0.9885, "Boltz-2", fontsize=7, color=INK2, va="top")

    # row 2 ------------------------------------------------------------------
    panel_c(fig.add_axes([0.080, 0.415, 0.310, 0.240]), data["c"])
    panel_letter(fig, 0.008, 0.700, "c")
    ax_d = fig.add_axes([0.465, 0.390, 0.525, 0.280])
    ax_d_inset = fig.add_axes([0.855, 0.565, 0.125, 0.095])
    example = panel_d(ax_d, ax_d_inset, dict(EXAMPLE))
    panel_letter(fig, 0.430, 0.700, "d")

    # row 3 ------------------------------------------------------------------
    panel_e(fig.add_axes([0.045, 0.040, 0.500, 0.265]), data["e"])
    panel_letter(fig, 0.008, 0.340, "e")
    panel_f(fig.add_axes([0.665, 0.080, 0.295, 0.215]), data["f"])
    panel_letter(fig, 0.600, 0.340, "f")

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", dict(dpi=dpi))):
        path = os.path.join(out_dir, f"main_figure.{ext}")
        fig.savefig(path, **kw)
        print("wrote", path)
    plt.close(fig)
    return example


def build_single_panels(out_dir, data, dpi=400):
    """Each panel on its own canvas, for re-arranging the layout in a vector editor."""
    single = os.path.join(out_dir, "panels")
    os.makedirs(single, exist_ok=True)
    specs = []

    fig = plt.figure(figsize=(6.6, 2.3))
    panel_a(fig.add_axes([0.01, 0.03, 0.98, 0.94]))
    specs.append(("a_schematic_workflow", fig))

    fig = plt.figure(figsize=(3.2, 3.5))
    panel_b(fig, (0.16, 0.30, 0.80, 0.63), data["b"], gap_x=0.14, gap_y=0.13, legend_dy=0.075)
    specs.append(("b_init_vs_hallucination", fig))

    fig = plt.figure(figsize=(3.3, 2.9))
    panel_c(fig.add_axes([0.16, 0.13, 0.80, 0.84]), data["c"])
    specs.append(("c_fold_vs_placement", fig))

    fig = plt.figure(figsize=(4.6, 2.9))
    panel_d(fig.add_axes([0.01, 0.02, 0.98, 0.96]), fig.add_axes([0.78, 0.66, 0.21, 0.28]),
            dict(EXAMPLE))
    specs.append(("d_example_structures", fig))

    fig = plt.figure(figsize=(5.6, 2.6))
    panel_e(fig.add_axes([0.01, 0.02, 0.98, 0.96]), data["e"])
    specs.append(("e_schematic_averaging", fig))

    fig = plt.figure(figsize=(3.2, 2.7))
    panel_f(fig.add_axes([0.17, 0.15, 0.79, 0.82]), data["f"])
    specs.append(("f_threshold_temperature", fig))

    for name, f in specs:
        for ext, kw in (("pdf", {}), ("png", dict(dpi=dpi))):
            f.savefig(os.path.join(single, f"panel_{name}.{ext}"), **kw)
        plt.close(f)
    print("wrote", len(specs), "single panels to", single)


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
        build_single_panels(args.out_path, data, args.dpi)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\npanel c -- fold vs placement recovery")
        print(data["c"][["label", "distogram_weight", "n", "fold_recovery", "placement_recovery",
                         "iptm", "pass_rate"]].to_string(index=False))
        print("\npanel d -- example")
        print(example)
        print(f"\npanel f -- threshold temperature (criterion: {COLLAPSE_FRAC:.0%} of the "
              f"T <= {max(PLATEAU_T)} plateau)")
        print(data["f"][0].pivot_table(index="sheet_bias", columns="label",
                                       values="threshold_T").to_string())


if __name__ == "__main__":
    main()
