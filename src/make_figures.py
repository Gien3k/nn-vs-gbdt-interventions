"""Publication figures for the manuscript.

Fig 1  margin-map.pdf      sorted per-dataset NN-GBDT margins, tie band
Fig 2  dose-response.pdf   small multiples: margin change vs dose per intervention
Fig 3  causal-align.pdf    H5 scatter: predicted vs observed margin change
Fig 4  architectures.pdf   noise dose-response and feature removal per network
Fig 5  confirmatory.pdf    protocol B (H13): noise, full rotation, removal

Design: Okabe-Ito CVD-safe hues (validated), thin marks, recessive grid,
direct labels, one axis per panel, diverging semantics only where polarity
is the message (Fig 1).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

FIG_DIR = RESULTS_DIR.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERMIL, GRAY = "#D55E00", "#8a8a86"
TIE_EPS = 0.005

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})

KIND_LABEL = {"rotate": "Rotation", "quantize": "Discretisation",
              "noise_features": "Uninformative features"}
KIND_COLOR = {"rotate": BLUE, "quantize": ORANGE, "noise_features": GREEN}
KIND_DOSES = {"rotate": ["0", "0.25", "0.5", "1.0", "2.0"],
              "quantize": ["none", "32", "8", "4", "2"],
              "noise_features": ["0", "0.5x", "1x", "2x", "4x"]}


def fig1_margin_map() -> None:
    rows = []
    for p in sorted(RESULTS_DIR.glob("[0-9]*.json")):
        r = json.loads(p.read_text())
        if "nn_minus_gbdt" in r:
            rows.append((r["nn_minus_gbdt"], r["name"]))
    rows.sort()
    margins = np.array([m for m, _ in rows])
    y = np.arange(len(rows))

    colors = [GRAY if abs(m) < TIE_EPS else (BLUE if m > 0 else VERMIL)
              for m in margins]
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    ax.axvspan(-TIE_EPS, TIE_EPS, color="#efefec", zorder=0)
    ax.barh(y, margins, height=0.62, color=colors)
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([(n[:20] + "…") if len(n) > 21 else n
                        for _, n in rows], fontsize=5.2)
    ax.set_xlabel("Gap $\\Delta$: test AUC of tuned MLP $-$ best tuned GBDT")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    # positive margins (MLP better) sort to the TOP of the chart
    ax.text(0.985, 0.90, "MLP better $\\rightarrow$", transform=ax.transAxes,
            ha="right", color=BLUE, fontsize=7)
    ax.text(0.015, 0.03, "$\\leftarrow$ GBDT better", transform=ax.transAxes,
            ha="left", color=VERMIL, fontsize=7)
    ax.text(0.5, 0.535, f"tie band $|\\Delta|<{TIE_EPS}$", rotation=90,
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=6, color="#777")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.995, bottom=0.075)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"margin-map.{ext}", dpi=300)
    plt.close(fig)


def fig2_dose_response() -> None:
    """Four panels built directly from intervention cells (pooled cohorts);
    significance annotations from h6 (first three) and h7 (selection)."""
    int_dir = RESULTS_DIR / "interventions"
    cells = [json.loads(p.read_text()) for p in int_dir.glob("*.json")]
    h6 = json.loads((RESULTS_DIR / "h6_report.json").read_text())["pooled"]
    h7 = json.loads((RESULTS_DIR / "h7_report.json").read_text())

    kinds = ("rotate", "quantize", "noise_features", "select_features")
    sev_doses = {"rotate": [0.25, 0.5, 1.0, 2.0],
                 "quantize": [32, 8, 4, 2],
                 "noise_features": [0.5, 1.0, 2.0, 4.0],
                 "select_features": [0.75, 0.5, 0.25, 0.1]}
    labels = dict(KIND_LABEL, select_features="Feature removal")
    colors = dict(KIND_COLOR, select_features=VERMIL)
    tick_lbls = dict(KIND_DOSES,
                     select_features=["all", "75%", "50%", "25%", "10%"])
    stats = {
        "rotate": (h6["rotate"]["confirmed_at_0.05"], h6["rotate"]["holm_p"]),
        "quantize": (h6["quantize"]["confirmed_at_0.05"],
                     h6["quantize"]["holm_p"]),
        "noise_features": (h6["noise_features"]["confirmed_at_0.05"],
                           h6["noise_features"]["holm_p"]),
        "select_features": (h7["primary_wilcoxon_p_onesided"] < 0.05,
                            h7["primary_wilcoxon_p_onesided"]),
    }

    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.6), sharey=True)
    for ax, kind in zip(axes, kinds):
        per_ds: dict[int, dict] = {}
        for c in (c for c in cells if c["kind"] == kind):
            per_ds.setdefault(c["dataset_id"], {})[c["dose"]] = c
        sev = np.arange(len(sev_doses[kind]) + 1)
        curves = []
        for did, doses in sorted(per_ds.items()):
            avail = [d for d in sev_doses[kind] if d in doses]
            base = doses[avail[0]]["baseline_margin"]
            vals = np.array([base] + [doses[d]["nn_minus_gbdt"]
                                      for d in avail])
            deltas = vals - vals[0]
            curves.append(deltas)
            ax.plot(sev[:len(deltas)], deltas, color="#c9c9c5", lw=0.6,
                    zorder=1)
        mean = np.mean([c for c in curves if len(c) == len(sev)], axis=0)
        c = colors[kind]
        ax.plot(sev, mean, color=c, lw=2.0, zorder=3, marker="o", ms=3.2)
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(sev)
        ax.set_xticklabels(tick_lbls[kind])
        ax.set_xlabel("dose")
        conf, p = stats[kind]
        ptxt = f"$p={p:.3f}$" if p >= 1e-3 else "$p<10^{-3}$"
        n = len(per_ds)
        ax.set_title(f"{labels[kind]} (n={n})\n"
                     f"{'confirmed' if conf else 'n.s.'}, {ptxt}",
                     color="#333", fontsize=8)
        ax.set_xlim(-0.3, len(sev) - 0.6)
        ax.text(sev[-1], mean[-1], f"  {mean[-1]:+.3f}", color=c,
                fontsize=7, va="center", clip_on=False)
    axes[0].set_ylabel("Change in gap vs dose 0\n(MLP $-$ best GBDT, test AUC)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"dose-response.{ext}", dpi=300)
    plt.close(fig)


def fig3_causal_align() -> None:
    h5 = json.loads((RESULTS_DIR / "h5_report.json").read_text())
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    for kind in ("rotate", "quantize", "noise_features"):
        rows = [r for r in h5["rows"] if r["kind"] == kind]
        ax.scatter([r["delta_pred"] for r in rows],
                   [r["delta_obs"] for r in rows],
                   s=14, color=KIND_COLOR[kind], label=KIND_LABEL[kind],
                   alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="#555", lw=0.7)
    ax.axvline(0, color="#555", lw=0.7)
    h12 = json.loads((RESULTS_DIR / "h12_report.json").read_text())
    pil = h12["B_alignment_clustered"]["pilot"]["all"]
    lo, hi = pil["cluster_bootstrap_ci95"]
    ax.text(0.03, 0.97,
            f"$\\rho_S={pil['rho']:.2f}$, cluster-bootstrap 95% CI "
            f"[{lo:.2f}, {hi:.2f}]\n{pil['n_cells']} cells, "
            f"{pil['n_datasets']} datasets",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("predicted change in gap (meta-model, LODO)")
    ax.set_ylabel("observed change in gap")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"causal-align.{ext}", dpi=300)
    plt.close(fig)


ARCH_STYLE = {"mlp": ("MLP", BLUE, "o"), "resnet": ("ResNet", ORANGE, "s"),
              "ftt": ("FT-Transformer", GREEN, "^"),
              "tabpfn": ("TabPFN v2", VERMIL, "D")}


def _arch_deltas(arch: str, kind: str, dose: float) -> list[float]:
    """Change in gap vs the architecture's own dose-0 gap, per dataset."""
    if arch == "mlp":
        cells = [json.loads(p.read_text()) for p in
                 (RESULTS_DIR / "interventions").glob(f"*_{kind}_{dose}.json")]
        return [c["nn_minus_gbdt"] - c["baseline_margin"] for c in cells]
    cells = [json.loads(p.read_text()) for p in
             (RESULTS_DIR / "arch").glob(f"*_{arch}_{kind}_{dose}.json")]
    return [c["margin"] - c["baseline_margin"] for c in cells]


def fig4_architectures() -> None:
    """Uninformative-feature dose-response and feature removal for every
    neural architecture (mean +- standard error across datasets)."""
    doses = [0.5, 1.0, 2.0, 4.0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    removal = {}
    for arch, (name, col, mk) in ARCH_STYLE.items():
        per_dose = [_arch_deltas(arch, "noise_features", d) for d in doses]
        if not per_dose[-1]:
            continue
        m = [0.0] + [float(np.mean(v)) for v in per_dose]
        se = [0.0] + [float(np.std(v, ddof=1) / np.sqrt(len(v)))
                      for v in per_dose]
        a1.errorbar(np.arange(len(m)), m, yerr=se, color=col, marker=mk,
                    ms=3.5, lw=1.6, capsize=2,
                    label=f"{name} (n={len(per_dose[-1])})")
        rem = _arch_deltas(arch, "select_features", 0.25)
        if rem:
            removal[arch] = rem
    a1.axhline(0, color="#555", lw=0.7)
    a1.set_xticks(np.arange(5))
    a1.set_xticklabels(["0", "0.5p", "p", "2p", "4p"])
    a1.set_xlabel("number of added uninformative features")
    a1.set_ylabel("Change in gap vs dose 0\n(network $-$ best GBDT, AUC)")
    a1.set_title("Adding uninformative features", fontsize=8, color="#333")
    a1.legend(frameon=False, fontsize=6.5, loc="lower left")
    rng = np.random.default_rng(0)
    for i, (arch, vals) in enumerate(removal.items()):
        name, col, mk = ARCH_STYLE[arch]
        a2.scatter(i + (rng.random(len(vals)) - 0.5) * 0.25, vals, s=9,
                   color=col, alpha=0.55, edgecolors="none")
        a2.errorbar(i, np.mean(vals),
                    yerr=np.std(vals, ddof=1) / np.sqrt(len(vals)),
                    color=col, marker=mk, ms=5, capsize=3, lw=1.6)
    a2.axhline(0, color="#555", lw=0.7)
    a2.set_xticks(range(len(removal)))
    a2.set_xticklabels([ARCH_STYLE[a][0] for a in removal], fontsize=6.5)
    a2.set_title("Keeping the top 25% of features", fontsize=8,
                 color="#333")
    a2.set_ylabel("Change in gap")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"architectures.{ext}", dpi=300)
    plt.close(fig)


def fig5_confirmatory() -> None:
    """Protocol B (H13): noise dose-response, full rotation vs
    Gaussianisation, feature removal; every family re-tuned per cell."""
    from h13_analysis import gap
    from h13_retuned import NOISE_DOSES, TIE
    reg = json.loads((RESULTS_DIR / "h13_prereg.json").read_text())
    ids = reg["datasets"]
    fav = [d for d in ids if json.loads((RESULTS_DIR / f"{d}.json")
                                        .read_text())["nn_minus_gbdt"] <= -TIE]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(8.6, 2.8),
                                     gridspec_kw={"width_ratios": [1.5, 1, 1]})
    nets = [a for a in ("mlp", "resnet", "ftt")]
    panels = {"rot": [], "sel": []}
    for arch in nets:
        name, col, mk = ARCH_STYLE[arch]
        g0 = {d: gap(d, "base", 0, arch) for d in ids}
        per_dose = []
        for x in NOISE_DOSES:
            per_dose.append([gap(d, "noise_features", x, arch) - g0[d]
                             for d in ids if g0[d] is not None
                             and gap(d, "noise_features", x, arch)
                             is not None])
        if len(per_dose[-1]) < 5:
            continue
        m = [0.0] + [float(np.mean(v)) for v in per_dose]
        se = [0.0] + [float(np.std(v, ddof=1) / np.sqrt(len(v)))
                      for v in per_dose]
        a1.errorbar(np.arange(5), m, yerr=se, color=col, marker=mk, ms=3.5,
                    lw=1.6, capsize=2,
                    label=f"{name} (n={len(per_dose[-1])})")
        rot = [gap(d, "rotate_full", 0, arch) - gap(d, "gaussianise", 0, arch)
               for d in ids if gap(d, "rotate_full", 0, arch) is not None
               and gap(d, "gaussianise", 0, arch) is not None]
        sel = [gap(d, "select_features", 0.25, arch) - g0[d] for d in fav
               if g0[d] is not None
               and gap(d, "select_features", 0.25, arch) is not None]
        panels["rot"].append((arch, rot))
        panels["sel"].append((arch, sel))
    a1.axhline(0, color="#555", lw=0.7)
    a1.set_xticks(np.arange(5))
    a1.set_xticklabels(["0", "0.5p", "p", "2p", "4p"])
    a1.set_xlabel("number of added uninformative features")
    a1.set_ylabel("Change in gap\n(network $-$ best GBDT, AUC)")
    a1.set_title("Uninformative features (P1, P2)", fontsize=8, color="#333")
    a1.legend(frameon=False, fontsize=6.5, loc="lower left")
    rng = np.random.default_rng(0)
    for ax, key, title in ((a2, "rot", "Full rotation (P4)"),
                           (a3, "sel", "Keep 25% of features (P5)")):
        for i, (arch, vals) in enumerate(panels[key]):
            if len(vals) < 5:
                continue
            name, col, mk = ARCH_STYLE[arch]
            ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.3, vals, s=7,
                       color=col, alpha=0.45, edgecolors="none")
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1)
                        / np.sqrt(len(vals)), color=col, marker=mk, ms=5,
                        capsize=3, lw=1.6)
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(range(len(panels[key])))
        ax.set_xticklabels([ARCH_STYLE[a][0] for a, _ in panels[key]],
                           fontsize=6.5)
        ax.set_title(title, fontsize=8, color="#333")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"confirmatory.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_margin_map()
    fig2_dose_response()
    fig3_causal_align()
    fig4_architectures()
    if (RESULTS_DIR / "h13_prereg.json").exists():
        fig5_confirmatory()
    print("figures written to", FIG_DIR)
