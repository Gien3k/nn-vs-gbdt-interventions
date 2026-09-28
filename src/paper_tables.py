"""Generate LaTeX table rows for the revised manuscript directly from the
result JSONs (no hand-copied numbers). Output: paper/tables/*.tex
"""
import json

from config import RESULTS_DIR, ROOT

TAB_DIR = ROOT / "paper" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def fmt_p(p: float | None) -> str:
    if p is None:
        return "--"
    if p < 1e-3:
        return r"$<$0.001"
    return f"{p:.3f}"


def ties_rows(h12: dict) -> str:
    label = {"|d|<0.001": r"$|\Delta|<0.001$",
             "|d|<0.0025": r"$|\Delta|<0.0025$",
             "|d|<0.005": r"$|\Delta|<0.005$ (pre-specified)",
             "|d|<0.01": r"$|\Delta|<0.01$",
             "|d|<0.02": r"$|\Delta|<0.02$",
             "|d|<2SE(seed)": r"$|\Delta|<2\,\mathrm{SE}$",
             "|d|<0.02*(1-AUC)": r"$|\Delta|<0.02\,(1-\mathrm{AUC}_{\max})$",
             "|d|<0.05*(1-AUC)": r"$|\Delta|<0.05\,(1-\mathrm{AUC}_{\max})$",
             "|d|<0.1*(1-AUC)": r"$|\Delta|<0.10\,(1-\mathrm{AUC}_{\max})$"}
    out = []
    for r in h12["A_tie_sensitivity"]["rules"]:
        w = r.get("winner_classification")
        cls = (f"{w['lodo_acc']:.3f} & {w['majority_baseline']:.3f} & "
               f"{fmt_p(w['perm_p'])}") if w else "-- & -- & --"
        out.append(f"{label[r['rule']]} & {r['n_ties']} & {r['n_mlp_wins']} "
                   f"& {r['n_gbdt_wins']} & {cls} \\\\")
    return "\n".join(out) + "\n\\bottomrule\n"


def align_rows(h12: dict) -> str:
    names = {"all": "all three", "without_noise": "without noise",
             "noise_features": "noise only", "rotate": "rotation only",
             "quantize": "discretisation only"}
    out = []
    for coh in ("pilot", "extension", "pooled"):
        first = True
        for key in ("all", "without_noise", "noise_features", "rotate",
                    "quantize"):
            s = h12["B_alignment_clustered"][coh][key]
            lo, hi = s["cluster_bootstrap_ci95"]
            coh_lbl = coh.capitalize() if first else ""
            first = False
            out.append(
                f"{coh_lbl} & {names[key]} & {s['n_cells']} & "
                f"{s['rho']:.2f} & [{lo:.2f}, {hi:.2f}] & "
                f"{s['per_dataset_rho_n_pos']}, {fmt_p(s['per_dataset_wilcoxon_p'])} & "
                f"{fmt_p(s['dataset_identity_perm_p'])} \\\\")
        if coh != "pooled":
            out.append(r"\addlinespace")
    return "\n".join(out) + "\n\\bottomrule\n"


def arch_rows(h11: dict) -> str:
    names = {"mlp": "MLP", "resnet": "ResNet", "ftt": "FT-Transformer",
             "tabpfn": "TabPFN v2"}
    out = []
    for a in ("mlp", "resnet", "ftt", "tabpfn"):
        r = h11["archs"].get(a)
        if not r:
            continue
        n4 = r["noise4x"]
        p_main = n4.get("holm_p", n4["wilcoxon_p"])
        real = r.get("noise_realistic4x")
        resc = r.get("rescue_f025")
        out.append(
            f"{names[a]} & {r['baseline_margin_mean']:+.3f} & "
            f"{n4['mean_delta']:+.3f} & {n4['n_direction_ok']}/{n4['n']} & "
            f"{fmt_p(p_main)} & {r['noise4x_abs_auc_change']:+.3f} & "
            f"{r['noise_trend_rho_mean']:+.2f} & "
            + (f"{real['mean_delta']:+.3f} ({real['n_direction_ok']}/{real['n']})"
               if real else "--") + " & "
            + (f"{resc['mean_delta']:+.3f} ({resc['n_direction_ok']}/{resc['n']}), "
               f"{fmt_p(resc['wilcoxon_p'])}" if resc else "--") + r" \\")
    return "\n".join(out) + "\n\\bottomrule\n"


def confirm_rows(h13: dict) -> str:
    names = {"mlp": "MLP", "resnet": "ResNet", "ftt": "FT-Transformer"}
    hyp = [("P1", "P1: $4p$ uninformative features"),
           ("P1_orig24", "\\quad protocol-A datasets"),
           ("P1_new", "\\quad new datasets"),
           ("P2", "P2: dose trend$^{a}$"),
           ("P3", "P3: realistic features"),
           ("P4", "P4: full rotation"),
           ("P5", "P5: keep 25\\% of features")]
    out = []
    for key, label in hyp:
        first = True
        for net in ("mlp", "resnet", "ftt"):
            e = h13[key].get(net, {})
            if "p" not in e or e.get("incomplete"):
                continue
            p = e.get("holm_p", e["p"])
            lo, hi = e["mean_ci95"]
            fmt = "{:+.2f}" if key == "P2" else "{:+.3f}"
            val = fmt.format(e["mean_trend_rho"] if key == "P2"
                             else e["mean"])
            ci = f"[{fmt.format(lo)}, {fmt.format(hi)}]"
            out.append(f"{label if first else ''} & {names[net]} & {e['n']} "
                       f"& {val} & {ci} & {e['n_direction_ok']} & "
                       f"{fmt_p(p)} \\\\")
            first = False
        if key in ("P1_new", "P3", "P4"):
            out.append(r"\addlinespace")
    out.append(r"\bottomrule")
    out.append(r"\multicolumn{7}{@{}l}{\scriptsize $^{a}$Mean "
               r"within-dataset Spearman correlation between dose and gap.}"
               r" \\")
    return "\n".join(out) + "\n"


def main() -> None:
    h12 = json.loads((RESULTS_DIR / "h12_report.json").read_text())
    (TAB_DIR / "ties_rows.tex").write_text(ties_rows(h12))
    (TAB_DIR / "align_rows.tex").write_text(align_rows(h12))
    h11p = RESULTS_DIR / "h11_report.json"
    if h11p.exists():
        (TAB_DIR / "arch_rows.tex").write_text(
            arch_rows(json.loads(h11p.read_text())))
    h13p = RESULTS_DIR / "h13_report.json"
    if h13p.exists():
        (TAB_DIR / "confirm_rows.tex").write_text(
            confirm_rows(json.loads(h13p.read_text())))
    print(sorted(p.name for p in TAB_DIR.glob("*.tex")))


if __name__ == "__main__":
    main()
