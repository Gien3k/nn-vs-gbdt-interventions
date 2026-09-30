"""H13 analysis: pre-registered endpoints P1-P5 (see h13_retuned.py).

For every neural family a, gap_a(cell) = AUC_a - max(AUC_xgb, AUC_lgbm),
all families re-tuned on the same manipulated data. A dataset enters an
endpoint only if every cell that endpoint needs exists for a, xgb and lgbm
(pre-registered completeness rule). Output: results/h13_report.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from config import RESULTS_DIR
from h13_retuned import NOISE_DOSES, OUT, TIE

REPORT = RESULTS_DIR / "h13_report.json"
NETS = ["mlp", "resnet", "ftt"]
# The FT-Transformer run was paused after tiers 0-1 (base, 4p noise,
# Gaussianisation, rotation) and resumed for tier 3 (noise doses .5/1/2),
# which finished on 2026-09-30, so its dose-response endpoint is complete.
# Endpoints listed here are reported as incomplete and excluded from
# confirmatory claims and the global correction.
INCOMPLETE: set = set()


def auc(did, kind, dose, fam):
    p = OUT / f"{did}_{kind}_{dose}_{fam}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return None if "skipped" in r else r["test_auc"]


def gap(did, kind, dose, net):
    a = auc(did, kind, dose, net)
    g = [auc(did, kind, dose, f) for f in ("xgb", "lgbm")]
    if a is None or None in g:
        return None
    return a - max(g)


N_BOOT = 5000


def endpoint(vals: dict, alt: str) -> dict:
    d = np.array(list(vals.values()))
    if len(d) < 5:
        return {"n": int(len(d))}
    sign = -1 if alt == "less" else 1
    rng = np.random.default_rng(13)
    boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))]
    return {"n": int(len(d)), "mean": float(d.mean()),
            "median": float(np.median(d)),
            "mean_ci95": [float(np.quantile(boot.mean(1), q))
                          for q in (0.025, 0.975)],
            "median_ci95": [float(np.quantile(np.median(boot, 1), q))
                            for q in (0.025, 0.975)],
            "n_direction_ok": int(((d * sign) > 0).sum()),
            "p": float(wilcoxon(d, alternative=alt).pvalue)}


def holm(entries: dict) -> None:
    ps = sorted(((k, v["p"]) for k, v in entries.items() if "p" in v),
                key=lambda kv: kv[1])
    run = 0.0
    for i, (k, p) in enumerate(ps):
        run = min(1.0, max(run, (len(ps) - i) * p))
        entries[k]["holm_p"] = float(run)


def main() -> None:
    reg = json.loads((RESULTS_DIR / "h13_prereg.json").read_text())
    ids = reg["datasets"]
    orig24 = {json.loads(p.read_text())["dataset_id"] for p in
              (RESULTS_DIR / "interventions").glob("*_noise_features_4.0.json")}
    base_gap = {d: json.loads((RESULTS_DIR / f"{d}.json").read_text())
                ["nn_minus_gbdt"] for d in ids}
    gbdt_fav = [d for d in ids if base_gap[d] <= -TIE]
    rep = {"n_datasets": len(ids), "P1": {}, "P1_orig24": {}, "P1_new": {},
           "P2": {}, "P3": {}, "P4": {}, "P5": {}, "P5_f010": {},
           "abs_auc_change_4p": {}, "gaussianise_alone": {},
           "per_dataset": {}}
    for net in NETS:
        g0 = {d: gap(d, "base", 0, net) for d in ids}
        g4 = {d: gap(d, "noise_features", 4.0, net) for d in ids}
        p1 = {d: g4[d] - g0[d] for d in ids
              if g0[d] is not None and g4[d] is not None}
        rep["P1"][net] = endpoint(p1, "less")
        rep["P1_orig24"][net] = endpoint(
            {d: v for d, v in p1.items() if d in orig24}, "less")
        rep["P1_new"][net] = endpoint(
            {d: v for d, v in p1.items() if d not in orig24}, "less")
        trends = {}
        for d in ids:
            seq = [g0[d]] + [gap(d, "noise_features", x, net)
                             for x in NOISE_DOSES]
            if None not in seq:
                trends[d] = spearmanr(range(len(seq)), seq).statistic
        trends = {d: t for d, t in trends.items() if not np.isnan(t)}
        rep["P2"][net] = endpoint(trends, "less")
        rep["P2"][net]["mean_trend_rho"] = float(np.mean(list(
            trends.values()))) if trends else None
        p3 = {d: gap(d, "noise_realistic", 4.0, net) - g0[d] for d in ids
              if g0[d] is not None
              and gap(d, "noise_realistic", 4.0, net) is not None}
        rep["P3"][net] = endpoint(p3, "less")
        p4 = {}
        ga = {}
        for d in ids:
            r_, gs = gap(d, "rotate_full", 0, net), gap(d, "gaussianise", 0,
                                                          net)
            if r_ is not None and gs is not None:
                p4[d] = r_ - gs
            if gs is not None and g0[d] is not None:
                ga[d] = gs - g0[d]
        rep["P4"][net] = endpoint(p4, "greater")
        rep["gaussianise_alone"][net] = endpoint(ga, "two-sided")
        for key, f in (("P5", 0.25), ("P5_f010", 0.1)):
            v = {d: gap(d, "select_features", f, net) - g0[d]
                 for d in gbdt_fav if g0[d] is not None
                 and gap(d, "select_features", f, net) is not None}
            rep[key][net] = endpoint(v, "greater")
        a_net = [auc(d, "noise_features", 4.0, net) - auc(d, "base", 0, net)
                 for d in p1]
        a_gb = [max(auc(d, "noise_features", 4.0, f) for f in ("xgb", "lgbm"))
                - max(auc(d, "base", 0, f) for f in ("xgb", "lgbm"))
                for d in p1]
        rep["abs_auc_change_4p"][net] = {"net": float(np.mean(a_net)),
                                         "best_gbdt": float(np.mean(a_gb))}
        rep["per_dataset"][net] = {"P1": {str(k): v for k, v in p1.items()},
                                   "P4": {str(k): v for k, v in p4.items()}}
    for key in ("P1", "P4"):
        holm(rep[key])
    for key, net in INCOMPLETE:
        rep[key][net]["incomplete"] = True
    # stricter, not pre-specified: Holm over ALL primary tests P1-P5 x nets
    flat = {f"{k}/{n}": {"p": rep[k][n]["p"]} for k in
            ("P1", "P2", "P3", "P4", "P5") for n in NETS
            if "p" in rep[k].get(n, {}) and (k, n) not in INCOMPLETE}
    holm(flat)
    rep["global_holm_all_primary"] = {k: v["holm_p"] for k, v in flat.items()}
    # secondary: re-tuned vs frozen effect for the MLP on the original 24
    frozen, ret = [], []
    for d in sorted(orig24):
        c = json.loads((RESULTS_DIR / "interventions" /
                        f"{d}_noise_features_4.0.json").read_text())
        if str(d) in rep["per_dataset"]["mlp"]["P1"]:
            frozen.append(c["nn_minus_gbdt"] - c["baseline_margin"])
            ret.append(rep["per_dataset"]["mlp"]["P1"][str(d)])
    diff = np.array(ret) - np.array(frozen)
    rep["secondary_retuned_vs_frozen_mlp"] = {
        "n": len(diff), "mean_frozen": float(np.mean(frozen)),
        "mean_retuned": float(np.mean(ret)), "mean_diff": float(diff.mean()),
        "p_two_sided": float(wilcoxon(diff).pvalue)}
    rep["incomplete_datasets"] = sorted(
        d for d in ids if any(not (OUT / f"{d}_{k}_{x}_{f}.json").exists()
                              for d_, k, x in reg["cells"] if d_ == d
                              for f in ("mlp", "resnet", "xgb", "lgbm")))
    REPORT.write_text(json.dumps(rep, indent=2))
    short = {k: v for k, v in rep.items() if k != "per_dataset"}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
