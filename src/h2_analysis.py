"""H2 confirmatory analysis: pre-registered directional tests + dose-response.

Primary endpoints (one per intervention, Holm-corrected, one-sided Wilcoxon
signed-rank on paired margin difference at maximum dose vs dose 0):
  rotate:          delta > 0   (margin moves toward the MLP)
  quantize:        delta < 0   (margin moves toward GBDT)
  noise_features:  delta < 0   (margin moves toward GBDT)

Secondary endpoint per intervention: dose-response monotonicity - Spearman
rho between severity rank and margin within each dataset, tested across
datasets with a one-sided Wilcoxon against 0 in the predicted direction.
"""
import json

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from config import RESULTS_DIR
from log_utils import log

INT_DIR = RESULTS_DIR / "interventions"
REPORT = RESULTS_DIR / "h2_report.json"

PREDICTED_SIGN = {"rotate": +1, "quantize": -1, "noise_features": -1}
# severity order (mildest -> harshest); quantize severity grows as bins shrink
SEVERITY = {
    "rotate": [0.25, 0.5, 1.0, 2.0],
    "quantize": [32, 8, 4, 2],
    "noise_features": [0.5, 1.0, 2.0, 4.0],
}


def main() -> None:
    cells = [json.loads(p.read_text()) for p in INT_DIR.glob("*.json")]
    report = {}
    for kind, sign in PREDICTED_SIGN.items():
        per_ds = {}
        for c in (c for c in cells if c["kind"] == kind):
            per_ds.setdefault(c["dataset_id"], {})[c["dose"]] = c
        deltas, trends = [], []
        rows = []
        for did, doses in sorted(per_ds.items()):
            base = next(iter(doses.values()))["baseline_margin"]
            sev_order = [d for d in SEVERITY[kind] if d in doses]
            if len(sev_order) < len(SEVERITY[kind]):
                log(f"[h2] {did}/{kind}: incomplete doses {sorted(doses)}")
            margins = [doses[d]["nn_minus_gbdt"] for d in sev_order]
            max_dose = sev_order[-1]
            delta = doses[max_dose]["nn_minus_gbdt"] - base
            deltas.append(delta)
            # severity rank vs margin, including dose 0 (rank 0)
            sev_ranks = list(range(len(margins) + 1))
            rho = spearmanr(sev_ranks, [base] + margins).statistic
            trends.append(rho)
            rows.append({"dataset_id": did, "baseline": base,
                         "margins_by_severity": margins,
                         "delta_at_max": delta, "trend_rho": float(rho)})
        deltas, trends = np.array(deltas), np.array(trends)
        alt = "greater" if sign > 0 else "less"
        w_delta = wilcoxon(deltas, alternative=alt)
        w_trend = wilcoxon(trends, alternative=alt)
        report[kind] = {
            "n_datasets": int(len(deltas)),
            "predicted_direction": "+" if sign > 0 else "-",
            "mean_delta_at_max_dose": float(deltas.mean()),
            "median_delta_at_max_dose": float(np.median(deltas)),
            "n_datasets_direction_ok": int(((deltas * sign) > 0).sum()),
            "wilcoxon_delta_p_onesided": float(w_delta.pvalue),
            "mean_trend_rho": float(trends.mean()),
            "wilcoxon_trend_p_onesided": float(w_trend.pvalue),
            "per_dataset": rows,
        }

    # Holm correction over the three primary (delta) endpoints
    prim = sorted(((k, report[k]["wilcoxon_delta_p_onesided"])
                   for k in report), key=lambda kv: kv[1])
    m = len(prim)
    running_max = 0.0
    for rank, (k, p) in enumerate(prim):
        adj = min(1.0, max(running_max, (m - rank) * p))
        running_max = adj
        report[k]["holm_adjusted_p"] = float(adj)
        report[k]["confirmed_at_0.05"] = bool(adj < 0.05)

    REPORT.write_text(json.dumps(report, indent=2))
    for k in report:
        r = report[k]
        log(f"[h2] {k}: mean_delta={r['mean_delta_at_max_dose']:+.4f} "
            f"dir_ok={r['n_datasets_direction_ok']}/{r['n_datasets']} "
            f"p={r['wilcoxon_delta_p_onesided']:.4f} "
            f"holm={r['holm_adjusted_p']:.4f} "
            f"trend_rho={r['mean_trend_rho']:+.3f} "
            f"(p={r['wilcoxon_trend_p_onesided']:.4f})")
    print(json.dumps({k: {kk: v for kk, v in r.items() if kk != "per_dataset"}
                      for k, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
