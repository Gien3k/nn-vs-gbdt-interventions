"""H6 analysis: intervention endpoints split by cohort.

The n=12 pilot (pre-registered in interventions.py) is analysed by
h2_analysis.py. H6 extends the grid to all feasible decided datasets; this
script reports, for each intervention: the pilot cohort, the EXTENSION
cohort (new datasets only - an independent confirmatory sample), and the
pooled suite. Same one-sided endpoints, Holm over the three interventions
within each cohort.
"""
import json

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from config import RESULTS_DIR
from log_utils import log

INT_DIR = RESULTS_DIR / "interventions"
PILOT_FILE = RESULTS_DIR / "pilot_ids.json"
REPORT = RESULTS_DIR / "h6_report.json"

PREDICTED_SIGN = {"rotate": +1, "quantize": -1, "noise_features": -1}
SEVERITY = {"rotate": [0.25, 0.5, 1.0, 2.0],
            "quantize": [32, 8, 4, 2],
            "noise_features": [0.5, 1.0, 2.0, 4.0]}


def endpoints(cells: list[dict], sign: int, kind: str) -> dict | None:
    per_ds: dict[int, dict] = {}
    for c in cells:
        per_ds.setdefault(c["dataset_id"], {})[c["dose"]] = c
    deltas, trends = [], []
    for did, doses in sorted(per_ds.items()):
        sev = [d for d in SEVERITY[kind] if d in doses]
        if not sev:
            continue
        base = doses[sev[0]]["baseline_margin"]
        margins = [doses[d]["nn_minus_gbdt"] for d in sev]
        deltas.append(doses[sev[-1]]["nn_minus_gbdt"] - base)
        trends.append(spearmanr(range(len(margins) + 1),
                                [base] + margins).statistic)
    if len(deltas) < 5:
        return None
    d, t = np.array(deltas), np.array(trends)
    alt = "greater" if sign > 0 else "less"
    return {
        "n_datasets": int(len(d)),
        "mean_delta_at_max": float(d.mean()),
        "n_direction_ok": int(((d * sign) > 0).sum()),
        "wilcoxon_delta_p": float(wilcoxon(d, alternative=alt).pvalue),
        "mean_trend_rho": float(t.mean()),
        "wilcoxon_trend_p": float(wilcoxon(t, alternative=alt).pvalue),
    }


def holm(entries: dict) -> None:
    prim = sorted(((k, v["wilcoxon_delta_p"]) for k, v in entries.items()
                   if v), key=lambda kv: kv[1])
    running = 0.0
    for rank, (k, p) in enumerate(prim):
        adj = min(1.0, max(running, (len(prim) - rank) * p))
        running = adj
        entries[k]["holm_p"] = float(adj)
        entries[k]["confirmed_at_0.05"] = bool(adj < 0.05)


def main() -> None:
    pilot = set(json.loads(PILOT_FILE.read_text()))
    cells = [json.loads(p.read_text()) for p in INT_DIR.glob("*.json")]
    report: dict = {"pilot_ids": sorted(pilot)}
    for cohort in ("pilot", "extension", "pooled"):
        entry = {}
        for kind, sign in PREDICTED_SIGN.items():
            sub = [c for c in cells if c["kind"] == kind]
            if cohort == "pilot":
                sub = [c for c in sub if c["dataset_id"] in pilot]
            elif cohort == "extension":
                sub = [c for c in sub if c["dataset_id"] not in pilot]
            entry[kind] = endpoints(sub, sign, kind)
        holm(entry)
        report[cohort] = entry
        for kind, v in entry.items():
            if v:
                log(f"[h6] {cohort}/{kind}: n={v['n_datasets']} "
                    f"delta={v['mean_delta_at_max']:+.4f} "
                    f"dir={v['n_direction_ok']}/{v['n_datasets']} "
                    f"holm_p={v.get('holm_p', float('nan')):.4f}")
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({c: {k: (v if not v else {kk: v[kk] for kk in
          ("n_datasets", "mean_delta_at_max", "n_direction_ok",
           "holm_p", "confirmed_at_0.05") if kk in v})
          for k, v in report[c].items()} for c in
          ("pilot", "extension", "pooled")}, indent=2))


if __name__ == "__main__":
    main()
