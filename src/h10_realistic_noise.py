"""H10: robustness of the headline effect to REALISTIC uninformative
features (audit-stage reviewer defence).

PRE-REGISTERED (fixed before execution): appending 4x p features drawn
i.i.d. from the TRAIN-split empirical marginal of randomly chosen real
columns (realistic marginals; zero label information; no test-row values
used) lowers the margin, as Gaussian noise does. One-sided Wilcoxon
(less) across all noise-cohort datasets; secondary: paired comparison of
effect size vs Gaussian noise at the same dose.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_DIR, RESULTS_DIR
from log_utils import log
from models import build_model
from train_baseline import _load, _score

INT_DIR = RESULTS_DIR / "interventions"
REPORT = RESULTS_DIR / "h10_report.json"
DOSE = 4.0
SEEDS = [0, 1, 2]


def realistic_noise(X: pd.DataFrame, itr: np.ndarray, num: list[str],
                    seed: int = 0) -> pd.DataFrame:
    X = X.copy()
    rng = np.random.default_rng(seed)
    k = max(1, int(round(DOSE * X.shape[1])))
    src_cols = rng.choice(num, size=k, replace=True) if num else []
    for j, col in enumerate(src_cols):
        vals = X[col].iloc[itr].dropna().values
        if len(vals) < 2:
            vals = np.array([0.0, 1.0])
        X[f"__mnoise_{j}"] = rng.choice(vals, size=len(X), replace=True)
    return X


def main() -> None:
    print("INITIATING AUTONOMOUS COMPUTATION. Rationale: reviewers will object "
          "that i.i.d. Gaussian noise features are trivially separable from "
          "real columns; drawing noise from train-split empirical marginals of "
          "real features removes that objection while preserving zero label "
          "information. Estimated time: 1-2 hours.")
    ids = sorted({json.loads(p.read_text())["dataset_id"]
                  for p in INT_DIR.glob("*_noise_features_4.0.json")})
    log(f"[h10] realistic-marginal noise on {len(ids)} datasets")
    deltas, pairs, rows = [], [], []
    for did in ids:
        out_path = INT_DIR / f"{did}_noise_realistic_{DOSE}.json"
        if out_path.exists():
            cell = json.loads(out_path.read_text())
        else:
            meta, X, y, itr, iva, ite = _load(did)
            base = json.loads((RESULTS_DIR / f"{did}.json").read_text())
            cat = meta["categorical_cols"]
            num = [c for c in X.columns if c not in cat]
            if not num:
                log(f"[h10] {did}: no numeric columns, skip")
                continue
            Xt = realistic_noise(X, itr, num)
            num_t = [c for c in Xt.columns if c not in cat]
            fams = {}
            for fam in ("mlp", "xgb", "lgbm"):
                cfg = base["families"][fam]["best_cfg"]
                aucs = []
                for s in SEEDS:
                    pipe = build_model(fam, cfg, cat, num_t, seed=s)
                    pipe.fit(Xt.iloc[itr], y.iloc[itr])
                    aucs.append(_score(pipe, Xt.iloc[ite],
                                       y.iloc[ite])["roc_auc_ovr"])
                fams[fam] = {"test_auc_mean": float(np.mean(aucs))}
            nn = fams["mlp"]["test_auc_mean"]
            gb = max(fams["xgb"]["test_auc_mean"],
                     fams["lgbm"]["test_auc_mean"])
            cell = {"dataset_id": did, "kind": "noise_realistic",
                    "dose": DOSE, "families": fams,
                    "nn_minus_gbdt": float(nn - gb),
                    "baseline_margin": base["nn_minus_gbdt"]}
            out_path.write_text(json.dumps(cell, indent=2))
            log(f"[h10] {did}: margin={nn - gb:+.4f} "
                f"(baseline {cell['baseline_margin']:+.4f})")
        delta = cell["nn_minus_gbdt"] - cell["baseline_margin"]
        deltas.append(delta)
        g = json.loads((INT_DIR / f"{did}_noise_features_4.0.json")
                       .read_text())
        pairs.append((delta, g["nn_minus_gbdt"] - g["baseline_margin"]))
        rows.append({"dataset_id": did, "delta_realistic": float(delta),
                     "delta_gaussian": float(pairs[-1][1])})

    d = np.array(deltas)
    diff = np.array([a - b for a, b in pairs])
    rep = {
        "n_datasets": int(len(d)),
        "mean_delta_realistic": float(d.mean()),
        "n_direction_ok": int((d < 0).sum()),
        "wilcoxon_p_onesided": float(wilcoxon(d, alternative="less").pvalue),
        "mean_delta_gaussian_same_ds": float(np.mean([b for _, b in pairs])),
        "paired_realistic_minus_gaussian_mean": float(diff.mean()),
        "paired_wilcoxon_p_twosided": float(wilcoxon(diff).pvalue),
        "per_dataset": rows,
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    log(f"[h10] delta={d.mean():+.4f} dir={int((d < 0).sum())}/{len(d)} "
        f"p={rep['wilcoxon_p_onesided']:.2e}; vs gaussian paired "
        f"diff={diff.mean():+.4f} (p={rep['paired_wilcoxon_p_twosided']:.3f})")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_dataset"},
                     indent=2))


if __name__ == "__main__":
    main()
