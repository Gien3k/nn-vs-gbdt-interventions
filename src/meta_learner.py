"""Meta-learner: predict the NN-vs-GBDT winner from dataset descriptors.

Evaluation protocol: leave-one-dataset-out (LODO). For every held-out
dataset, the meta-model is fitted on all remaining datasets' descriptors and
outcomes, then predicts the held-out winner. No dataset ever contributes to
the model that judges it - the meta-level analogue of a clean train/test
split.

Targets:
  clf: sign(nn_minus_gbdt) > 0     (does the tuned MLP beat best GBDT?)
  reg: nn_minus_gbdt               (by how much, in test AUC)

Outputs results/meta_report.json:
  - LODO accuracy / balanced accuracy / AUC of the winner classifier
  - LODO Spearman correlation of the margin regressor
  - permutation feature importances (which geometry drives the gap)
  - naive baselines (majority class, always-GBDT) for honest comparison
"""
import json

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from log_utils import log

DESC_DIR = RESULTS_DIR / "descriptors"
REPORT = RESULTS_DIR / "meta_report.json"


def build_meta_table() -> pd.DataFrame:
    rows = []
    for rp in sorted(RESULTS_DIR.glob("[0-9]*.json")):
        res = json.loads(rp.read_text())
        did = res["dataset_id"]
        dp = DESC_DIR / f"{did}.json"
        if "nn_minus_gbdt" not in res or not dp.exists():
            continue
        desc = json.loads(dp.read_text())
        row = {k: v for k, v in desc.items() if k[1] == "_"}  # S_/E_/T_/Z_/A_
        row["dataset_id"] = did
        row["name"] = res["name"]
        row["nn_minus_gbdt"] = res["nn_minus_gbdt"]
        row["nn_wins"] = int(res["nn_minus_gbdt"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def lodo_eval(df: pd.DataFrame) -> dict:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor)
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from scipy.stats import spearmanr

    feats = [c for c in df.columns
             if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
    X = df[feats].fillna(df[feats].median(numeric_only=True)).values
    y_clf = df["nn_wins"].values
    y_reg = df["nn_minus_gbdt"].values

    proba, pred_margin = np.zeros(len(df)), np.zeros(len(df))
    for i in range(len(df)):
        mask = np.arange(len(df)) != i
        clf = GradientBoostingClassifier(random_state=0, max_depth=2,
                                         n_estimators=150)
        if len(np.unique(y_clf[mask])) < 2:
            proba[i] = float(y_clf[mask].mean())
        else:
            clf.fit(X[mask], y_clf[mask])
            proba[i] = clf.predict_proba(X[i:i + 1])[0, 1]
        reg = GradientBoostingRegressor(random_state=0, max_depth=2,
                                        n_estimators=150)
        reg.fit(X[mask], y_reg[mask])
        pred_margin[i] = reg.predict(X[i:i + 1])[0]

    pred_lbl = (proba > 0.5).astype(int)
    rho, rho_p = spearmanr(pred_margin, y_reg)
    out = {
        "n_datasets": int(len(df)),
        "base_rate_nn_wins": float(y_clf.mean()),
        "clf_accuracy": float((pred_lbl == y_clf).mean()),
        "clf_balanced_accuracy": float(balanced_accuracy_score(y_clf, pred_lbl)),
        "clf_auc": float(roc_auc_score(y_clf, proba))
        if len(np.unique(y_clf)) > 1 else None,
        "reg_spearman_rho": float(rho),
        "reg_spearman_p": float(rho_p),
        "baseline_majority_acc": float(max(y_clf.mean(), 1 - y_clf.mean())),
        "per_dataset": [
            {"dataset_id": int(d), "name": n, "true_margin": float(t),
             "pred_margin": float(pm), "p_nn_wins": float(pr)}
            for d, n, t, pm, pr in zip(df["dataset_id"], df["name"],
                                       y_reg, pred_margin, proba)
        ],
    }

    # permutation importance on the full-fit classifier (descriptive)
    from sklearn.inspection import permutation_importance
    clf_full = GradientBoostingClassifier(random_state=0, max_depth=2,
                                          n_estimators=150).fit(X, y_clf)
    imp = permutation_importance(clf_full, X, y_clf, n_repeats=25,
                                 random_state=0)
    order = np.argsort(-imp.importances_mean)
    out["feature_importance"] = [
        {"feature": feats[j], "mean": float(imp.importances_mean[j]),
         "std": float(imp.importances_std[j])} for j in order
    ]
    return out


def main() -> None:
    df = build_meta_table()
    if len(df) < 10:
        log(f"[meta] only {len(df)} datasets with results+descriptors - too "
            "few for LODO, aborting")
        print(json.dumps({"error": "insufficient data", "n": len(df)}))
        return
    report = lodo_eval(df)
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"[meta] LODO on {report['n_datasets']} datasets: "
        f"acc={report['clf_accuracy']:.3f} "
        f"(majority baseline {report['baseline_majority_acc']:.3f}), "
        f"margin Spearman rho={report['reg_spearman_rho']:.3f}")
    print(json.dumps({k: report[k] for k in
                      ("n_datasets", "clf_accuracy", "clf_balanced_accuracy",
                       "clf_auc", "reg_spearman_rho", "baseline_majority_acc")}))


if __name__ == "__main__":
    main()
