"""H1c: does the meta-regressor's margin PREDICTION survive a permutation
test? (The winner CLASSIFIER failed H1b on the decided subset.)

Statistic: Spearman rho between LODO-predicted and true nn_minus_gbdt over
all 61 datasets. Null: margins permuted across datasets, full LODO re-run.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from config import RESULTS_DIR
from log_utils import log
from meta_learner import build_meta_table

N_PERM = 200
REPORT = RESULTS_DIR / "h1c_report.json"


def lodo_rho(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.ensemble import GradientBoostingRegressor
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        reg = GradientBoostingRegressor(random_state=0, max_depth=2,
                                        n_estimators=150)
        reg.fit(X[mask], y[mask])
        preds[i] = reg.predict(X[i:i + 1])[0]
    return float(spearmanr(preds, y).statistic)


def main() -> None:
    df = build_meta_table()
    feats = [c for c in df.columns
             if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
    X = df[feats].fillna(df[feats].median(numeric_only=True)).values
    y = df["nn_minus_gbdt"].values

    rho = lodo_rho(X, y)
    rng = np.random.default_rng(11)
    null = np.array([lodo_rho(X, rng.permutation(y)) for _ in range(N_PERM)])
    p_val = float((1 + (null >= rho).sum()) / (1 + N_PERM))

    report = {"lodo_spearman_rho": rho,
              "perm_null_mean": float(null.mean()),
              "perm_null_q95": float(np.quantile(null, 0.95)),
              "perm_pvalue": p_val, "n_datasets": int(len(y))}
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"[h1c] LODO margin rho={rho:.3f} null_mean={null.mean():.3f} "
        f"q95={np.quantile(null, 0.95):.3f} p={p_val:.4f}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
