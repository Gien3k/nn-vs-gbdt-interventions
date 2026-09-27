"""H1b: is the meta-signal real? Decided-subset LODO + permutation test.

Two refinements over the first-pass H1:
  1. Near-tie datasets (|nn_minus_gbdt| < TIE_EPS) have coin-flip winner
     labels; H1b evaluates on the DECIDED subset where the label is
     meaningful (the tie threshold equals the typical seed-level noise).
  2. Statistical rigour: a permutation test - winner labels are shuffled
     across datasets and the FULL LODO procedure is re-run each time,
     giving the exact null distribution of LODO accuracy.
"""
import json

import numpy as np

from config import RESULTS_DIR
from log_utils import log
from meta_learner import build_meta_table

TIE_EPS = 0.005
N_PERM = 200
REPORT = RESULTS_DIR / "h1b_report.json"


def lodo_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.ensemble import GradientBoostingClassifier
    correct = 0
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        if len(np.unique(y[mask])) < 2:
            pred = int(y[mask].mean() > 0.5)
        else:
            clf = GradientBoostingClassifier(random_state=0, max_depth=2,
                                             n_estimators=150)
            clf.fit(X[mask], y[mask])
            pred = int(clf.predict(X[i:i + 1])[0])
        correct += int(pred == y[i])
    return correct / len(y)


def main() -> None:
    df = build_meta_table()
    feats = [c for c in df.columns
             if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
    dec = df[df["nn_minus_gbdt"].abs() >= TIE_EPS].reset_index(drop=True)
    Xd = dec[feats].fillna(dec[feats].median(numeric_only=True)).values
    yd = dec["nn_wins"].values

    acc = lodo_accuracy(Xd, yd)
    base = max(yd.mean(), 1 - yd.mean())

    rng = np.random.default_rng(7)
    null = np.array([lodo_accuracy(Xd, rng.permutation(yd))
                     for _ in range(N_PERM)])
    p_val = float((1 + (null >= acc).sum()) / (1 + N_PERM))

    report = {
        "tie_eps": TIE_EPS,
        "n_decided": int(len(dec)),
        "n_total": int(len(df)),
        "decided_base_rate_nn": float(yd.mean()),
        "lodo_accuracy_decided": float(acc),
        "majority_baseline_decided": float(base),
        "perm_null_mean": float(null.mean()),
        "perm_null_q95": float(np.quantile(null, 0.95)),
        "perm_pvalue": p_val,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"[h1b] decided={len(dec)}/{len(df)} LODO acc={acc:.3f} "
        f"(baseline {base:.3f}, null mean {null.mean():.3f}, "
        f"q95 {np.quantile(null, 0.95):.3f}) perm p={p_val:.4f}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
