"""H14 (EXPLORATORY, not pre-specified): do naturally occurring weak
features relate to the network-GBDT gap?

The manipulations show that ADDED uninformative features widen the gap.
H14 asks whether the same property, measured on the unmodified data, is
associated with (E1) the baseline gap of the MLP and the ResNet (61
datasets; both re-tuned under the identical protocol) and (E2) the gain
from keeping 25% of features in protocol B (P5 datasets).

A feature is "weak" if its mutual information (MI) with the label on the
training split does not exceed the 95th percentile of the MI of the same
feature under shuffled labels (5 shuffles), which removes the positive
bias of the MI estimator. Training rows are subsampled to CAP. Spearman
correlations with two-sided p-values; no multiplicity correction
(exploratory). Output: results/h14_report.json.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from config import RESULTS_DIR
from h13_analysis import gap
from h13_retuned import TIE
from log_utils import log
from train_baseline import _load

REPORT = RESULTS_DIR / "h14_report.json"
CAP = 4000
N_SHUFFLE = 5


def weak_fraction(did: int) -> float:
    meta, X, y, itr, *_ = _load(did)
    cat = meta["categorical_cols"]
    rng = np.random.default_rng(0)
    idx = itr if len(itr) <= CAP else rng.choice(itr, CAP, replace=False)
    Xtr, ytr = X.iloc[idx], y.iloc[idx].values
    num = [c for c in X.columns if c not in cat]
    enc = pd.DataFrame(index=Xtr.index)
    if num:
        # keep_empty_features: an all-missing column stays (as MI = 0)
        enc[num] = SimpleImputer(strategy="median",
                                 keep_empty_features=True).fit_transform(
                                     Xtr[num])
    if cat:
        enc[cat] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-2).fit_transform(Xtr[cat].astype(object))
    Z = enc[X.columns.tolist()].values
    disc = np.array([c in cat for c in X.columns])
    mi = mutual_info_classif(Z, ytr, discrete_features=disc, random_state=0)
    null = np.stack([mutual_info_classif(Z, rng.permutation(ytr),
                                         discrete_features=disc,
                                         random_state=0)
                     for _ in range(N_SHUFFLE)])
    thr = np.quantile(null, 0.95, axis=0)
    return float((mi <= thr).mean())


def corr(x: dict, y: dict) -> dict:
    keys = [k for k in x if k in y and y[k] is not None]
    r = spearmanr([x[k] for k in keys], [y[k] for k in keys])
    return {"n": len(keys), "rho": float(r.statistic),
            "p_two_sided": float(r.pvalue)}


def main() -> None:
    reg = json.loads((RESULTS_DIR / "h13_prereg.json").read_text())
    ids = reg["datasets"]
    wf = {}
    for d in ids:
        wf[d] = weak_fraction(d)
        log(f"[h14] {d}: weak fraction {wf[d]:.3f}")
    base = {d: json.loads((RESULTS_DIR / f"{d}.json").read_text())
            ["nn_minus_gbdt"] for d in ids}
    fav = [d for d in ids if base[d] <= -TIE]
    rep = {"weak_fraction": {str(d): v for d, v in wf.items()},
           "weak_fraction_median": float(np.median(list(wf.values()))),
           "n_datasets_with_weak_ge_25pct": int(sum(v >= 0.25
                                                    for v in wf.values())),
           "E1": {}, "E2": {}}
    for net in ("mlp", "resnet"):
        g0 = {d: gap(d, "base", 0, net) for d in ids}
        rep["E1"][net] = corr(wf, g0)
        p5 = {d: (gap(d, "select_features", 0.25, net) - g0[d])
              if gap(d, "select_features", 0.25, net) is not None
              and g0[d] is not None else None for d in fav}
        rep["E2"][net] = corr({d: wf[d] for d in fav}, p5)
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "weak_fraction"},
                     indent=1))


if __name__ == "__main__":
    main()
