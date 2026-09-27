"""Zero-Error verification agent.

Independent audit of the experimental pipeline. Run after data_prep and
after every training batch. Checks:

  A. Split integrity      - train/val/test indices disjoint, sizes match meta.
  B. Duplicate leakage    - no identical row content crosses split boundaries.
  C. Numeric hygiene      - no NaN/inf after preprocessing transform.
  D. Label-shuffle canary - a model trained on SHUFFLED labels must score at
                            chance on test. Materially above chance => leakage
                            somewhere in the pipeline. Strongest single test.
  E. Results sanity       - every result JSON parses, metrics within bounds.

Exit code 0 = all green; 1 = any check failed (details in the report).
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, ROOT
from log_utils import log

REPORT = ROOT / "results" / "verification_report.json"


def _row_hashes(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    # keep byte-identical with data_prep._row_hashes
    Xs = X.astype(object).where(X.notna(), "<NA>").astype(str)
    joined = Xs.agg("|".join, axis=1) + "||" + y.astype(str)
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def check_dataset(dataset_id: int) -> dict:
    d = DATA_DIR / str(dataset_id)
    meta = json.loads((d / "meta.json").read_text())
    X = pd.read_parquet(d / "X.parquet")
    y = pd.read_parquet(d / "y.parquet")["target"]
    sp = np.load(d / "splits.npz")
    itr, iva, ite = sp["train"], sp["val"], sp["test"]
    findings = {}

    # A. split integrity
    all_idx = np.concatenate([itr, iva, ite])
    findings["A_disjoint"] = bool(len(np.unique(all_idx)) == len(all_idx))
    findings["A_covers_all"] = bool(len(all_idx) == len(X))
    findings["A_sizes_match_meta"] = bool(
        meta["split_sizes"] == {"train": len(itr), "val": len(iva),
                                "test": len(ite)})

    # B. duplicate content across splits
    h = _row_hashes(X, y)
    tr_set = set(h.iloc[itr])
    findings["B_dup_train_val"] = int(sum(x in tr_set for x in h.iloc[iva]))
    findings["B_dup_train_test"] = int(sum(x in tr_set for x in h.iloc[ite]))

    # C. numeric hygiene after preprocessing (fit on train only)
    from models import make_preprocessor
    cat = meta["categorical_cols"]
    num = [c for c in X.columns if c not in cat]
    prep = make_preprocessor(cat, num, onehot=True)
    Xt = prep.fit_transform(X.iloc[itr])
    Xv = prep.transform(X.iloc[ite])
    findings["C_train_finite"] = bool(np.isfinite(np.asarray(Xt, dtype=float)).all())
    findings["C_test_finite"] = bool(np.isfinite(np.asarray(Xv, dtype=float)).all())

    # D. label-shuffle canary (fast LightGBM), statistically calibrated:
    # 5 independent shuffles; the MEDIAN folded AUC must sit within a
    # size-aware tolerance band around chance. A single-shuffle test on a
    # small test split has SE ~ 0.04-0.05 and false-alarms routinely.
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    # Subtlety discovered during Phase 4 QA: a model trained on SHUFFLED
    # labels can still emit x-dependent scores (inductive-bias artifacts,
    # e.g. more extreme scores in sparse regions), and the true labels are
    # also x-dependent - so folded AUC > chance does NOT by itself prove
    # pipeline leakage. The decisive control: hold out part of TRAIN itself.
    # An internal train-holdout (B) cannot possibly be contaminated by a
    # train/test pipeline leak, so:
    #     folded AUC on TEST  >>  folded AUC on B   => genuine leakage
    #     folded AUC on TEST  ~=  folded AUC on B   => innocent x-dependence
    def _folded(proba, y_codes):
        if proba.shape[1] == 2:
            a = roc_auc_score(y_codes, proba[:, 1])
            return max(a, 1 - a)
        return roc_auc_score(y_codes, proba, multi_class="ovr",
                             average="macro")

    y_te = y.iloc[ite]
    test_aucs, ctrl_aucs = [], []
    for cs in range(5):
        rng = np.random.default_rng(cs)
        perm = rng.permutation(len(itr))
        n_a = int(0.7 * len(itr))
        ia, ib = itr[perm[:n_a]], itr[perm[n_a:]]      # A: canary train, B: control
        y_shuf = pd.Series(rng.permutation(y.iloc[ia].values),
                           index=y.iloc[ia].index)
        canary = Pipeline([
            ("prep", make_preprocessor(cat, num, onehot=False)),
            ("clf", LGBMClassifier(n_estimators=100, verbosity=-1,
                                   random_state=cs)),
        ])
        canary.fit(X.iloc[ia], y_shuf)
        cls = list(canary.classes_)
        test_aucs.append(_folded(canary.predict_proba(X.iloc[ite]),
                                 pd.Categorical(y_te, categories=cls).codes))
        ctrl_aucs.append(_folded(canary.predict_proba(X.iloc[ib]),
                                 pd.Categorical(y.iloc[ib],
                                                categories=cls).codes))
    med_test = float(np.median(test_aucs))
    med_ctrl = float(np.median(ctrl_aucs))

    # Size-aware tolerance. Each seed's folded AUC has chance-level
    # SE ~ Hanley-McNeil for its split; the median of 5 seeds has
    # SD ~ 0.56*SE, and the gap subtracts two such medians. A fixed 0.02
    # tolerance false-alarms on small datasets (suite-wide gap distribution
    # is symmetric noise centred at 0: mean -0.001, 29+/31- on 61 datasets).
    def _chance_se(y_split, frac: float = 1.0) -> float:
        counts = pd.Series(y_split).value_counts() * frac
        n1 = float(counts.min())
        n2 = float(counts.sum() - counts.min())
        return float(np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2)))

    # control split B is 30% of train
    se_gap = 0.56 * float(np.hypot(_chance_se(y_te.values),
                                   _chance_se(y.iloc[itr].values, frac=0.3)))
    tol = max(0.02, 3.0 * se_gap)
    findings["D_test_aucs"] = [round(a, 4) for a in test_aucs]
    findings["D_ctrl_aucs"] = [round(a, 4) for a in ctrl_aucs]
    findings["D_median_test"] = round(med_test, 4)
    findings["D_median_ctrl"] = round(med_ctrl, 4)
    findings["D_gap"] = round(med_test - med_ctrl, 4)
    findings["D_tolerance"] = round(tol, 4)
    findings["D_canary_pass"] = bool(med_test - med_ctrl <= tol)

    ok = (findings["A_disjoint"] and findings["A_covers_all"]
          and findings["A_sizes_match_meta"]
          and findings["B_dup_train_val"] == 0
          and findings["B_dup_train_test"] == 0
          and findings["C_train_finite"] and findings["C_test_finite"]
          and findings["D_canary_pass"])
    findings["ALL_OK"] = bool(ok)
    return findings


def check_results() -> dict:
    findings = {"files": 0, "parse_errors": 0, "metric_violations": []}
    for p in sorted(RESULTS_DIR.glob("[0-9]*.json")):
        findings["files"] += 1
        try:
            res = json.loads(p.read_text())
        except json.JSONDecodeError:
            findings["parse_errors"] += 1
            continue
        for fam, r in res.get("families", {}).items():
            if "test_auc_mean" not in r:
                continue
            if not (0.0 <= r["test_auc_mean"] <= 1.0):
                findings["metric_violations"].append(f"{p.name}:{fam}:auc")
            for run in r["test_runs"]:
                if not np.isfinite(list(run.values())).all():
                    findings["metric_violations"].append(f"{p.name}:{fam}:nan")
    findings["OK"] = (findings["parse_errors"] == 0
                      and not findings["metric_violations"])
    return findings


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", type=int, default=[],
                    help="verify only these dataset ids (default: all)")
    args = ap.parse_args()

    report = {"datasets": {}, "results": None}
    all_ok = True
    ids = args.datasets or sorted(int(p.name) for p in DATA_DIR.iterdir()
                                  if (p / "meta.json").exists())
    for did in ids:
        try:
            f = check_dataset(did)
        except Exception as e:  # noqa: BLE001
            f = {"ALL_OK": False, "error": f"{type(e).__name__}: {e}"}
        report["datasets"][did] = f
        all_ok &= f["ALL_OK"]
        log(f"[verify] dataset {did}: {'OK' if f['ALL_OK'] else 'FAIL'} {f}")

    report["results"] = check_results()
    all_ok &= report["results"]["OK"]

    REPORT.write_text(json.dumps(report, indent=2))
    log(f"[verify] overall: {'ALL GREEN' if all_ok else 'FAILURES PRESENT'}")
    print(json.dumps({"all_ok": all_ok,
                      "n_datasets": len(ids),
                      "results_ok": report["results"]["OK"]}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
