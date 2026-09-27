"""Baseline comparison: tuned MLP vs XGBoost vs LightGBM per dataset.

Protocol (identical for every family - budget fairness is a core validity
requirement of the study):
  1. Random-search N_TUNING_CONFIGS configs, trained on TRAIN, scored on VAL.
  2. Best config re-trained on TRAIN with FINAL_EVAL_SEEDS seeds,
     scored ONCE per seed on TEST. TEST is touched only at this stage.
  3. Results appended incrementally to results/<dataset_id>.json;
     discovery_state.json updated after every dataset (crash-safe resume).

Usage:
  python train_baseline.py --smoke            # smoke suite, reduced budget
  python train_baseline.py --datasets 31 37   # explicit ids
  python train_baseline.py --full             # every prepared dataset
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import state
from config import (DATA_DIR, FINAL_EVAL_SEEDS, N_TUNING_CONFIGS, RESULTS_DIR,
                    TUNING_SEED)
from log_utils import log
from models import build_model, sample_config

FAMILIES = ["mlp", "xgb", "lgbm"]


def _load(dataset_id: int):
    d = DATA_DIR / str(dataset_id)
    meta = json.loads((d / "meta.json").read_text())
    X = pd.read_parquet(d / "X.parquet")
    y = pd.read_parquet(d / "y.parquet")["target"]
    sp = np.load(d / "splits.npz")
    return meta, X, y, sp["train"], sp["val"], sp["test"]


def _score(pipe, X, y) -> dict:
    proba = pipe.predict_proba(X)
    classes = list(pipe.classes_)
    y_idx = pd.Categorical(y, categories=classes).codes
    if proba.shape[1] == 2:
        auc = roc_auc_score(y_idx, proba[:, 1])
    else:
        auc = roc_auc_score(y_idx, proba, multi_class="ovr", average="macro")
    return {
        "roc_auc_ovr": float(auc),
        "accuracy": float(accuracy_score(y_idx, proba.argmax(1))),
        "log_loss": float(log_loss(y_idx, proba, labels=range(len(classes)))),
    }


def run_dataset(dataset_id: int, n_configs: int, seeds: list[int]) -> dict:
    meta, X, y, itr, iva, ite = _load(dataset_id)
    cat = meta["categorical_cols"]
    num = [c for c in X.columns if c not in cat]
    out = {"dataset_id": dataset_id, "name": meta["name"], "families": {}}
    rng = np.random.default_rng(TUNING_SEED)

    for fam in FAMILIES:
        t0 = time.time()
        trials = []
        for k in range(n_configs):
            cfg = sample_config(fam, rng)
            try:
                pipe = build_model(fam, cfg, cat, num, seed=TUNING_SEED)
                pipe.fit(X.iloc[itr], y.iloc[itr])
                val = _score(pipe, X.iloc[iva], y.iloc[iva])
                trials.append({"cfg": cfg, "val": val})
                log(f"[train] {dataset_id}/{fam} trial {k+1}/{n_configs} "
                    f"val_auc={val['roc_auc_ovr']:.4f}")
            except Exception as e:  # noqa: BLE001
                log(f"[train] {dataset_id}/{fam} trial {k+1} ERROR "
                    f"{type(e).__name__}: {e}")
        if not trials:
            out["families"][fam] = {"error": "all trials failed"}
            continue
        best = max(trials, key=lambda t: t["val"]["roc_auc_ovr"])

        # final multi-seed evaluation on the untouched TEST split
        test_runs = []
        for s in seeds:
            pipe = build_model(fam, best["cfg"], cat, num, seed=s)
            pipe.fit(X.iloc[itr], y.iloc[itr])
            test_runs.append(_score(pipe, X.iloc[ite], y.iloc[ite]))
        aucs = [r["roc_auc_ovr"] for r in test_runs]
        out["families"][fam] = {
            "best_cfg": best["cfg"],
            "val_auc": best["val"]["roc_auc_ovr"],
            "test_runs": test_runs,
            "test_auc_mean": float(np.mean(aucs)),
            "test_auc_std": float(np.std(aucs)),
            "n_trials_ok": len(trials),
            "tune_seconds": round(time.time() - t0, 1),
        }
        log(f"[train] {dataset_id}/{fam} DONE test_auc="
            f"{np.mean(aucs):.4f}+-{np.std(aucs):.4f} "
            f"({time.time()-t0:.0f}s)")

    fams_ok = {f: r for f, r in out["families"].items() if "test_auc_mean" in r}
    if fams_ok:
        winner = max(fams_ok, key=lambda f: fams_ok[f]["test_auc_mean"])
        nn_auc = fams_ok.get("mlp", {}).get("test_auc_mean")
        gbdt_auc = max((fams_ok[f]["test_auc_mean"] for f in ("xgb", "lgbm")
                        if f in fams_ok), default=None)
        out["winner"] = winner
        if nn_auc is not None and gbdt_auc is not None:
            out["nn_minus_gbdt"] = float(nn_auc - gbdt_auc)
    (RESULTS_DIR / f"{dataset_id}.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--datasets", nargs="*", type=int, default=[])
    args = ap.parse_args()

    if args.smoke:
        from config import SMOKE_SUITE
        ids, n_cfg, seeds = SMOKE_SUITE, 4, FINAL_EVAL_SEEDS[:2]
    elif args.datasets:
        ids, n_cfg, seeds = args.datasets, N_TUNING_CONFIGS, FINAL_EVAL_SEEDS
    elif args.full:
        ids = sorted(int(p.name) for p in DATA_DIR.iterdir()
                     if (p / "meta.json").exists())
        n_cfg, seeds = N_TUNING_CONFIGS, FINAL_EVAL_SEEDS
    else:
        ap.error("choose --smoke, --datasets or --full")

    done = set(state.load()["completed_datasets"])
    todo = [d for d in ids if d not in done]
    log(f"[train] starting: {len(todo)} datasets to run, {len(done)} already done")

    for did in todo:
        try:
            res = run_dataset(did, n_cfg, seeds)
            state.mark_dataset_done(
                did, note=f"dataset {did} done, winner={res.get('winner')}")
        except Exception as e:  # noqa: BLE001
            state.mark_dataset_failed(did, f"{type(e).__name__}: {e}")
            log(f"[train] {did} FAILED {type(e).__name__}: {e}")

    summary = {"completed": len(state.load()["completed_datasets"]),
               "failed": len(state.load()["failed_datasets"])}
    log(f"[train] finished: {summary}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
