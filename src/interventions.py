"""H2/H4: causal interventions on dataset geometry with dose-response.

PRE-REGISTERED PREDICTIONS (fixed before running, informed by H1
importances - T_mi_max / A_stump_auc_mean / T_linear_auc drive the winner):

  I1 rotate(alpha):  random orthogonal rotation of standardized numeric
     features destroys axis-aligned single-feature signal (the GBDT edge).
     PREDICTION: margin (nn_minus_gbdt) INCREASES with alpha.
  I2 quantize(bins): quantile-binning numeric features destroys smooth
     structure that MLPs interpolate; trees are near-invariant to monotone
     per-feature transforms. PREDICTION: margin DECREASES as bins shrink.
  I3 noise_features(mult): appending mult*p Gaussian noise features hurts
     MLPs more than GBDTs (Grinsztajn et al. 2022).
     PREDICTION: margin DECREASES with mult.

Dose 0 always equals the untransformed baseline (reused from results/).
Protocol per (dataset, intervention, dose): re-train each family's BEST
baseline config (3 seeds), evaluate on the identically-transformed test
split. Transform parameters are fitted on TRAIN only. Analysis:
per-intervention Wilcoxon signed-rank on (margin@max_dose - margin@0)
across datasets + monotonic dose-response (Page-like trend via Spearman).

Limitation (documented for the paper): configs are not re-tuned per dose;
sensitivity check re-tunes on a subset for the headline intervention.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.linalg import expm

import state
from config import DATA_DIR, RESULTS_DIR
from log_utils import log
from models import build_model
from train_baseline import _load, _score

INT_DIR = RESULTS_DIR / "interventions"
INT_DIR.mkdir(parents=True, exist_ok=True)

TIE_EPS = 0.005
SEEDS = [0, 1, 2]
DOSES = {
    "rotate": [0.25, 0.5, 1.0, 2.0],       # skew-exponential angle scale
    "quantize": [32, 8, 4, 2],             # bins (descending severity ->)
    "noise_features": [0.5, 1.0, 2.0, 4.0],  # multiples of p appended
}


def pick_datasets(max_n: int = 12) -> list[int]:
    """Decided datasets, mostly-numeric, small enough for many refits."""
    chosen = []
    for p in sorted(RESULTS_DIR.glob("[0-9]*.json")):
        r = json.loads(p.read_text())
        if "nn_minus_gbdt" not in r or abs(r["nn_minus_gbdt"]) < TIE_EPS:
            continue
        meta = json.loads((DATA_DIR / str(r["dataset_id"]) / "meta.json")
                          .read_text())
        frac_cat = len(meta["categorical_cols"]) / meta["n_features"]
        if meta["n_rows"] > 20000 or meta["n_features"] > 100 or frac_cat > 0.3:
            continue
        chosen.append((abs(r["nn_minus_gbdt"]), r["dataset_id"],
                       r["nn_minus_gbdt"]))
    chosen.sort(reverse=True)
    # balance: up to half from each side of the margin where available
    gbdt_side = [c for c in chosen if c[2] < 0][:max_n // 2]
    nn_side = [c for c in chosen if c[2] > 0][:max_n - len(gbdt_side)]
    rest = [c for c in chosen if c not in gbdt_side + nn_side]
    sel = gbdt_side + nn_side
    sel += rest[:max_n - len(sel)]
    return [c[1] for c in sel]


def transform(X: pd.DataFrame, itr: np.ndarray, num_cols: list[str],
              kind: str, dose: float, seed: int = 0) -> pd.DataFrame:
    """Apply intervention; all fitted statistics come from TRAIN rows only."""
    X = X.copy()
    rng = np.random.default_rng(seed)
    Xn = X[num_cols].astype(float)
    mu = Xn.iloc[itr].mean()
    sd = Xn.iloc[itr].std().replace(0, 1.0)

    if kind == "rotate":
        Z = ((Xn - mu) / sd).fillna(0.0).values
        p = len(num_cols)
        G = rng.standard_normal((p, p))
        A = (G - G.T) / np.sqrt(2 * p)      # skew-symmetric, scale-stable
        R = expm(dose * A)                   # orthogonal
        X[num_cols] = Z @ R
    elif kind == "quantize":
        bins = int(dose)
        for c in num_cols:
            edges = np.unique(np.quantile(Xn[c].iloc[itr].dropna(),
                                          np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                continue
            mids = (edges[:-1] + edges[1:]) / 2
            idx = np.clip(np.searchsorted(edges[1:-1], Xn[c].values), 0,
                          len(mids) - 1)
            vals = mids[idx].astype(float)
            vals[Xn[c].isna().values] = np.nan
            X[c] = vals
    elif kind == "noise_features":
        k = max(1, int(round(dose * X.shape[1])))
        scale_ref = float(sd.mean()) if len(num_cols) else 1.0
        noise = rng.standard_normal((len(X), k)) * scale_ref
        for j in range(k):
            X[f"__noise_{j}"] = noise[:, j]
    else:
        raise ValueError(kind)
    return X


def run_cell(dataset_id: int, kind: str, dose: float) -> dict | None:
    out_path = INT_DIR / f"{dataset_id}_{kind}_{dose}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    meta, X, y, itr, iva, ite = _load(dataset_id)
    base = json.loads((RESULTS_DIR / f"{dataset_id}.json").read_text())
    cat = meta["categorical_cols"]
    num = [c for c in X.columns if c not in cat]
    Xt = transform(X, itr, num, kind, dose)
    num_t = [c for c in Xt.columns if c not in cat]

    fams = {}
    for fam in ("mlp", "xgb", "lgbm"):
        cfg = base["families"][fam]["best_cfg"]
        aucs = []
        for s in SEEDS:
            pipe = build_model(fam, cfg, cat, num_t, seed=s)
            pipe.fit(Xt.iloc[itr], y.iloc[itr])
            aucs.append(_score(pipe, Xt.iloc[ite], y.iloc[ite])["roc_auc_ovr"])
        fams[fam] = {"test_auc_mean": float(np.mean(aucs)),
                     "test_auc_std": float(np.std(aucs))}
    nn = fams["mlp"]["test_auc_mean"]
    gbdt = max(fams["xgb"]["test_auc_mean"], fams["lgbm"]["test_auc_mean"])
    cell = {"dataset_id": dataset_id, "kind": kind, "dose": dose,
            "families": fams, "nn_minus_gbdt": float(nn - gbdt),
            "baseline_margin": base["nn_minus_gbdt"]}
    out_path.write_text(json.dumps(cell, indent=2))
    log(f"[interv] {dataset_id}/{kind}@{dose}: margin={nn - gbdt:+.4f} "
        f"(baseline {base['nn_minus_gbdt']:+.4f})")
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-datasets", type=int, default=12)
    args = ap.parse_args()
    ids = pick_datasets(args.max_datasets)
    log(f"[interv] selected datasets: {ids}")
    state.update(current_hypothesis=(
        "H2: interventions move nn_minus_gbdt in PRE-REGISTERED directions "
        "(rotate: +, quantize: -, noise_features: -), dose-monotonically"),
        note=f"H2 interventions started on {len(ids)} datasets")
    n_cells = 0
    for kind, doses in DOSES.items():
        for dose in doses:
            for did in ids:
                try:
                    run_cell(did, kind, dose)
                    n_cells += 1
                except Exception as e:  # noqa: BLE001
                    log(f"[interv] {did}/{kind}@{dose} ERROR "
                        f"{type(e).__name__}: {e}")
    log(f"[interv] done: {n_cells} cells")
    print(json.dumps({"datasets": ids, "cells": n_cells}))


if __name__ == "__main__":
    main()
