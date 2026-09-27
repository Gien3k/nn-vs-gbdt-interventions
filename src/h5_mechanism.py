"""H5: causal alignment of the meta-regressor.

Question: does the H1c meta-regressor (trained only on UNTRANSFORMED
datasets, LODO discipline) predict the margin CHANGES that interventions
actually cause? If yes, the descriptor->margin mapping is not a mere
correlation: moving a dataset along descriptor axes moves the margin as
the model says it should.

For every intervention cell (dataset, kind, dose):
  delta_obs  = margin(cell) - margin(baseline)
  delta_pred = reg_-d(desc(transformed train)) - reg_-d(desc(original train))
where reg_-d is trained on the other 60 baseline datasets (LODO).

Endpoints: Spearman(delta_pred, delta_obs) over cells, per intervention and
pooled, with a permutation test (delta_obs shuffled within intervention).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_DIR, RESULTS_DIR
from descriptors import (axis_features, simple_features, spectral_features,
                         target_features, zerocost_features)
from interventions import transform
from log_utils import log
from meta_learner import build_meta_table
from models import make_preprocessor

INT_DIR = RESULTS_DIR / "interventions"
REPORT = RESULTS_DIR / "h5_report.json"
N_PERM = 2000


def desc_vector(X: pd.DataFrame, y: pd.Series, itr: np.ndarray,
                cat: list[str], feats: list[str]) -> dict:
    num = [c for c in X.columns if c not in cat]
    prep = make_preprocessor(cat, num, onehot=True)
    Xt = np.asarray(prep.fit_transform(X.iloc[itr]), dtype=np.float64)
    _, y_idx = np.unique(y.iloc[itr], return_inverse=True)
    d = {}
    d.update(simple_features(X.iloc[itr], y.iloc[itr], cat))
    d.update(spectral_features(Xt))
    d.update(target_features(Xt, y_idx))
    d.update(zerocost_features(Xt, y_idx))
    d.update(axis_features(Xt, y_idx))
    return {f: d.get(f, np.nan) for f in feats}


def main() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    df = build_meta_table()
    feats = [c for c in df.columns
             if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
    med = df[feats].median(numeric_only=True)
    X_meta = df[feats].fillna(med).values
    y_meta = df["nn_minus_gbdt"].values
    ids = list(df["dataset_id"])

    cells = [json.loads(p.read_text()) for p in INT_DIR.glob("*.json")]
    rows = []
    reg_cache: dict[int, GradientBoostingRegressor] = {}
    orig_pred_cache: dict[int, float] = {}

    for c in sorted(cells, key=lambda c: (c["dataset_id"], c["kind"], c["dose"])):
        did = c["dataset_id"]
        if did not in ids:
            continue
        i = ids.index(did)
        if did not in reg_cache:
            mask = np.arange(len(ids)) != i
            reg = GradientBoostingRegressor(random_state=0, max_depth=2,
                                            n_estimators=150)
            reg.fit(X_meta[mask], y_meta[mask])
            reg_cache[did] = reg
            orig_pred_cache[did] = float(reg.predict(X_meta[i:i + 1])[0])
        reg = reg_cache[did]

        d = DATA_DIR / str(did)
        meta = json.loads((d / "meta.json").read_text())
        X = pd.read_parquet(d / "X.parquet")
        y = pd.read_parquet(d / "y.parquet")["target"]
        itr = np.load(d / "splits.npz")["train"]
        cat = meta["categorical_cols"]
        num = [col for col in X.columns if col not in cat]

        Xt = transform(X, itr, num, c["kind"], c["dose"])
        dv = desc_vector(Xt, y, itr, cat, feats)
        vec = np.array([[dv[f] if np.isfinite(dv[f]) else med[f]
                         for f in feats]])
        pred_t = float(reg.predict(vec)[0])
        rows.append({
            "dataset_id": did, "kind": c["kind"], "dose": c["dose"],
            "delta_obs": float(c["nn_minus_gbdt"] - c["baseline_margin"]),
            "delta_pred": float(pred_t - orig_pred_cache[did]),
        })
        log(f"[h5] {did}/{c['kind']}@{c['dose']}: "
            f"d_obs={rows[-1]['delta_obs']:+.4f} "
            f"d_pred={rows[-1]['delta_pred']:+.4f}")

    rep = {"n_cells": len(rows), "per_kind": {}, "rows": rows}
    rng = np.random.default_rng(3)

    def rho_p(obs, pred):
        rho = float(spearmanr(pred, obs).statistic)
        null = [float(spearmanr(pred, rng.permutation(obs)).statistic)
                for _ in range(N_PERM)]
        p = float((1 + sum(n >= rho for n in null)) / (1 + N_PERM))
        return rho, p

    all_obs = np.array([r["delta_obs"] for r in rows])
    all_pred = np.array([r["delta_pred"] for r in rows])
    rho, p = rho_p(all_obs, all_pred)
    sign_ok = int(((all_obs * all_pred) > 0).sum())
    rep["pooled"] = {"spearman_rho": rho, "perm_p": p,
                     "sign_agreement": f"{sign_ok}/{len(rows)}"}
    for kind in ("rotate", "quantize", "noise_features"):
        sub = [r for r in rows if r["kind"] == kind]
        o = np.array([r["delta_obs"] for r in sub])
        pr = np.array([r["delta_pred"] for r in sub])
        r_, p_ = rho_p(o, pr)
        rep["per_kind"][kind] = {
            "n": len(sub), "spearman_rho": r_, "perm_p": p_,
            "sign_agreement": f"{int(((o * pr) > 0).sum())}/{len(sub)}"}

    REPORT.write_text(json.dumps(rep, indent=2))
    log(f"[h5] pooled rho={rho:.3f} p={p:.4f} sign={sign_ok}/{len(rows)}")
    print(json.dumps({"pooled": rep["pooled"], "per_kind": rep["per_kind"]},
                     indent=2))


if __name__ == "__main__":
    main()
