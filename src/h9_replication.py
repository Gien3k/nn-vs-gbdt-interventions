"""H9: out-of-sample replication of the causal-alignment result (H5).

PRE-REGISTERED: on intervention cells from datasets NOT in the n=12 pilot
(i.e., the H6 extension cohort, never used in the H5 analysis), the LODO
meta-regressor's predicted margin changes correlate positively with observed
changes (Spearman rho > 0, permutation test, alpha=0.05).

Also reported (exploratory): alignment on the H7 select_features cells.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_DIR, RESULTS_DIR
from h5_mechanism import desc_vector
from h7_rescue import mi_ranking
from interventions import transform
from log_utils import log
from meta_learner import build_meta_table

INT_DIR = RESULTS_DIR / "interventions"
PILOT_FILE = RESULTS_DIR / "pilot_ids.json"
REPORT = RESULTS_DIR / "h9_report.json"
N_PERM = 2000


def main() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    df = build_meta_table()
    feats = [c for c in df.columns
             if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
    med = df[feats].median(numeric_only=True)
    X_meta = df[feats].fillna(med).values
    y_meta = df["nn_minus_gbdt"].values
    ids = list(df["dataset_id"])
    pilot = set(json.loads(PILOT_FILE.read_text()))

    cells = [json.loads(p.read_text()) for p in INT_DIR.glob("*.json")]
    reg_cache, orig_pred = {}, {}
    rows = []
    for c in sorted(cells, key=lambda c: (c["dataset_id"], c["kind"],
                                          str(c["dose"]))):
        did = c["dataset_id"]
        if did not in ids:
            continue
        is_pilot = did in pilot
        if is_pilot and c["kind"] != "select_features":
            continue          # pilot rotate/quantize/noise already in H5
        i = ids.index(did)
        if did not in reg_cache:
            mask = np.arange(len(ids)) != i
            reg = GradientBoostingRegressor(random_state=0, max_depth=2,
                                            n_estimators=150)
            reg.fit(X_meta[mask], y_meta[mask])
            reg_cache[did] = reg
            orig_pred[did] = float(reg.predict(X_meta[i:i + 1])[0])
        reg = reg_cache[did]

        d = DATA_DIR / str(did)
        meta = json.loads((d / "meta.json").read_text())
        X = pd.read_parquet(d / "X.parquet")
        y = pd.read_parquet(d / "y.parquet")["target"]
        itr = np.load(d / "splits.npz")["train"]
        cat = meta["categorical_cols"]
        num = [col for col in X.columns if col not in cat]

        if c["kind"] == "select_features":
            ranked = mi_ranking(X, y, itr, cat)
            k = max(2, int(round(c["dose"] * len(ranked))))
            Xt = X[ranked[:k]]
            cat_t = [x for x in ranked[:k] if x in cat]
        else:
            Xt = transform(X, itr, num, c["kind"], c["dose"])
            cat_t = cat
        dv = desc_vector(Xt, y, itr, cat_t, feats)
        vec = np.array([[dv[f] if np.isfinite(dv[f]) else med[f]
                         for f in feats]])
        rows.append({
            "dataset_id": did, "kind": c["kind"], "dose": c["dose"],
            "cohort": "pilot" if is_pilot else "extension",
            "delta_obs": float(c["nn_minus_gbdt"] - c["baseline_margin"]),
            "delta_pred": float(reg.predict(vec)[0] - orig_pred[did]),
        })
        log(f"[h9] {did}/{c['kind']}@{c['dose']}: "
            f"obs={rows[-1]['delta_obs']:+.4f} "
            f"pred={rows[-1]['delta_pred']:+.4f}")

    rng = np.random.default_rng(13)

    def rho_p(sub):
        o = np.array([r["delta_obs"] for r in sub])
        pr = np.array([r["delta_pred"] for r in sub])
        rho = float(spearmanr(pr, o).statistic)
        null = [float(spearmanr(pr, rng.permutation(o)).statistic)
                for _ in range(N_PERM)]
        p = float((1 + sum(n >= rho for n in null)) / (1 + N_PERM))
        return {"n": len(sub), "spearman_rho": rho, "perm_p": p,
                "sign_agreement": f"{int(((o * pr) > 0).sum())}/{len(sub)}"}

    ext = [r for r in rows
           if r["cohort"] == "extension" and r["kind"] != "select_features"]
    sel = [r for r in rows if r["kind"] == "select_features"]
    rep = {"primary_extension_replication": rho_p(ext) if len(ext) >= 10
           else None,
           "exploratory_select_features": rho_p(sel) if len(sel) >= 10
           else None,
           "rows": rows}
    REPORT.write_text(json.dumps(rep, indent=2))
    log(f"[h9] extension replication: {rep['primary_extension_replication']}")
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
