"""H2s: sensitivity check for the headline noise-features result.

Objection to rule out: "the MLP collapse under noise injection is an
artifact of freezing baseline-tuned configs". Here each family is RE-TUNED
(10 random configs, train->val) on the TRANSFORMED data at maximum dose,
then evaluated on the transformed test (3 seeds). If margins still collapse,
the causal claim is robust to hyperparameter adaptation.
"""
import json

import numpy as np

from config import RESULTS_DIR, TUNING_SEED
from interventions import transform
from log_utils import log
from models import build_model, sample_config
from train_baseline import _load, _score

REPORT = RESULTS_DIR / "h2s_report.json"
N_CFG = 10
SEEDS = [0, 1, 2]
DOSE = 4.0
N_DATASETS = 4


def main() -> None:
    h2 = json.loads((RESULTS_DIR / "h2_report.json").read_text())
    rows = sorted(h2["noise_features"]["per_dataset"],
                  key=lambda r: r["delta_at_max"])[:N_DATASETS]
    out = []
    for row in rows:
        did = row["dataset_id"]
        meta, X, y, itr, iva, ite = _load(did)
        cat = meta["categorical_cols"]
        num = [c for c in X.columns if c not in cat]
        Xt = transform(X, itr, num, "noise_features", DOSE)
        num_t = [c for c in Xt.columns if c not in cat]

        fams = {}
        rng = np.random.default_rng(TUNING_SEED)
        for fam in ("mlp", "xgb", "lgbm"):
            trials = []
            for _ in range(N_CFG):
                cfg = sample_config(fam, rng)
                try:
                    pipe = build_model(fam, cfg, cat, num_t, seed=TUNING_SEED)
                    pipe.fit(Xt.iloc[itr], y.iloc[itr])
                    v = _score(pipe, Xt.iloc[iva], y.iloc[iva])["roc_auc_ovr"]
                    trials.append((v, cfg))
                except Exception as e:  # noqa: BLE001
                    log(f"[h2s] {did}/{fam} trial ERROR {type(e).__name__}: {e}")
            best_cfg = max(trials)[1]
            aucs = []
            for s in SEEDS:
                pipe = build_model(fam, best_cfg, cat, num_t, seed=s)
                pipe.fit(Xt.iloc[itr], y.iloc[itr])
                aucs.append(_score(pipe, Xt.iloc[ite], y.iloc[ite])["roc_auc_ovr"])
            fams[fam] = float(np.mean(aucs))
        margin_retuned = fams["mlp"] - max(fams["xgb"], fams["lgbm"])
        out.append({
            "dataset_id": did,
            "baseline_margin": row["baseline"],
            "frozen_cfg_margin_at_max": row["baseline"] + row["delta_at_max"],
            "retuned_margin_at_max": float(margin_retuned),
        })
        log(f"[h2s] {did}: baseline={row['baseline']:+.4f} "
            f"frozen={out[-1]['frozen_cfg_margin_at_max']:+.4f} "
            f"retuned={margin_retuned:+.4f}")

    deltas_retuned = [r["retuned_margin_at_max"] - r["baseline_margin"]
                      for r in out]
    rep = {"dose": DOSE, "n_cfg": N_CFG, "rows": out,
           "mean_delta_retuned": float(np.mean(deltas_retuned)),
           "all_still_negative": bool(all(d < 0 for d in deltas_retuned))}
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
