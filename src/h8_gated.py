"""H8: architectural defence - gated MLP under noise injection.

PRE-REGISTERED (fixed before execution):
  Model: the baseline MLP augmented with a learnable sigmoid input gate and
  an L1 penalty on gate openness (soft feature selection); penalty strength
  selected on the validation split from {1e-3, 1e-2, 1e-1}.
  PRIMARY endpoint: across the pilot noise datasets, the gated MLP's AUC
  drop from dose 0 to dose 4x is SMALLER than the plain MLP's drop
  (paired one-sided Wilcoxon on robustness = drop_plain - drop_gated > 0).
  SECONDARY: at dose 4x the gated margin vs best GBDT exceeds the plain
  margin (partial rescue of competitiveness).
"""
import json

import numpy as np
from scipy.stats import wilcoxon
from sklearn.pipeline import Pipeline

from config import RESULTS_DIR, TUNING_SEED
from interventions import DOSES, transform
from log_utils import log
from models import TorchMLP, make_preprocessor
from train_baseline import _load, _score

INT_DIR = RESULTS_DIR / "interventions"
REPORT = RESULTS_DIR / "h8_report.json"
LAMBDAS = [1e-3, 1e-2, 1e-1]
SEEDS = [0, 1, 2]
NOISE_DOSES = [0.0] + DOSES["noise_features"]


def gated_auc(X, y, itr, iva, ite, cat, num, cfg) -> float:
    best_lam, best_val = None, -np.inf
    for lam in LAMBDAS:
        pipe = Pipeline([
            ("prep", make_preprocessor(cat, num, onehot=True)),
            ("clf", TorchMLP(seed=TUNING_SEED, gate_l1=lam, **cfg)),
        ])
        pipe.fit(X.iloc[itr], y.iloc[itr])
        v = _score(pipe, X.iloc[iva], y.iloc[iva])["roc_auc_ovr"]
        if v > best_val:
            best_val, best_lam = v, lam
    aucs = []
    for s in SEEDS:
        pipe = Pipeline([
            ("prep", make_preprocessor(cat, num, onehot=True)),
            ("clf", TorchMLP(seed=s, gate_l1=best_lam, **cfg)),
        ])
        pipe.fit(X.iloc[itr], y.iloc[itr])
        aucs.append(_score(pipe, X.iloc[ite], y.iloc[ite])["roc_auc_ovr"])
    return float(np.mean(aucs))


def main() -> None:
    # pilot datasets = those with existing noise cells at every dose
    pilot = sorted({json.loads(p.read_text())["dataset_id"]
                    for p in INT_DIR.glob("*_noise_features_4.0.json")})
    log(f"[h8] gated-MLP defence on {len(pilot)} pilot noise datasets")
    cache_path = RESULTS_DIR / "h8_cells.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = []
    for did in pilot:
        meta, X, y, itr, iva, ite = _load(did)
        base = json.loads((RESULTS_DIR / f"{did}.json").read_text())
        cfg = base["families"]["mlp"]["best_cfg"]
        cat = meta["categorical_cols"]
        num = [c for c in X.columns if c not in cat]
        gated = {}
        for dose in NOISE_DOSES:
            key = f"{did}@{dose}"
            if key in cache:
                gated[dose] = cache[key]
                continue
            Xt = X if dose == 0 else transform(X, itr, num,
                                               "noise_features", dose)
            num_t = [c for c in Xt.columns if c not in cat]
            gated[dose] = gated_auc(Xt, y, itr, iva, ite, cat, num_t, cfg)
            cache[key] = gated[dose]
            cache_path.write_text(json.dumps(cache, indent=2))
            log(f"[h8] {did}/gated@{dose}: auc={gated[dose]:.4f}")
        plain0 = base["families"]["mlp"]["test_auc_mean"]
        cell4 = json.loads(
            (INT_DIR / f"{did}_noise_features_4.0.json").read_text())
        plain4 = cell4["families"]["mlp"]["test_auc_mean"]
        gbdt4 = max(cell4["families"]["xgb"]["test_auc_mean"],
                    cell4["families"]["lgbm"]["test_auc_mean"])
        rows.append({
            "dataset_id": did,
            "plain_drop": float(plain0 - plain4),
            "gated_drop": float(gated[0.0] - gated[4.0]),
            "robustness_gain": float((plain0 - plain4)
                                     - (gated[0.0] - gated[4.0])),
            "margin_plain_at4": float(plain4 - gbdt4),
            "margin_gated_at4": float(gated[4.0] - gbdt4),
            "gated_by_dose": gated,
        })

    gains = np.array([r["robustness_gain"] for r in rows])
    rescue = np.array([r["margin_gated_at4"] - r["margin_plain_at4"]
                       for r in rows])
    rep = {
        "n_datasets": int(len(rows)),
        "primary_mean_robustness_gain": float(gains.mean()),
        "primary_n_direction_ok": int((gains > 0).sum()),
        "primary_wilcoxon_p_onesided": float(
            wilcoxon(gains, alternative="greater").pvalue),
        "secondary_mean_margin_rescue_at4": float(rescue.mean()),
        "secondary_wilcoxon_p_onesided": float(
            wilcoxon(rescue, alternative="greater").pvalue),
        "per_dataset": rows,
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    log(f"[h8] robustness gain mean={gains.mean():+.4f} "
        f"dir={int((gains > 0).sum())}/{len(rows)} "
        f"p={rep['primary_wilcoxon_p_onesided']:.4f}; "
        f"rescue mean={rescue.mean():+.4f} "
        f"p={rep['secondary_wilcoxon_p_onesided']:.4f}")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_dataset"},
                     indent=2))


if __name__ == "__main__":
    main()
