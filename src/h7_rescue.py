"""H7: the rescue experiment (causal converse of the noise result).

PRE-REGISTERED (fixed before execution):
  Intervention: mutual-information feature selection - keep the top
  fraction f of ORIGINAL columns ranked by train-split MI with the target,
  f in {0.75, 0.5, 0.25, 0.1}.
  PRIMARY endpoint: on GBDT-favoured decided datasets, the margin
  (MLP - best GBDT) at f=0.25 INCREASES vs baseline
  (one-sided Wilcoxon, alpha=0.05).
  SECONDARY: monotone trend over f in {1, 0.75, 0.5, 0.25}.
  EXPLORATORY: f=0.1 (aggressive selection may destroy signal for both
  families; no direction registered).
  Additional registered spotlight: madelon (480/500 uninformative features
  by construction) should show a large positive margin recovery.

Cells are stored in results/interventions as {id}_select_features_{f}.json
so alignment analyses can consume them uniformly.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from config import DATA_DIR, RESULTS_DIR
from log_utils import log
from models import build_model
from train_baseline import _load, _score

INT_DIR = RESULTS_DIR / "interventions"
REPORT = RESULTS_DIR / "h7_report.json"
FRACS = [0.75, 0.5, 0.25, 0.1]
SEEDS = [0, 1, 2]
TIE_EPS = 0.005


def pick_datasets(max_n: int = 12) -> list[int]:
    """GBDT-favoured, decided, feasible; madelon explicitly admitted."""
    chosen = []
    for p in sorted(RESULTS_DIR.glob("[0-9]*.json")):
        r = json.loads(p.read_text())
        if r.get("nn_minus_gbdt", 0) >= -TIE_EPS:
            continue
        meta = json.loads((DATA_DIR / str(r["dataset_id"]) / "meta.json")
                          .read_text())
        frac_cat = len(meta["categorical_cols"]) / meta["n_features"]
        if meta["n_rows"] > 20000 or meta["n_features"] > 520 or frac_cat > 0.3:
            continue
        chosen.append((abs(r["nn_minus_gbdt"]), r["dataset_id"]))
    chosen.sort(reverse=True)
    return [c[1] for c in chosen[:max_n]]


def mi_ranking(X: pd.DataFrame, y: pd.Series, itr: np.ndarray,
               cat: list[str]) -> list[str]:
    """Columns ranked by train-split MI, best first. Train rows only."""
    Xtr = X.iloc[itr].copy()
    num = [c for c in X.columns if c not in cat]
    enc = pd.DataFrame(index=Xtr.index)
    if num:
        enc[num] = SimpleImputer(strategy="median").fit_transform(Xtr[num])
    if cat:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value",
                            unknown_value=-1, encoded_missing_value=-2)
        enc[cat] = oe.fit_transform(Xtr[cat].astype(object))
    enc = enc[X.columns.tolist()]
    mi = mutual_info_classif(enc.values, y.iloc[itr], random_state=0)
    order = np.argsort(-mi)
    return [X.columns[j] for j in order]


def run_cell(did: int, frac: float) -> dict:
    out_path = INT_DIR / f"{did}_select_features_{frac}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    meta, X, y, itr, iva, ite = _load(did)
    base = json.loads((RESULTS_DIR / f"{did}.json").read_text())
    cat = meta["categorical_cols"]
    ranked = mi_ranking(X, y, itr, cat)
    k = max(2, int(round(frac * len(ranked))))
    keep = ranked[:k]
    Xs = X[keep]
    cat_s = [c for c in keep if c in cat]
    num_s = [c for c in keep if c not in cat]

    fams = {}
    for fam in ("mlp", "xgb", "lgbm"):
        cfg = base["families"][fam]["best_cfg"]
        aucs = []
        for s in SEEDS:
            pipe = build_model(fam, cfg, cat_s, num_s, seed=s)
            pipe.fit(Xs.iloc[itr], y.iloc[itr])
            aucs.append(_score(pipe, Xs.iloc[ite], y.iloc[ite])["roc_auc_ovr"])
        fams[fam] = {"test_auc_mean": float(np.mean(aucs))}
    nn = fams["mlp"]["test_auc_mean"]
    gbdt = max(fams["xgb"]["test_auc_mean"], fams["lgbm"]["test_auc_mean"])
    cell = {"dataset_id": did, "kind": "select_features", "dose": frac,
            "n_features_kept": k, "families": fams,
            "nn_minus_gbdt": float(nn - gbdt),
            "baseline_margin": base["nn_minus_gbdt"]}
    out_path.write_text(json.dumps(cell, indent=2))
    log(f"[h7] {did}/select@{frac}: margin={nn - gbdt:+.4f} "
        f"(baseline {base['nn_minus_gbdt']:+.4f}, kept {k})")
    return cell


def main() -> None:
    ids = pick_datasets()
    log(f"[h7] selected GBDT-favoured datasets: {ids}")
    cells: dict[int, dict[float, dict]] = {}
    for did in ids:
        for frac in FRACS:
            try:
                cells.setdefault(did, {})[frac] = run_cell(did, frac)
            except Exception as e:  # noqa: BLE001
                log(f"[h7] {did}@{frac} ERROR {type(e).__name__}: {e}")

    deltas_primary, trends, rows = [], [], []
    madelon = None
    for did, by_frac in sorted(cells.items()):
        if 0.25 not in by_frac:
            continue
        base = by_frac[0.25]["baseline_margin"]
        deltas_primary.append(by_frac[0.25]["nn_minus_gbdt"] - base)
        sev = [1.0, 0.75, 0.5, 0.25]
        margins = [base] + [by_frac[f]["nn_minus_gbdt"]
                            for f in sev[1:] if f in by_frac]
        rho = spearmanr(range(len(margins)), margins).statistic
        trends.append(rho)
        row = {"dataset_id": did, "baseline": base,
               "delta_at_0.25": deltas_primary[-1],
               "delta_at_0.1": (by_frac[0.1]["nn_minus_gbdt"] - base)
               if 0.1 in by_frac else None,
               "trend_rho": float(rho)}
        rows.append(row)
        name = json.loads((RESULTS_DIR / f"{did}.json").read_text())["name"]
        if name == "madelon":
            madelon = row

    d = np.array(deltas_primary)
    t = np.array(trends)
    rep = {
        "n_datasets": int(len(d)),
        "primary_mean_delta_at_0.25": float(d.mean()),
        "primary_n_direction_ok": int((d > 0).sum()),
        "primary_wilcoxon_p_onesided": float(
            wilcoxon(d, alternative="greater").pvalue),
        "secondary_mean_trend_rho": float(t.mean()),
        "secondary_wilcoxon_p_onesided": float(
            wilcoxon(t, alternative="greater").pvalue),
        "madelon_spotlight": madelon,
        "per_dataset": rows,
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    log(f"[h7] primary delta@0.25 mean={d.mean():+.4f} "
        f"dir={int((d > 0).sum())}/{len(d)} "
        f"p={rep['primary_wilcoxon_p_onesided']:.4f}")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_dataset"},
                     indent=2))


if __name__ == "__main__":
    main()
