"""H12: revision re-analyses of existing results (no new model training).

Addresses reviewer requests that need no new compute:
  A. Tie-threshold sensitivity: absolute thresholds, a seed-noise-based
     threshold (|delta| < 2 SE of the paired per-seed difference) and a
     relative threshold (|delta| as a fraction of the remaining error
     1 - AUC_best); winner-classification LODO test re-run at each
     threshold; margins in accuracy and log-loss.
  B. Clustered inference for the alignment analysis. The 144 cells per
     cohort come from 12 datasets, so cells are not exchangeable. Three
     cluster-respecting analyses:
       (i)  cluster bootstrap CI of the pooled Spearman rho (datasets
            resampled with replacement);
       (ii) per-dataset rho, one-sided Wilcoxon across datasets (n = 12);
       (iii) dataset-identity permutation: predicted changes of dataset i
            are paired with observed changes of dataset pi(i) at the same
            (intervention, dose). This null keeps the intervention-type
            structure, so it tests whether the meta-model captures
            DATASET-SPECIFIC variation beyond the type of edit.
  C. Practical value of the meta-model: LODO decision analysis (AUC regret
     of family-selection policies) and a prioritisation analysis (training
     the MLP only on the datasets ranked highest by predicted margin).
Output: results/h12_report.json.
"""
import json

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon

from config import RESULTS_DIR
from h1b_analysis import lodo_accuracy
from log_utils import log
from meta_learner import build_meta_table

REPORT = RESULTS_DIR / "h12_report.json"
N_PERM = 5000
N_BOOT = 5000
N_PERM_CLF = 200
ABS_TAUS = [0.001, 0.0025, 0.005, 0.01, 0.02]
REL_TAUS = [0.02, 0.05, 0.10]


# --- A. tie-threshold sensitivity -------------------------------------------

def per_dataset_margins() -> list[dict]:
    rows = []
    for p in sorted(RESULTS_DIR.glob("[0-9]*.json")):
        r = json.loads(p.read_text())
        if "nn_minus_gbdt" not in r:
            continue
        f = r["families"]
        g = max(("xgb", "lgbm"), key=lambda k: f[k]["test_auc_mean"])
        runs_m, runs_g = f["mlp"]["test_runs"], f[g]["test_runs"]
        d_auc = np.array([a["roc_auc_ovr"] - b["roc_auc_ovr"]
                          for a, b in zip(runs_m, runs_g)])
        best_auc = max(f[k]["test_auc_mean"] for k in ("mlp", "xgb", "lgbm"))

        def fam_mean(fam, key):
            return float(np.mean([t[key] for t in f[fam]["test_runs"]]))
        rows.append({
            "dataset_id": r["dataset_id"],
            "margin": r["nn_minus_gbdt"],
            "se": float(d_auc.std(ddof=1) / np.sqrt(len(d_auc))),
            "headroom": float(1 - best_auc),
            "acc_margin": fam_mean("mlp", "accuracy") - max(
                fam_mean("xgb", "accuracy"), fam_mean("lgbm", "accuracy")),
            # log-loss: lower is better, so sign flipped to keep "+ = MLP"
            "ll_margin": min(fam_mean("xgb", "log_loss"),
                             fam_mean("lgbm", "log_loss"))
            - fam_mean("mlp", "log_loss"),
        })
    return rows


def winner_test(df, tie_mask, seed=7) -> dict:
    feats = [c for c in df.columns
             if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
    dec = df[~tie_mask].reset_index(drop=True)
    X = dec[feats].fillna(dec[feats].median(numeric_only=True)).values
    y = dec["nn_wins"].values
    acc = lodo_accuracy(X, y)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(N_PERM_CLF)]
    null = np.array(Parallel(n_jobs=6)(delayed(lodo_accuracy)(X, yp)
                                       for yp in perms))
    return {"n_decided": int(len(y)), "n_mlp_wins": int(y.sum()),
            "lodo_acc": float(acc),
            "majority_baseline": float(max(y.mean(), 1 - y.mean())),
            "perm_p": float((1 + (null >= acc).sum()) / (1 + N_PERM_CLF))}


def tie_sensitivity() -> dict:
    rows = per_dataset_margins()
    m = np.array([r["margin"] for r in rows])
    se = np.array([r["se"] for r in rows])
    head = np.array([r["headroom"] for r in rows])
    df = build_meta_table()
    order = {d: i for i, d in enumerate(r["dataset_id"] for r in rows)}
    df_ix = np.array([order[d] for d in df["dataset_id"]])

    def summarise(tie: np.ndarray, label: str, run_clf: bool) -> dict:
        out = {"rule": label, "n_ties": int(tie.sum()),
               "n_mlp_wins": int(((m > 0) & ~tie).sum()),
               "n_gbdt_wins": int(((m < 0) & ~tie).sum())}
        if run_clf:
            out["winner_classification"] = winner_test(df, tie[df_ix])
        log(f"[h12] ties {label}: {out['n_ties']}/61")
        return out

    res = [summarise(np.abs(m) < t, f"|d|<{t}", True) for t in ABS_TAUS]
    res.append(summarise(np.abs(m) < 2 * se, "|d|<2SE(seed)", True))
    res += [summarise(np.abs(m) < t * np.maximum(head, 1e-6),
                      f"|d|<{t}*(1-AUC)", False) for t in REL_TAUS]
    acc = np.array([r["acc_margin"] for r in rows])
    ll = np.array([r["ll_margin"] for r in rows])
    dec = np.abs(m) >= 0.005
    return {
        "rules": res,
        "seed_se_median": float(np.median(se)),
        "seed_se_iqr": [float(np.quantile(se, .25)),
                        float(np.quantile(se, .75))],
        "ties_005_also_2se_tie": int(((np.abs(m) < 0.005)
                                      & (np.abs(m) < 2 * se)).sum()),
        "ties_005_below_1pct_headroom": int(((np.abs(m) < 0.005)
                                             & (head < 0.01)).sum()),
        "other_metrics": {
            "spearman_auc_vs_acc_margin": float(spearmanr(m, acc).statistic),
            "spearman_auc_vs_logloss_margin": float(
                spearmanr(m, ll).statistic),
            "sign_agree_acc_on_auc_decided":
                f"{int((np.sign(m[dec]) == np.sign(acc[dec])).sum())}"
                f"/{int(dec.sum())}",
            "sign_agree_ll_on_auc_decided":
                f"{int((np.sign(m[dec]) == np.sign(ll[dec])).sum())}"
                f"/{int(dec.sum())}",
            "acc_gbdt_wins": int((acc < -0.005).sum()),
            "acc_mlp_wins": int((acc > 0.005).sum()),
        },
    }


# --- B. clustered alignment inference ---------------------------------------

def alignment_rows() -> list[dict]:
    pilot = json.loads((RESULTS_DIR / "h5_report.json").read_text())["rows"]
    for r in pilot:
        r["cohort"] = "pilot"
    ext = json.loads((RESULTS_DIR / "h9_report.json").read_text())["rows"]
    return pilot + [r for r in ext if r["cohort"] == "extension"]


def _rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def clustered(rows: list[dict], rng: np.random.Generator) -> dict:
    ids = sorted({r["dataset_id"] for r in rows})
    by = {d: [r for r in rows if r["dataset_id"] == d] for d in ids}
    obs = np.array([r["delta_obs"] for r in rows])
    pred = np.array([r["delta_pred"] for r in rows])
    rho = _rho(pred, obs)
    free_null = np.array([_rho(pred, rng.permutation(obs))
                          for _ in range(N_PERM)])

    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        o = np.concatenate([[r["delta_obs"] for r in by[d]] for d in pick])
        p = np.concatenate([[r["delta_pred"] for r in by[d]] for d in pick])
        boots.append(_rho(p, o))
    boots = np.array(boots)

    per_ds = []
    for d in ids:
        o = [r["delta_obs"] for r in by[d]]
        p = [r["delta_pred"] for r in by[d]]
        if len(o) >= 3 and np.std(o) > 0 and np.std(p) > 0:
            per_ds.append(_rho(p, o))
    per_ds = np.array(per_ds)

    key = {(r["dataset_id"], r["kind"], r["dose"]): r["delta_obs"]
           for r in rows}
    ident_null = []
    for _ in range(N_PERM):
        perm = dict(zip(ids, rng.permutation(ids)))
        pp, oo = [], []
        for r in rows:
            k = (perm[r["dataset_id"]], r["kind"], r["dose"])
            if k in key:
                pp.append(r["delta_pred"])
                oo.append(key[k])
        ident_null.append(_rho(pp, oo))
    ident_null = np.array(ident_null)
    return {
        "n_cells": len(rows), "n_datasets": len(ids), "rho": rho,
        "free_perm_p (original, ignores clustering)":
            float((1 + (free_null >= rho).sum()) / (1 + N_PERM)),
        "cluster_bootstrap_ci95": [float(np.quantile(boots, .025)),
                                   float(np.quantile(boots, .975))],
        "cluster_bootstrap_p_le0": float((boots <= 0).mean()),
        "per_dataset_rho_mean": float(per_ds.mean()) if len(per_ds) else None,
        "per_dataset_rho_n_pos": f"{int((per_ds > 0).sum())}/{len(per_ds)}",
        "per_dataset_wilcoxon_p": float(wilcoxon(
            per_ds, alternative="greater").pvalue) if len(per_ds) >= 5
        else None,
        "dataset_identity_perm_p":
            float((1 + (ident_null >= rho).sum()) / (1 + N_PERM)),
        "dataset_identity_null_mean": float(ident_null.mean()),
    }


def alignment() -> dict:
    rng = np.random.default_rng(11)
    rows = [r for r in alignment_rows() if r["kind"] != "select_features"]
    out = {}
    for coh in ("pilot", "extension", "pooled"):
        sub = rows if coh == "pooled" else [r for r in rows
                                            if r["cohort"] == coh]
        out[coh] = {"all": clustered(sub, rng)}
        out[coh]["without_noise"] = clustered(
            [r for r in sub if r["kind"] != "noise_features"], rng)
        for kind in ("noise_features", "rotate", "quantize"):
            out[coh][kind] = clustered([r for r in sub if r["kind"] == kind],
                                       rng)
        log(f"[h12] align {coh}: rho={out[coh]['all']['rho']:.3f} "
            f"CI={out[coh]['all']['cluster_bootstrap_ci95']} "
            f"ident_p={out[coh]['all']['dataset_identity_perm_p']:.4f}")
    return out


# --- C. practical value of the meta-model -----------------------------------

def single_descriptor_preds(df, feat: str) -> np.ndarray:
    from sklearn.ensemble import GradientBoostingRegressor
    X1 = df[[feat]].fillna(df[feat].median()).values
    y = df["nn_minus_gbdt"].values
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        reg = GradientBoostingRegressor(random_state=0, max_depth=2,
                                        n_estimators=150)
        reg.fit(X1[mask], y[mask])
        preds[i] = reg.predict(X1[i:i + 1])[0]
    return preds


def decision_analysis() -> dict:
    meta = json.loads((RESULTS_DIR / "meta_report.json").read_text())
    pm = {r["dataset_id"]: r["pred_margin"] for r in meta["per_dataset"]}
    df = build_meta_table()
    d = df["nn_minus_gbdt"].values
    p_full = np.array([pm[i] for i in df["dataset_id"]])
    p_mi = single_descriptor_preds(df, "T_mi_max")
    n = len(d)
    gain_total = float(np.clip(d, 0, None).sum())

    def regret(choose_mlp: np.ndarray) -> float:
        return float(np.where(choose_mlp, np.clip(-d, 0, None),
                              np.clip(d, 0, None)).mean())

    pol = {"oracle": 0.0,
           "always_gbdt": regret(np.zeros(n, bool)),
           "always_mlp": regret(np.ones(n, bool)),
           "meta_regressor_sign": regret(p_full > 0),
           "T_mi_max_sign": regret(p_mi > 0)}

    # prioritisation: default GBDT; the MLP is additionally trained (and the
    # better model kept) on the top-k datasets by predicted margin.
    def captured(pred, k):
        top = np.argsort(-pred)[:k]
        return float(np.clip(d[top], 0, None).sum() / gain_total)

    curve = []
    for frac in (0.1, 0.2, 0.25, 0.33, 0.5):
        k = int(round(frac * n))
        curve.append({"frac_datasets_mlp_trained": frac, "k": k,
                      "gain_captured_meta": captured(p_full, k),
                      "gain_captured_T_mi_max": captured(p_mi, k),
                      "gain_captured_random_expected": k / n})
    rng = np.random.default_rng(5)
    k = int(round(0.25 * n))
    null = np.array([np.clip(d[rng.permutation(n)[:k]], 0, None).sum()
                     / gain_total for _ in range(N_PERM)])
    obs = captured(p_full, k)
    return {
        "n": n, "mean_regret_auc": pol,
        "mlp_gain_total_auc": gain_total,
        "n_datasets_mlp_better": int((d > 0).sum()),
        "prioritisation_curve": curve,
        "top25pct_perm_p_meta": float((1 + (null >= obs).sum())
                                      / (1 + N_PERM)),
        "top25pct_perm_p_T_mi_max": float(
            (1 + (null >= captured(p_mi, k)).sum()) / (1 + N_PERM)),
        "spearman_T_mi_max_lodo": _rho(p_mi, d),
    }


def main() -> None:
    rep = {"A_tie_sensitivity": tie_sensitivity(),
           "B_alignment_clustered": alignment(),
           "C_decision_analysis": decision_analysis()}
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1)[:6000])


if __name__ == "__main__":
    main()
