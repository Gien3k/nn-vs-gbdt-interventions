"""H11: does the uninformative-feature effect generalise beyond the MLP?

Revision experiment (reviewers: "only one neural architecture"). The same
interventions are re-run for three further neural families:
ResNet, FT-Transformer (Gorishniy et al. 2021) and TabPFN v2 (Hollmann et
al. 2025).

PRE-REGISTERED (fixed before execution; same endpoints as the MLP):
  P1 noise_features 4xp: margin_arch = AUC_arch - best GBDT DECREASES vs
     the arch's untransformed margin (one-sided Wilcoxon over the 24 noise
     datasets, Holm over architectures).
  P2 noise_realistic 4xp: same direction (secondary).
  P3 select_features f=0.25 on the 12 rescue datasets: margin INCREASES
     (one-sided Wilcoxon, secondary). f=0.1 exploratory.
  No direction is registered for TabPFN beyond P1-P3; prior work reports
  it is comparatively robust to uninformative features, so a null for
  TabPFN is an informative outcome, not a failure.

Protocol per architecture, identical to the MLP study: random search on
TRAIN scored on VAL (TabPFN: no tuning), best config re-trained with three
seeds on TRAIN, scored once per seed on TEST; interventions re-train the
frozen best config on the transformed data (transform statistics from TRAIN
only). GBDT reference values are the existing XGBoost/LightGBM results for
the identical cells, so every margin compares against the same trees.

Outputs: results/arch/{id}_{arch}_{kind}_{dose}.json and
results/h11_report.json (python h11_architectures.py --report).
"""
import argparse
import json
import time

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from config import RESULTS_DIR, TUNING_SEED
from h7_rescue import mi_ranking
from h10_realistic_noise import realistic_noise
from interventions import transform
from log_utils import log
from models import build_model, sample_config
from train_baseline import _load, _score

INT_DIR = RESULTS_DIR / "interventions"
ARCH_DIR = RESULTS_DIR / "arch"
ARCH_DIR.mkdir(parents=True, exist_ok=True)
REPORT = RESULTS_DIR / "h11_report.json"

ARCHS = ["resnet", "ftt", "tabpfn"]
N_CONFIGS = {"resnet": 30, "ftt": 30, "tabpfn": 0}
SEEDS = [0, 1, 2]
NOISE_DOSES = [0.5, 1.0, 2.0, 4.0]
RESCUE_FRACS = [0.25, 0.1]


def noise_ids() -> list[int]:
    return sorted({json.loads(p.read_text())["dataset_id"]
                   for p in INT_DIR.glob("*_noise_features_4.0.json")})


def rescue_ids() -> list[int]:
    return sorted({json.loads(p.read_text())["dataset_id"]
                   for p in INT_DIR.glob("*_select_features_0.25.json")})


def _fit_eval(arch, cfg, X, y, itr, ite, cat, num) -> list[float]:
    aucs = []
    for s in SEEDS:
        pipe = build_model(arch, cfg, cat, num, seed=s)
        pipe.fit(X.iloc[itr], y.iloc[itr])
        aucs.append(_score(pipe, X.iloc[ite], y.iloc[ite])["roc_auc_ovr"])
    return aucs


def baseline(did: int, arch: str, n_configs: int) -> dict:
    out_path = ARCH_DIR / f"{did}_{arch}_base_0.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    t0 = time.time()
    meta, X, y, itr, iva, ite = _load(did)
    cat = meta["categorical_cols"]
    num = [c for c in X.columns if c not in cat]
    rng = np.random.default_rng(TUNING_SEED)
    best_cfg, best_val, n_ok = {}, None, 0
    for k in range(n_configs):
        cfg = sample_config(arch, rng)
        try:
            pipe = build_model(arch, cfg, cat, num, seed=TUNING_SEED)
            pipe.fit(X.iloc[itr], y.iloc[itr])
            val = _score(pipe, X.iloc[iva], y.iloc[iva])["roc_auc_ovr"]
        except Exception as e:  # noqa: BLE001
            log(f"[h11] {did}/{arch} trial {k+1} ERROR {type(e).__name__}: {e}")
            continue
        n_ok += 1
        if best_val is None or val > best_val:
            best_cfg, best_val = cfg, val
    if n_configs and best_val is None:
        raise RuntimeError(f"{did}/{arch}: all tuning trials failed")
    aucs = _fit_eval(arch, best_cfg, X, y, itr, ite, cat, num)
    base = json.loads((RESULTS_DIR / f"{did}.json").read_text())
    gbdt = max(base["families"][f]["test_auc_mean"] for f in ("xgb", "lgbm"))
    cell = {"dataset_id": did, "arch": arch, "kind": "base", "dose": 0,
            "best_cfg": best_cfg, "val_auc": best_val, "n_trials_ok": n_ok,
            "test_aucs": aucs, "arch_auc": float(np.mean(aucs)),
            "gbdt_auc": float(gbdt), "margin": float(np.mean(aucs) - gbdt),
            "mlp_auc": base["families"]["mlp"]["test_auc_mean"],
            "seconds": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(cell, indent=2))
    log(f"[h11] {did}/{arch} base: auc={cell['arch_auc']:.4f} "
        f"margin={cell['margin']:+.4f} ({cell['seconds']:.0f}s)")
    return cell


def intervention(did: int, arch: str, kind: str, dose: float,
                 base_cell: dict) -> dict:
    out_path = ARCH_DIR / f"{did}_{arch}_{kind}_{dose}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    ref_path = INT_DIR / f"{did}_{kind}_{dose}.json"
    ref = json.loads(ref_path.read_text())
    meta, X, y, itr, iva, ite = _load(did)
    cat = meta["categorical_cols"]
    num = [c for c in X.columns if c not in cat]
    if kind == "noise_features":
        Xt, cat_t = transform(X, itr, num, kind, dose), cat
    elif kind == "noise_realistic":
        Xt, cat_t = realistic_noise(X, itr, num), cat
    elif kind == "select_features":
        ranked = mi_ranking(X, y, itr, cat)
        keep = ranked[:max(2, int(round(dose * len(ranked))))]
        Xt, cat_t = X[keep], [c for c in keep if c in cat]
    else:
        raise ValueError(kind)
    num_t = [c for c in Xt.columns if c not in cat_t]
    aucs = _fit_eval(arch, base_cell["best_cfg"], Xt, y, itr, ite,
                     cat_t, num_t)
    fams = ref["families"]
    gbdt = max(fams["xgb"]["test_auc_mean"], fams["lgbm"]["test_auc_mean"])
    cell = {"dataset_id": did, "arch": arch, "kind": kind, "dose": dose,
            "test_aucs": aucs, "arch_auc": float(np.mean(aucs)),
            "gbdt_auc": float(gbdt), "margin": float(np.mean(aucs) - gbdt),
            "baseline_margin": base_cell["margin"],
            "baseline_arch_auc": base_cell["arch_auc"],
            "baseline_gbdt_auc": base_cell["gbdt_auc"],
            "mlp_auc": fams["mlp"]["test_auc_mean"]}
    out_path.write_text(json.dumps(cell, indent=2))
    log(f"[h11] {did}/{arch}/{kind}@{dose}: margin={cell['margin']:+.4f} "
        f"(base {base_cell['margin']:+.4f})")
    return cell


def run(archs, ids_noise, ids_rescue, doses, n_configs) -> None:
    for arch in archs:
        nc = N_CONFIGS[arch] if n_configs is None else (
            n_configs if N_CONFIGS[arch] else 0)
        for did in sorted(set(ids_noise) | set(ids_rescue)):
            try:
                b = baseline(did, arch, nc)
                todo = []
                if did in ids_noise:
                    todo += [("noise_features", d) for d in doses]
                    if (INT_DIR / f"{did}_noise_realistic_4.0.json").exists():
                        todo.append(("noise_realistic", 4.0))
                if did in ids_rescue:
                    todo += [("select_features", f) for f in RESCUE_FRACS]
                for kind, dose in todo:
                    intervention(did, arch, kind, dose, b)
            except Exception as e:  # noqa: BLE001
                log(f"[h11] {did}/{arch} ERROR {type(e).__name__}: {e}")


# --- analysis ---------------------------------------------------------------

def _mlp_cells(kind: str, dose: float, ids: list[int]) -> dict[int, float]:
    out = {}
    for did in ids:
        p = INT_DIR / f"{did}_{kind}_{dose}.json"
        if p.exists():
            c = json.loads(p.read_text())
            out[did] = c["nn_minus_gbdt"] - c["baseline_margin"]
    return out


def _arch_cells(arch: str, kind: str, dose: float) -> dict[int, dict]:
    return {c["dataset_id"]: c for c in
            (json.loads(p.read_text()) for p in
             ARCH_DIR.glob(f"*_{arch}_{kind}_{dose}.json"))}


def _endpoint(deltas: list[float], alt: str) -> dict:
    d = np.array(deltas)
    sign = -1 if alt == "less" else 1
    return {"n": int(len(d)), "mean_delta": float(d.mean()),
            "n_direction_ok": int(((d * sign) > 0).sum()),
            "wilcoxon_p": float(wilcoxon(d, alternative=alt).pvalue)
            if len(d) >= 5 else None}


def report() -> dict:
    pilot = set(json.loads((RESULTS_DIR / "pilot_ids.json").read_text()))
    rep: dict = {"archs": {}}
    for arch in ["mlp"] + ARCHS:
        r: dict = {}
        if arch == "mlp":
            nf = _mlp_cells("noise_features", 4.0, noise_ids())
            nr = _mlp_cells("noise_realistic", 4.0, noise_ids())
            sf = _mlp_cells("select_features", 0.25, rescue_ids())
            sf10 = _mlp_cells("select_features", 0.1, rescue_ids())
            base = {did: json.loads((RESULTS_DIR / f"{did}.json").read_text())
                    ["nn_minus_gbdt"] for did in noise_ids()}
            abs_drop = {}
            for did in noise_ids():
                c = json.loads((INT_DIR / f"{did}_noise_features_4.0.json")
                               .read_text())
                b = json.loads((RESULTS_DIR / f"{did}.json").read_text())
                abs_drop[did] = (c["families"]["mlp"]["test_auc_mean"]
                                 - b["families"]["mlp"]["test_auc_mean"])
            trend = {}
            for did in noise_ids():
                ms = [base[did]] + [json.loads(
                    (INT_DIR / f"{did}_noise_features_{d}.json").read_text())
                    ["nn_minus_gbdt"] for d in NOISE_DOSES]
                trend[did] = spearmanr(range(len(ms)), ms).statistic
        else:
            cells = {d: _arch_cells(arch, "noise_features", d)
                     for d in NOISE_DOSES}
            if not cells[4.0]:
                continue
            nf = {k: c["margin"] - c["baseline_margin"]
                  for k, c in cells[4.0].items()}
            nr = {k: c["margin"] - c["baseline_margin"] for k, c in
                  _arch_cells(arch, "noise_realistic", 4.0).items()}
            sf = {k: c["margin"] - c["baseline_margin"] for k, c in
                  _arch_cells(arch, "select_features", 0.25).items()}
            sf10 = {k: c["margin"] - c["baseline_margin"] for k, c in
                    _arch_cells(arch, "select_features", 0.1).items()}
            base = {c["dataset_id"]: c["margin"] for c in
                    (json.loads(p.read_text()) for p in
                     ARCH_DIR.glob(f"*_{arch}_base_0.json"))}
            abs_drop = {k: c["arch_auc"] - c["baseline_arch_auc"]
                        for k, c in cells[4.0].items()}
            trend = {}
            for did in cells[4.0]:
                if all(did in cells[d] for d in NOISE_DOSES):
                    ms = [base[did]] + [cells[d][did]["margin"]
                                        for d in NOISE_DOSES]
                    trend[did] = spearmanr(range(len(ms)), ms).statistic
        r["baseline_margin_mean"] = float(np.mean(
            [base[k] for k in noise_ids() if k in base]))
        r["baseline_n_better_than_gbdt"] = int(sum(
            base[k] > 0.005 for k in noise_ids() if k in base))
        r["noise4x"] = _endpoint(list(nf.values()), "less")
        for coh, sel in (("pilot", lambda k: k in pilot),
                         ("extension", lambda k: k not in pilot)):
            vals = [v for k, v in nf.items() if sel(k)]
            if len(vals) >= 5:
                r[f"noise4x_{coh}"] = _endpoint(vals, "less")
        r["noise4x_abs_auc_change"] = float(np.mean(list(abs_drop.values())))
        r["noise_trend_rho_mean"] = float(np.nanmean(list(trend.values()))) \
            if trend else None
        if nr:
            r["noise_realistic4x"] = _endpoint(list(nr.values()), "less")
        if sf:
            r["rescue_f025"] = _endpoint(list(sf.values()), "greater")
        if sf10:
            r["rescue_f010_exploratory"] = _endpoint(list(sf10.values()),
                                                     "greater")
        r["per_dataset_noise4x_delta"] = {str(k): float(v)
                                          for k, v in sorted(nf.items())}
        rep["archs"][arch] = r
    # Holm over the new architectures' primary endpoint (P1)
    prim = sorted(((a, v["noise4x"]["wilcoxon_p"]) for a, v in
                   rep["archs"].items() if a != "mlp"
                   and v["noise4x"]["wilcoxon_p"] is not None),
                  key=lambda kv: kv[1])
    running = 0.0
    for rank, (a, p) in enumerate(prim):
        running = min(1.0, max(running, (len(prim) - rank) * p))
        rep["archs"][a]["noise4x"]["holm_p"] = float(running)
    REPORT.write_text(json.dumps(rep, indent=2))
    for a, v in rep["archs"].items():
        e = v["noise4x"]
        log(f"[h11] {a}: noise4x d={e['mean_delta']:+.4f} "
            f"{e['n_direction_ok']}/{e['n']} p={e['wilcoxon_p']}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=ARCHS)
    ap.add_argument("--datasets", nargs="+", type=int)
    ap.add_argument("--doses", nargs="+", type=float, default=NOISE_DOSES)
    ap.add_argument("--n-configs", type=int)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        rep = report()
        print(json.dumps({a: {k: v for k, v in r.items()
                              if not k.startswith("per_dataset")}
                          for a, r in rep["archs"].items()}, indent=2))
        return
    ids_n, ids_r = noise_ids(), rescue_ids()
    if args.datasets:
        ids_n = [d for d in ids_n if d in args.datasets]
        ids_r = [d for d in ids_r if d in args.datasets]
    log(f"[h11] archs={args.archs} noise={len(ids_n)} rescue={len(ids_r)}")
    run(args.archs, ids_n, ids_r, args.doses, args.n_configs)
    log("[h11] run finished")


if __name__ == "__main__":
    main()
