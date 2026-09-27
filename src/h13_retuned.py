"""H13: the manipulation experiments under the FULL protocol.

Motivation (revision): H2/H6/H7/H11 retrain the configuration tuned on the
unmodified data (frozen configs), use 3 seeds, 24 datasets and a mild
rotation. H13 removes these weaknesses:
  * every family is re-tuned from scratch (30 random-search configurations,
    validation split) on EVERY manipulated dataset, then evaluated with 5
    seeds - exactly the baseline protocol;
  * all 61 datasets (ties included: the predicted direction of the noise
    and rotation effects does not depend on who wins at baseline);
  * a full random rotation after Gaussianisation (as in Grinsztajn et al.
    2022), compared against Gaussianisation alone, which isolates the
    effect of the rotation from that of the marginal transform.

PRE-REGISTERED HYPOTHESES (fixed before execution; see h13_prereg.json,
which stores the SHA-256 of this file and of the transforms it uses; the
run refuses to start if the hashes change). For every neural family
a in {mlp, resnet, ftt}, gap_a = AUC_a - max(AUC_xgb, AUC_lgbm), all
families re-tuned on the same manipulated data:
  P1 noise_features 4p: gap_a decreases vs dose 0 (one-sided Wilcoxon over
     datasets; Holm over the three families).                    [primary]
  P2 dose-response: mean within-dataset Spearman(dose, gap_a) over doses
     {0, .5, 1, 2, 4}p is negative (one-sided Wilcoxon).
  P3 noise_realistic 4p: gap_a decreases.
  P4 rotate_full: gap_a(rotated) - gap_a(Gaussianised only) > 0 (datasets
     with >= 2 numeric features; one-sided Wilcoxon; Holm over families).
  P5 select_features f=0.25 on GBDT-favoured decided datasets (baseline
     gap_mlp <= -0.005): gap_a increases (one-sided Wilcoxon).
  Replication: P1 is additionally tested separately on the 24 original
  intervention datasets and on the remaining (new) datasets.
  Secondary: re-tuned vs frozen effect size at 4p on the 24 original
  datasets (paired, two-sided), for the MLP.
  Feasibility rules (fixed in advance): FT-Transformer cells whose input
  exceeds FTT_MAX_TOKENS columns after preprocessing, or whose dataset has
  more than FTT_MAX_ROWS rows, are not run and are reported as missing;
  the FTT analysis uses the datasets it covers. Cells are executed in the
  priority order of TIERS (primary endpoint first); if the compute budget
  ends before a tier is complete, the analysis reports that tier as
  incomplete and uses only datasets with every required cell.

Work units are (dataset, kind, dose, family); results are written
incrementally to results/h13/ and several worker processes may run in
parallel on disjoint families (e.g. --families mlp resnet ftt xgb on the
GPU and --families lgbm on the CPU).
"""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import special_ortho_group
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

from config import RESULTS_DIR, ROOT, TUNING_SEED
from h7_rescue import mi_ranking
from h10_realistic_noise import realistic_noise
from interventions import transform
from log_utils import log
from models import build_model, sample_config
from train_baseline import _load, _score

OUT = RESULTS_DIR / "h13"
OUT.mkdir(parents=True, exist_ok=True)
PREREG = RESULTS_DIR / "h13_prereg.json"
SRC = ROOT / "src"
HASHED = ["h13_retuned.py", "interventions.py", "h10_realistic_noise.py",
          "h7_rescue.py", "models.py", "models_arch.py"]

N_CONFIGS = 30
SEEDS = [0, 1, 2, 3, 4]
NOISE_DOSES = [0.5, 1.0, 2.0, 4.0]
SELECT_FRACS = [0.75, 0.5, 0.25, 0.1]
TIE = 0.005
FTT_MAX_TOKENS = 256
FTT_MAX_ROWS = 10_000
TIERS = {("base", 0): 0, ("noise_features", 4.0): 0,
         ("gaussianise", 0): 1, ("rotate_full", 0): 1,
         ("select_features", 0.25): 2,
         ("noise_features", 0.5): 3, ("noise_features", 1.0): 3,
         ("noise_features", 2.0): 3}          # everything else: tier 4
FAMILIES = ["mlp", "resnet", "ftt", "xgb", "lgbm"]
# baseline (dose 0) results for these families already follow the identical
# protocol (30 configs from TUNING_SEED, 5 seeds) and are reused
REUSE_BASE = {"mlp", "xgb", "lgbm"}


# --- manipulations ----------------------------------------------------------

def gaussianise(X, itr, num):
    """Median-impute and map every numeric column to a standard normal
    marginal; both fitted on TRAIN rows only."""
    X = X.copy()
    imp = SimpleImputer(strategy="median").fit(X[num].iloc[itr])
    Z = imp.transform(X[num])
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(itr)),
                             random_state=0).fit(Z[itr])
    X[num] = qt.transform(Z)
    return X


def rotate_full(X, itr, num):
    X = gaussianise(X, itr, num)
    R = special_ortho_group.rvs(len(num), random_state=0)
    X[num] = X[num].values @ R
    return X


def manipulate(kind, dose, X, y, itr, cat):
    num = [c for c in X.columns if c not in cat]
    if kind == "base":
        return X, cat
    if kind == "noise_features":
        return transform(X, itr, num, kind, dose), cat
    if kind == "noise_realistic":
        return realistic_noise(X, itr, num), cat
    if kind == "gaussianise":
        return gaussianise(X, itr, num), cat
    if kind == "rotate_full":
        return rotate_full(X, itr, num), cat
    if kind == "select_features":
        ranked = mi_ranking(X, y, itr, cat)
        keep = ranked[:max(2, int(round(dose * len(ranked))))]
        return X[keep], [c for c in keep if c in cat]
    raise ValueError(kind)


# --- work units -------------------------------------------------------------

def dataset_ids() -> list[int]:
    return sorted(json.loads(p.read_text())["dataset_id"]
                  for p in RESULTS_DIR.glob("[0-9]*.json")
                  if "nn_minus_gbdt" in json.loads(p.read_text()))


def plan() -> list[tuple]:
    units = []
    for did in dataset_ids():
        base = json.loads((RESULTS_DIR / f"{did}.json").read_text())
        meta, X, *_ = _load(did)
        n_num = len([c for c in X.columns if c not in meta["categorical_cols"]])
        cells = [("base", 0)] + [("noise_features", d) for d in NOISE_DOSES]
        if n_num >= 1:
            cells.append(("noise_realistic", 4.0))
        if n_num >= 2:
            cells += [("gaussianise", 0), ("rotate_full", 0)]
        if base["nn_minus_gbdt"] <= -TIE:
            cells += [("select_features", f) for f in SELECT_FRACS]
        units += [(did, k, d) for k, d in cells]
    return units


def unit_path(did, kind, dose, fam):
    return OUT / f"{did}_{kind}_{dose}_{fam}.json"


def run_unit(did, kind, dose, fam) -> None:
    path = unit_path(did, kind, dose, fam)
    if path.exists():
        return
    lock = path.with_suffix(".lock")
    try:
        lock.touch(exist_ok=False)          # another worker owns this unit
    except FileExistsError:
        return
    try:
        t0 = time.time()
        meta, X, y, itr, iva, ite = _load(did)
        if kind == "base" and fam in REUSE_BASE:
            f = json.loads((RESULTS_DIR / f"{did}.json").read_text())
            f = f["families"][fam]
            res = {"best_cfg": f["best_cfg"], "val_auc": f["val_auc"],
                   "test_aucs": [r["roc_auc_ovr"] for r in f["test_runs"]],
                   "n_trials_ok": f["n_trials_ok"], "reused": True}
        else:
            Xt, cat = manipulate(kind, dose, X, y, itr,
                                 meta["categorical_cols"])
            num = [c for c in Xt.columns if c not in cat]
            if fam == "ftt":
                width = build_model("mlp", sample_config(
                    "mlp", np.random.default_rng(0)), cat, num, 0)["prep"] \
                    .fit(Xt.iloc[itr]).transform(Xt.iloc[:2]).shape[1]
                why = (f"{width} tokens > {FTT_MAX_TOKENS}"
                       if width > FTT_MAX_TOKENS else
                       f"{len(X)} rows > {FTT_MAX_ROWS}"
                       if len(X) > FTT_MAX_ROWS else None)
                if why:
                    path.write_text(json.dumps({"skipped": why}))
                    log(f"[h13] {did}/{kind}@{dose}/{fam}: skipped ({why})")
                    return
            rng = np.random.default_rng(TUNING_SEED)
            best_cfg, best_val, n_ok = None, -np.inf, 0
            for _ in range(N_CONFIGS):
                cfg = sample_config(fam, rng)
                try:
                    pipe = build_model(fam, cfg, cat, num, seed=TUNING_SEED)
                    pipe.fit(Xt.iloc[itr], y.iloc[itr])
                    v = _score(pipe, Xt.iloc[iva], y.iloc[iva])["roc_auc_ovr"]
                except Exception as e:  # noqa: BLE001
                    log(f"[h13] {did}/{kind}@{dose}/{fam} trial ERROR "
                        f"{type(e).__name__}: {e}")
                    continue
                n_ok += 1
                if v > best_val:
                    best_cfg, best_val = cfg, v
            if best_cfg is None:
                raise RuntimeError("all tuning trials failed")
            aucs = []
            for s in SEEDS:
                pipe = build_model(fam, best_cfg, cat, num, seed=s)
                pipe.fit(Xt.iloc[itr], y.iloc[itr])
                aucs.append(_score(pipe, Xt.iloc[ite],
                                   y.iloc[ite])["roc_auc_ovr"])
            res = {"best_cfg": best_cfg, "val_auc": best_val,
                   "test_aucs": aucs, "n_trials_ok": n_ok, "reused": False}
        res.update({"dataset_id": did, "kind": kind, "dose": dose,
                    "family": fam, "test_auc": float(np.mean(res["test_aucs"])),
                    "seconds": round(time.time() - t0, 1)})
        path.write_text(json.dumps(res, indent=2))
        log(f"[h13] {did}/{kind}@{dose}/{fam}: auc={res['test_auc']:.4f} "
            f"({res['seconds']:.0f}s)")
    except Exception as e:  # noqa: BLE001
        log(f"[h13] {did}/{kind}@{dose}/{fam} ERROR {type(e).__name__}: {e}")
    finally:
        lock.unlink(missing_ok=True)


# --- pre-registration -------------------------------------------------------

def _hashes() -> dict:
    return {f: hashlib.sha256((SRC / f).read_bytes()).hexdigest()
            for f in HASHED}


def write_prereg() -> None:
    if PREREG.exists():
        raise SystemExit("pre-registration already exists; never overwrite")
    units = plan()
    reg = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "hypotheses": __doc__, "sha256": _hashes(),
           "n_configs": N_CONFIGS, "seeds": SEEDS,
           "datasets": dataset_ids(), "n_cells": len(units),
           "cells": [list(u) for u in units]}
    PREREG.write_text(json.dumps(reg, indent=2))
    print(f"pre-registered {len(units)} cells on {len(reg['datasets'])} "
          f"datasets")


def check_prereg() -> list[tuple]:
    reg = json.loads(PREREG.read_text())
    if reg["sha256"] != _hashes():
        raise SystemExit("code changed after pre-registration - refusing")
    return [tuple(u) for u in reg["cells"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--kinds", nargs="+")
    ap.add_argument("--clear-locks", action="store_true",
                    help="remove stale locks after a crash (no workers running)")
    args = ap.parse_args()
    if args.prereg:
        write_prereg()
        return
    if args.clear_locks:
        n = sum(1 for p in OUT.glob("*.lock") if not p.unlink())
        print(f"removed {n} stale locks")
        return
    cells = check_prereg()
    if args.kinds:
        cells = [c for c in cells if c[1] in args.kinds]
    # pre-registered priority: primary endpoint first (see TIERS)
    cells.sort(key=lambda c: (TIERS.get((c[1], c[2]), 4), c[0]))
    log(f"[h13] worker families={args.families}: {len(cells)} cells")
    for did, kind, dose in cells:
        for fam in args.families:
            run_unit(did, kind, dose, fam)
    log(f"[h13] worker {args.families} finished")


if __name__ == "__main__":
    main()
