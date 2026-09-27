"""Dataset-geometry descriptors and zero-cost proxies.

The meta-learning hypothesis of the paper: cheap, training-free (or nearly
training-free) properties of a dataset predict whether a tuned MLP or a tuned
GBDT will win, and by how much.

CRITICAL VALIDITY RULE: every descriptor is computed ONLY from the TRAIN
split. If any statistic here ever touched val/test, the meta-learner's
evaluation would be circular (meta-level leakage).

Descriptor families:
  S*  simple meta-features (size, classes, imbalance, missingness)
  E*  spectral geometry of the feature space (train correlation spectrum)
  T*  target geometry (boundary irregularity, linear separability, MI)
  Z*  zero-cost neural proxies (statistics of an UNTRAINED / 1-epoch MLP)
  A*  axis-alignment proxies (what depth-1 stumps can already buy)

Usage: python descriptors.py            # all prepared datasets
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import DATA_DIR, RESULTS_DIR, SPLIT_SEED
from log_utils import log
from models import make_preprocessor

RNG = np.random.default_rng(SPLIT_SEED)
DESC_DIR = RESULTS_DIR / "descriptors"
DESC_DIR.mkdir(parents=True, exist_ok=True)

CAP = 4000   # descriptor computations subsample train to at most this many rows


def _subsample(Xt: np.ndarray, y_idx: np.ndarray, cap: int = CAP):
    if len(Xt) <= cap:
        return Xt, y_idx
    sel = RNG.choice(len(Xt), size=cap, replace=False)
    return Xt[sel], y_idx[sel]


# --- S: simple -------------------------------------------------------------

def simple_features(X: pd.DataFrame, y: pd.Series, cat_cols: list[str]) -> dict:
    p = y.value_counts(normalize=True).values
    return {
        "S_log_n": float(np.log10(len(X))),
        "S_log_p": float(np.log10(X.shape[1])),
        "S_n_over_p": float(len(X) / X.shape[1]),
        "S_n_classes": int(y.nunique()),
        "S_class_entropy_norm": float(-(p * np.log(p)).sum() / np.log(len(p)))
        if len(p) > 1 else 0.0,
        "S_frac_categorical": float(len(cat_cols) / X.shape[1]),
        "S_frac_missing": float(X.isna().mean().mean()),
    }


# --- E: spectral geometry --------------------------------------------------

def spectral_features(Xt: np.ndarray) -> dict:
    Xs = Xt - Xt.mean(0, keepdims=True)
    sd = Xs.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    C = np.corrcoef((Xs / sd).T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = np.clip(ev, 1e-12, None)
    share = ev / ev.sum()
    k = len(ev)
    return {
        "E_eff_rank_frac": float(np.exp(-(share * np.log(share)).sum()) / k),
        "E_top1_share": float(share[0]),
        "E_top5_share": float(share[:5].sum()) if k >= 5 else 1.0,
        "E_mean_abs_corr": float(np.abs(C[np.triu_indices(k, 1)]).mean())
        if k > 1 else 0.0,
    }


# --- T: target geometry ----------------------------------------------------

def target_features(Xt: np.ndarray, y_idx: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import NearestNeighbors

    Xs, ys = _subsample(Xt, y_idx)
    out = {}

    # boundary irregularity: kNN label disagreement
    nn_idx = NearestNeighbors(n_neighbors=min(6, len(Xs) - 1)).fit(Xs)
    _, ind = nn_idx.kneighbors(Xs)
    neigh = ys[ind[:, 1:]]
    out["T_knn_disagreement"] = float((neigh != ys[:, None]).mean())

    # linear separability via 3-fold CV logistic probe (train-internal CV)
    try:
        proba = cross_val_predict(
            LogisticRegression(max_iter=300), Xs, ys, cv=3,
            method="predict_proba")
        if proba.shape[1] == 2:
            out["T_linear_auc"] = float(roc_auc_score(ys, proba[:, 1]))
        else:
            out["T_linear_auc"] = float(roc_auc_score(
                ys, proba, multi_class="ovr", average="macro"))
    except Exception:  # noqa: BLE001 - degenerate folds on tiny/rare classes
        out["T_linear_auc"] = np.nan

    # feature-target MI distribution
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(Xs, ys, random_state=0,
                             n_neighbors=3, discrete_features=False)
    out["T_mi_max"] = float(mi.max()) if len(mi) else 0.0
    out["T_mi_mean"] = float(mi.mean()) if len(mi) else 0.0
    srt = np.sort(mi)
    out["T_mi_gini"] = float(
        (2 * np.arange(1, len(srt) + 1) - len(srt) - 1).dot(srt)
        / (len(srt) * srt.sum())) if srt.sum() > 0 else 0.0
    return out


# --- Z: zero-cost neural proxies -------------------------------------------

def zerocost_features(Xt: np.ndarray, y_idx: np.ndarray) -> dict:
    torch.manual_seed(0)
    Xs, ys = _subsample(Xt, y_idx, cap=1024)
    d_in, d_out = Xs.shape[1], int(ys.max()) + 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, d_out)).to(dev)
    Xb = torch.tensor(Xs, dtype=torch.float32, device=dev)
    yb = torch.tensor(ys, dtype=torch.long, device=dev)
    lossf = nn.CrossEntropyLoss()

    # gradient-norm and gradient-conflict at initialization
    halves = torch.chunk(torch.randperm(len(Xb), device=dev), 2)
    grads = []
    for h in halves:
        net.zero_grad()
        lossf(net(Xb[h]), yb[h]).backward()
        grads.append(torch.cat([p.grad.flatten() for p in net.parameters()]))
    g1, g2 = grads
    out = {
        "Z_grad_norm_init": float(torch.log10((g1 + g2).norm() / 2 + 1e-12)),
        "Z_grad_cosine": float(torch.nn.functional.cosine_similarity(
            g1, g2, dim=0)),
    }

    # loss decrease after one epoch of SGD (normalized)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    with torch.no_grad():
        loss0 = float(lossf(net(Xb), yb))
    perm = torch.randperm(len(Xb), device=dev)
    for i in range(0, len(Xb), 128):
        b = perm[i:i + 128]
        opt.zero_grad()
        lossf(net(Xb[b]), yb[b]).backward()
        opt.step()
    with torch.no_grad():
        loss1 = float(lossf(net(Xb), yb))
    out["Z_one_epoch_gain"] = float((loss0 - loss1) / (loss0 + 1e-12))
    return out


# --- A: axis-alignment proxies ---------------------------------------------

def axis_features(Xt: np.ndarray, y_idx: np.ndarray) -> dict:
    """How much do axis-aligned depth-1 splits already explain? Trees exploit
    exactly this; smooth rotated structure favours nets."""
    from sklearn.metrics import roc_auc_score
    Xs, ys = _subsample(Xt, y_idx)
    ybin = (ys == np.bincount(ys).argmax()).astype(int)  # majority-vs-rest
    aucs = []
    for j in range(Xs.shape[1]):
        col = Xs[:, j]
        if np.unique(col).size < 2 or np.unique(ybin).size < 2:
            continue
        a = roc_auc_score(ybin, col)
        aucs.append(max(a, 1 - a))
    if not aucs:
        return {"A_stump_auc_max": 0.5, "A_stump_auc_mean": 0.5,
                "A_rot_gap": 0.0}
    out = {"A_stump_auc_max": float(np.max(aucs)),
           "A_stump_auc_mean": float(np.mean(aucs))}

    # rotation gap: same statistic after a random rotation - axis-aligned
    # signal should evaporate, smooth signal should survive
    Q, _ = np.linalg.qr(RNG.standard_normal((Xs.shape[1], Xs.shape[1])))
    Xr = Xs @ Q
    r_aucs = []
    for j in range(Xr.shape[1]):
        if np.unique(ybin).size < 2:
            continue
        a = roc_auc_score(ybin, Xr[:, j])
        r_aucs.append(max(a, 1 - a))
    out["A_rot_gap"] = float(np.max(aucs) - np.max(r_aucs)) if r_aucs else 0.0
    return out


# --- driver ----------------------------------------------------------------

def compute_for_dataset(dataset_id: int) -> dict:
    d = DATA_DIR / str(dataset_id)
    meta = json.loads((d / "meta.json").read_text())
    X = pd.read_parquet(d / "X.parquet")
    y = pd.read_parquet(d / "y.parquet")["target"]
    itr = np.load(d / "splits.npz")["train"]

    # TRAIN ONLY - the validity rule of the whole study
    Xtr, ytr = X.iloc[itr], y.iloc[itr]
    cat = meta["categorical_cols"]
    num = [c for c in X.columns if c not in cat]
    prep = make_preprocessor(cat, num, onehot=True)
    Xt = np.asarray(prep.fit_transform(Xtr), dtype=np.float64)
    _, y_idx = np.unique(ytr, return_inverse=True)

    desc = {"dataset_id": dataset_id, "name": meta["name"]}
    desc.update(simple_features(Xtr, ytr, cat))
    desc.update(spectral_features(Xt))
    desc.update(target_features(Xt, y_idx))
    desc.update(zerocost_features(Xt, y_idx))
    desc.update(axis_features(Xt, y_idx))
    (DESC_DIR / f"{dataset_id}.json").write_text(json.dumps(desc, indent=2))
    log(f"[descriptors] {dataset_id} ({meta['name']}): "
        f"{len(desc) - 2} descriptors computed")
    return desc


def main() -> None:
    ids = sorted(int(p.name) for p in DATA_DIR.iterdir()
                 if (p / "meta.json").exists())
    done, failed = 0, 0
    for did in ids:
        if (DESC_DIR / f"{did}.json").exists():
            done += 1
            continue
        try:
            compute_for_dataset(did)
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            log(f"[descriptors] {did} ERROR {type(e).__name__}: {e}")
    log(f"[descriptors] finished: {done} ok, {failed} failed")
    print(json.dumps({"ok": done, "failed": failed}))


if __name__ == "__main__":
    main()
