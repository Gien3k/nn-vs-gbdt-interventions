"""Model families under identical evaluation protocol.

Every model is an sklearn Pipeline whose preprocessing is FITTED ONLY on the
training portion it receives - structural leakage protection: no scaler,
imputer or encoder can ever see validation/test statistics.

Families:
  - mlp: PyTorch MLP (skorch-free minimal wrapper, GPU if available)
  - xgb: XGBoost histogram
  - lgbm: LightGBM
  - resnet, ftt, tabpfn: revision families, see models_arch.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import BATCH_SIZE, EARLY_STOP_PATIENCE, MAX_EPOCHS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- preprocessing (leakage-safe by construction) --------------------------

def make_preprocessor(cat_cols: list[str], num_cols: list[str],
                      onehot: bool) -> ColumnTransformer:
    cat_enc = (OneHotEncoder(handle_unknown="ignore", max_categories=32,
                             sparse_output=False)
               if onehot else
               OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1, encoded_missing_value=-2))
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", cat_enc),
    ])
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    return ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)],
        remainder="drop")


# --- PyTorch MLP -----------------------------------------------------------

class _Net(nn.Module):
    def __init__(self, d_in: int, d_out: int, width: int, depth: int,
                 dropout: float, gated: bool = False):
        super().__init__()
        # learnable soft feature-selection gates (H8); init mostly open
        self.gate = nn.Parameter(torch.full((d_in,), 2.0)) if gated else None
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D102
        if self.gate is not None:
            x = x * torch.sigmoid(self.gate)
        return self.net(x)


class TorchMLP(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-compatible MLP with early stopping on a val slice.

    The early-stopping validation slice is carved from the TRAINING data
    passed to fit() - the experiment's held-out val/test sets are never
    touched here.
    """

    def __init__(self, width=256, depth=2, dropout=0.1, lr=1e-3,
                 weight_decay=1e-4, seed=0, gate_l1=0.0):
        self.width, self.depth, self.dropout = width, depth, dropout
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.gate_l1 = gate_l1

    def _make_net(self, d_in: int, d_out: int) -> nn.Module:
        return _Net(d_in, d_out, self.width, self.depth, self.dropout,
                    gated=self.gate_l1 > 0)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d_in = X.shape
        d_out = len(self.classes_)

        # internal early-stop split (from training data only)
        perm = rng.permutation(n)
        n_es = max(64, int(0.15 * n))
        es_idx, tr_idx = perm[:n_es], perm[n_es:]
        Xt = torch.tensor(X[tr_idx], device=DEVICE)
        yt = torch.tensor(y_idx[tr_idx], dtype=torch.long, device=DEVICE)
        Xe = torch.tensor(X[es_idx], device=DEVICE)
        ye = torch.tensor(y_idx[es_idx], dtype=torch.long, device=DEVICE)

        self.model_ = self._make_net(d_in, d_out).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        lossf = nn.CrossEntropyLoss()
        gate_pen = (lambda: self.gate_l1
                    * torch.sigmoid(self.model_.gate).mean()) \
            if self.gate_l1 > 0 else (lambda: 0.0)

        best_loss, best_state, patience = np.inf, None, 0
        n_tr = len(tr_idx)
        for _epoch in range(MAX_EPOCHS):
            self.model_.train()
            order = torch.randperm(n_tr, device=DEVICE)
            for i in range(0, n_tr, BATCH_SIZE):
                b = order[i:i + BATCH_SIZE]
                if len(b) < 2:      # BatchNorm needs >=2 samples
                    continue
                opt.zero_grad()
                loss = lossf(self.model_(Xt[b]), yt[b]) + gate_pen()
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                es_loss = float(lossf(self.model_(Xe), ye))
            if es_loss < best_loss - 1e-5:
                best_loss, patience = es_loss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        X = torch.tensor(np.asarray(X, dtype=np.float32), device=DEVICE)
        self.model_.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 4096):
                out.append(torch.softmax(self.model_(X[i:i + 4096]), dim=1))
            return torch.cat(out).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# --- XGBoost label wrapper -------------------------------------------------

class XGBWrapped(BaseEstimator, ClassifierMixin):
    """XGBClassifier demands integer labels in [0..k); encode transparently."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.model_ = XGBClassifier(**self.params)
        self.model_.fit(X, y_idx)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.model_.predict(X)]


# --- config sampling (identical budget across families) --------------------

def sample_config(family: str, rng: np.random.Generator) -> dict:
    if family == "mlp":
        return {
            "width": int(rng.choice([64, 128, 256, 512])),
            "depth": int(rng.choice([1, 2, 3, 4])),
            "dropout": float(rng.choice([0.0, 0.1, 0.25, 0.4])),
            "lr": float(10 ** rng.uniform(-4, -2.3)),
            "weight_decay": float(10 ** rng.uniform(-6, -2)),
        }
    if family == "xgb":
        return {
            "n_estimators": int(rng.choice([200, 500, 1000])),
            "learning_rate": float(10 ** rng.uniform(-2.3, -0.7)),
            "max_depth": int(rng.integers(3, 11)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "min_child_weight": float(10 ** rng.uniform(0, 1.3)),
            "reg_lambda": float(10 ** rng.uniform(-2, 2)),
        }
    if family == "lgbm":
        return {
            "n_estimators": int(rng.choice([200, 500, 1000])),
            "learning_rate": float(10 ** rng.uniform(-2.3, -0.7)),
            "num_leaves": int(rng.choice([15, 31, 63, 127, 255])),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "min_child_samples": int(rng.integers(5, 60)),
            "reg_lambda": float(10 ** rng.uniform(-2, 2)),
        }
    if family in ("resnet", "ftt", "tabpfn"):
        from models_arch import sample_config as arch_config
        return arch_config(family, rng)
    raise ValueError(family)


def build_model(family: str, cfg: dict, cat_cols: list[str],
                num_cols: list[str], seed: int) -> Pipeline:
    if family == "mlp":
        est = TorchMLP(seed=seed, **cfg)
        prep = make_preprocessor(cat_cols, num_cols, onehot=True)
    elif family == "xgb":
        est = XGBWrapped(tree_method="hist", device=DEVICE,
                         random_state=seed, n_jobs=-1,
                         eval_metric="logloss", **cfg)
        prep = make_preprocessor(cat_cols, num_cols, onehot=False)
    elif family == "lgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **cfg)
        prep = make_preprocessor(cat_cols, num_cols, onehot=False)
    elif family in ("resnet", "ftt", "tabpfn"):
        from models_arch import make_estimator
        est = make_estimator(family, cfg, seed)
        prep = make_preprocessor(cat_cols, num_cols,
                                 onehot=family != "tabpfn")
    else:
        raise ValueError(family)
    return Pipeline([("prep", prep), ("clf", est)])
