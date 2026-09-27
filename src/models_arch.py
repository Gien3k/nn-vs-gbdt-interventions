"""Additional neural families for the revision (reviewer request: the
original study compared a single MLP against GBDTs).

  - resnet: tabular ResNet of Gorishniy et al. (2021)
  - ftt:    FT-Transformer of Gorishniy et al. (2021); every preprocessed
            column (standardised numeric or one-hot indicator) becomes one
            token through a per-feature linear tokenizer
  - tabpfn: TabPFN v2 (Hollmann et al. 2025), used as-is without tuning

ResNet and FT-Transformer reuse TorchMLP's training loop unchanged (AdamW,
early stopping on a 15% slice carved from the training split, identical
epoch/patience/batch constants), so the three trained networks differ only
in architecture and search space.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from models import DEVICE, TorchMLP


# --- ResNet ----------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int, hidden_dropout: float,
                 residual_dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.lin1 = nn.Linear(d, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d)
        self.hd, self.rd = nn.Dropout(hidden_dropout), nn.Dropout(residual_dropout)

    def forward(self, x):  # noqa: D102
        z = self.hd(F.relu(self.lin1(self.norm(x))))
        return x + self.rd(self.lin2(z))


class _ResNet(nn.Module):
    def __init__(self, d_in, d_out, width, depth, hidden_factor,
                 hidden_dropout, residual_dropout):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[
            _ResBlock(width, int(width * hidden_factor), hidden_dropout,
                      residual_dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.BatchNorm1d(width), nn.ReLU(),
                                  nn.Linear(width, d_out))

    def forward(self, x):  # noqa: D102
        return self.head(self.blocks(self.inp(x)))


class TorchResNet(TorchMLP):
    def __init__(self, width=256, depth=2, hidden_factor=2.0,
                 hidden_dropout=0.1, residual_dropout=0.0, lr=1e-3,
                 weight_decay=1e-4, seed=0):
        super().__init__(width=width, depth=depth, dropout=hidden_dropout,
                         lr=lr, weight_decay=weight_decay, seed=seed)
        self.hidden_factor = hidden_factor
        self.hidden_dropout = hidden_dropout
        self.residual_dropout = residual_dropout

    def _make_net(self, d_in, d_out):
        return _ResNet(d_in, d_out, self.width, self.depth,
                       self.hidden_factor, self.hidden_dropout,
                       self.residual_dropout)


# --- FT-Transformer --------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, d, n_heads, attn_dropout, ffn_dropout,
                 residual_dropout, first: bool):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.Identity() if first else nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        d_ffn = int(d * 4 / 3)
        self.ffn1 = nn.Linear(d, 2 * d_ffn)          # ReGLU
        self.ffn2 = nn.Linear(d_ffn, d)
        self.attn_dropout = attn_dropout
        self.fd, self.rd = nn.Dropout(ffn_dropout), nn.Dropout(residual_dropout)

    def forward(self, x):  # noqa: D102
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.rd(self.out(a.transpose(1, 2).reshape(b, t, d)))
        h, gate = self.ffn1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.rd(self.ffn2(self.fd(h * F.relu(gate))))


class _FTT(nn.Module):
    def __init__(self, d_in, d_out, d_token, n_blocks, attn_dropout,
                 ffn_dropout, residual_dropout, n_heads=8):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_in, d_token))
        self.b = nn.Parameter(torch.empty(d_in, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        for p in (self.w, self.b, self.cls):
            nn.init.uniform_(p, -d_token ** -0.5, d_token ** -0.5)
        self.blocks = nn.ModuleList([
            _Block(d_token, n_heads, attn_dropout, ffn_dropout,
                   residual_dropout, first=(i == 0))
            for i in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, d_out))

    def forward(self, x):  # noqa: D102
        tok = x.unsqueeze(-1) * self.w + self.b
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(tok[:, 0])


class TorchFTT(TorchMLP):
    def __init__(self, d_token=128, n_blocks=2, attn_dropout=0.1,
                 ffn_dropout=0.1, residual_dropout=0.0, lr=1e-4,
                 weight_decay=1e-5, seed=0):
        super().__init__(width=d_token, depth=n_blocks, dropout=ffn_dropout,
                         lr=lr, weight_decay=weight_decay, seed=seed)
        self.d_token, self.n_blocks = d_token, n_blocks
        self.attn_dropout, self.ffn_dropout = attn_dropout, ffn_dropout
        self.residual_dropout = residual_dropout

    def _make_net(self, d_in, d_out):
        return _FTT(d_in, d_out, self.d_token, self.n_blocks,
                    self.attn_dropout, self.ffn_dropout,
                    self.residual_dropout)


# --- TabPFN v2 -------------------------------------------------------------

TABPFN_MAX_TRAIN = 10_000


class TabPFNWrapped(BaseEstimator, ClassifierMixin):
    """TabPFN v2 without tuning. Training sets above TABPFN_MAX_TRAIN rows
    are stratified-subsampled (seeded) to the model's supported size."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(X) > TABPFN_MAX_TRAIN:
            from sklearn.model_selection import train_test_split
            X, _, y_idx, _ = train_test_split(
                X, y_idx, train_size=TABPFN_MAX_TRAIN, stratify=y_idx,
                random_state=self.seed)
        self.model_ = TabPFNClassifier(device=DEVICE, random_state=self.seed)
        self.model_.fit(X, y_idx)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# --- search spaces ---------------------------------------------------------

def sample_config(family: str, rng: np.random.Generator) -> dict:
    if family == "resnet":
        return {
            "width": int(rng.choice([64, 128, 256, 512])),
            "depth": int(rng.choice([1, 2, 3, 4])),
            "hidden_factor": float(rng.choice([1.0, 2.0, 3.0, 4.0])),
            "hidden_dropout": float(rng.choice([0.0, 0.1, 0.25, 0.4])),
            "residual_dropout": float(rng.choice([0.0, 0.1, 0.2])),
            "lr": float(10 ** rng.uniform(-4, -2.3)),
            "weight_decay": float(10 ** rng.uniform(-6, -2)),
        }
    if family == "ftt":
        return {
            "d_token": int(rng.choice([64, 128, 192])),
            "n_blocks": int(rng.choice([1, 2, 3])),
            "attn_dropout": float(rng.choice([0.0, 0.1, 0.2])),
            "ffn_dropout": float(rng.choice([0.0, 0.1, 0.25])),
            "residual_dropout": float(rng.choice([0.0, 0.1])),
            "lr": float(10 ** rng.uniform(-4.5, -3)),
            "weight_decay": float(10 ** rng.uniform(-6, -3)),
        }
    if family == "tabpfn":
        return {}
    raise ValueError(family)


def make_estimator(family: str, cfg: dict, seed: int):
    if family == "resnet":
        return TorchResNet(seed=seed, **cfg)
    if family == "ftt":
        return TorchFTT(seed=seed, **cfg)
    if family == "tabpfn":
        return TabPFNWrapped(seed=seed)
    raise ValueError(family)
