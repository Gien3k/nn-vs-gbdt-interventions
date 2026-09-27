"""Dataset acquisition and leakage-safe splitting.

Anti-leakage design decisions (verified later by verify_results.py):
  1. Rows are DEDUPLICATED before splitting - identical rows landing in both
     train and test are a classic silent leakage vector.
  2. Split indices are computed ONCE, seeded, stratified, and frozen to disk;
     every downstream script consumes the same frozen indices.
  3. No preprocessing happens here beyond type coercion - imputation, scaling
     and encoding are fitted strictly inside training folds (models.py uses
     sklearn Pipelines), so no test-set statistic can ever reach a model.

Usage:
  python data_prep.py --smoke          # 3 tiny datasets, pipeline validation
  python data_prep.py --full           # OpenML-CC18 filtered suite
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_DIR, MAX_FEATURES, MAX_ROWS, MIN_ROWS,
                    OPENML_STUDY_ID, SMOKE_SUITE, SPLIT_SEED, TEST_FRAC,
                    TRAIN_FRAC, VAL_FRAC)
from log_utils import log


def _row_hashes(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Stable per-row content hash used for cross-split duplicate detection.

    NaNs (incl. in Categorical columns) are mapped to a sentinel first -
    astype(str) alone leaves float NaN behind and breaks string joins.
    """
    Xs = X.astype(object).where(X.notna(), "<NA>").astype(str)
    joined = Xs.agg("|".join, axis=1) + "||" + y.astype(str)
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def prepare_dataset(dataset_id: int) -> dict | None:
    """Download one OpenML dataset, dedupe, split, freeze to disk."""
    out_dir = DATA_DIR / str(dataset_id)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        log(f"[data_prep] {dataset_id}: cached, skipping")
        return json.loads(meta_path.read_text())

    ds = openml.datasets.get_dataset(dataset_id, download_data=True,
                                     download_qualities=False,
                                     download_features_meta_data=True)
    X, y, cat_mask, _ = ds.get_data(target=ds.default_target_attribute,
                                    dataset_format="dataframe")
    if y is None or X.shape[1] == 0:
        log(f"[data_prep] {dataset_id}: no target/features, SKIP")
        return None

    # Drop rows with missing target; features keep NaNs (imputers handle them).
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep]

    # --- leakage guard #1: content-level deduplication ---------------------
    hashes = _row_hashes(X, y)
    dup_count = int(hashes.duplicated().sum())
    keep_idx = ~hashes.duplicated()
    X, y = X.loc[keep_idx].reset_index(drop=True), y.loc[keep_idx].reset_index(drop=True)

    n, p = X.shape
    n_classes = y.nunique()
    if n < MIN_ROWS or p > MAX_FEATURES or n_classes < 2:
        log(f"[data_prep] {dataset_id}: fails size filter (n={n}, p={p}, k={n_classes}), SKIP")
        return None
    min_class = y.value_counts().min()
    if min_class < 10:
        log(f"[data_prep] {dataset_id}: rarest class has {min_class} rows, SKIP")
        return None

    # Seeded stratified subsample of oversized datasets.
    if n > MAX_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS,
                                      stratify=y, random_state=SPLIT_SEED)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        n = MAX_ROWS

    # --- leakage guard #2: frozen stratified 60/20/20 split ----------------
    idx = np.arange(n)
    idx_train, idx_rest = train_test_split(
        idx, train_size=TRAIN_FRAC, stratify=y, random_state=SPLIT_SEED)
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    idx_val, idx_test = train_test_split(
        idx_rest, train_size=rel_val, stratify=y.iloc[idx_rest],
        random_state=SPLIT_SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_dir / "X.parquet")
    y.rename("target").to_frame().to_parquet(out_dir / "y.parquet")
    np.savez(out_dir / "splits.npz", train=idx_train, val=idx_val, test=idx_test)

    meta = {
        "dataset_id": dataset_id,
        "name": ds.name,
        "n_rows": int(n),
        "n_features": int(p),
        "n_classes": int(n_classes),
        "categorical_cols": [c for c, is_cat in zip(X.columns, cat_mask) if is_cat],
        "duplicates_removed": dup_count,
        "split_seed": SPLIT_SEED,
        "split_sizes": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"[data_prep] {dataset_id} ({ds.name}): n={n} p={p} k={n_classes} "
        f"dups_removed={dup_count} -> frozen")
    return meta


def full_suite_ids() -> list[int]:
    """Dataset ids of the OpenML-CC18 study (filtering happens per-dataset)."""
    study = openml.study.get_suite(OPENML_STUDY_ID)
    return sorted(study.data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    ids = SMOKE_SUITE if args.smoke else (full_suite_ids() if args.full else [])
    if not ids:
        ap.error("choose --smoke or --full")

    prepared = []
    for did in ids:
        try:
            meta = prepare_dataset(did)
            if meta:
                prepared.append(did)
        except Exception as e:  # noqa: BLE001 - log-and-continue by design
            log(f"[data_prep] {did}: ERROR {type(e).__name__}: {e}")
    log(f"[data_prep] done: {len(prepared)}/{len(ids)} datasets ready")
    print(json.dumps({"prepared": prepared}))


if __name__ == "__main__":
    sys.exit(main())
