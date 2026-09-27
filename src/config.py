"""Central configuration for the NN-vs-GBDT meta-learning study.

All experiment-wide constants live here so that every script (data_prep,
train_baseline, verify_results, discovery loop) shares one source of truth.
"""
from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # cached datasets (parquet + split indices)
RESULTS_DIR = ROOT / "results"    # incremental per-dataset result JSONs
CHECKPOINT_DIR = ROOT / "checkpoints"  # best-config model weights per dataset
LOG_FILE = ROOT / "logs" / "training.log"
STATE_FILE = ROOT / "discovery_state.json"

for d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, LOG_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)

# --- dataset suites --------------------------------------------------------
# Smoke suite: tiny, fast, used to validate the full pipeline end-to-end
# before any heavy computation is authorized.
SMOKE_SUITE = [
    31,     # credit-g (1000 x 20, binary)
    37,     # diabetes (768 x 8, binary)
    54,     # vehicle (846 x 18, 4-class)
]

# Full suite is materialized by data_prep.py from the OpenML-CC18 benchmark
# (study id 99), filtered by the size limits below. Kept as a function of the
# live OpenML metadata rather than a hardcoded list so the filter criteria
# are auditable.
OPENML_STUDY_ID = 99          # OpenML-CC18 curated classification benchmark
MAX_ROWS = 50_000             # subsample above this (seeded, stratified)
MAX_FEATURES = 500
MIN_ROWS = 500

# --- split protocol --------------------------------------------------------
SPLIT_SEED = 20260717         # frozen: date of protocol definition
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

# --- tuning budgets (identical budget for every model family) -------------
N_TUNING_CONFIGS = 30         # random-search configs per model per dataset
FINAL_EVAL_SEEDS = [0, 1, 2, 3, 4]  # best config re-trained on these seeds
TUNING_SEED = 1234

# --- MLP training ----------------------------------------------------------
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
BATCH_SIZE = 256

# --- metrics ---------------------------------------------------------------
PRIMARY_METRIC = "roc_auc_ovr"   # AUC (one-vs-rest for multiclass)
SECONDARY_METRICS = ["accuracy", "log_loss"]
