# How uninformative features and feature rotation change the gap between neural networks and gradient-boosted trees

Code, results and pre-specification for the manuscript *"How uninformative
features and feature rotation change the gap between neural networks and
gradient-boosted trees on tabular data: pre-specified experiments with full
re-tuning"* (M. Grabowski, Rzeszów University of Technology).

The study compares a tuned MLP, a ResNet and an FT-Transformer with tuned
XGBoost and LightGBM on 61 OpenML-CC18 classification datasets, modifies the
datasets in controlled ways (added uninformative features, feature removal,
rotation, discretisation) and measures how the test-AUC gap between each
network and the best GBDT changes.

## Repository layout

| Path | Content |
|---|---|
| `src/` | all code (one script per analysis step, see below) |
| `results/<id>.json` | baseline comparison per dataset (30 configurations per family, 5 seeds) |
| `results/descriptors/` | the 22 dataset descriptors per dataset |
| `results/interventions/` | protocol A cells (frozen configurations, 3 seeds) |
| `results/arch/` | protocol A for ResNet and FT-Transformer |
| `results/h13/` | protocol B cells (every family re-tuned on every modified dataset, 5 seeds) |
| `results/h13_prereg.json` | **pre-specification of protocol B**: hypotheses, list of 566 cells, SHA-256 of the code |
| `results/*_report.json` | analysis reports used in the manuscript |
| `data/<id>/splits.npz`, `meta.json` | frozen train/validation/test indices and metadata (features themselves are re-downloaded from OpenML) |

The manuscript source is not included until publication.
`src/paper_tables.py` and `src/make_figures.py` regenerate every table row
and figure of the manuscript into `paper/tables/` and `paper/figures/`.

## Reproducing

```bash
pip install -r requirements.txt
cd src
python data_prep.py --full            # download OpenML-CC18, deduplicate, split
python train_baseline.py --full       # baseline: MLP, XGBoost, LightGBM
python verify_results.py              # leakage checks (label-shuffle canary)
python descriptors.py                 # 22 descriptors per dataset
python meta_learner.py                # meta-model (LODO)
python h1b_analysis.py; python h1c_regression_test.py
python interventions.py               # protocol A, pilot cohort
python h2_analysis.py; python h2s_sensitivity.py; python h5_mechanism.py
python h6_analysis.py; python h7_rescue.py; python h8_gated.py
python h9_replication.py; python h10_realistic_noise.py
python h11_architectures.py; python h11_architectures.py --report   # ResNet, FT-Transformer
python h12_revision_stats.py          # tie thresholds, clustered tests, decision analysis
python h13_retuned.py --families mlp  # protocol B (repeat for resnet, xgb, lgbm, ftt)
python h13_analysis.py
python h14_natural_link.py            # exploratory
python paper_tables.py; python make_figures.py
```

`src/audit_claims.py` (checks every number quoted in the manuscript against
`results/`) and `src/make_blinded.py` need the manuscript source and are
included for transparency.

`h13_retuned.py` refuses to run if any of the hashed source files differ
from the hashes stored in `results/h13_prereg.json`. Finished work units are
skipped, so every step can be interrupted and resumed. Heavy output goes to
`logs/training.log`.

Note on terminology: code comments written during the study use
"pre-registered"; the manuscript uses the more precise term
"pre-specified", because the hypotheses were fixed in code and in a file
before execution, not in a public registry.

## Use of generative AI

A large language model (Claude, Anthropic) assisted in writing and reviewing
the code and in drafting the manuscript; see the manuscript for details.

## Licence

Code: MIT (see `LICENSE`). Datasets: OpenML, under their original licences.
