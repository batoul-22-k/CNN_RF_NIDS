# CNN+RF NIDS (KDD99, UNSW-NB15)

Reimplementation of the Scientific Reports 2025 paper A new intrusion detection method using ensemble classification and feature selection using a CNN for feature extraction and a Random Forest classifier. Includes baselines, paper-style metrics, plots, and deterministic runs.

## Setup
Python 3.9 or 3.10 is required (TensorFlow 2.9 does not support 3.11+).
```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Data

### KDD99
Default uses `sklearn.datasets.fetch_kddcup99`. 
You can also provide a local CSV in `--data_dir` with one of these names:
- `kddcup.data`
- `kddcup.data_10_percent`
- `kdd99.csv`
- `kddcup99.csv`

### UNSW-NB15
Place these in `--data_dir`:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Reproduce (paper-style defaults)
```bash
python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18
python scripts/run_unsw_nb15.py --data_dir data --binary true --feature_k 22
python scripts/run_all.py --data_dir data --binary true
```

## Useful options
```bash
--max_rows 50000 # quick smoke run
--cv_folds 5 # 5-fold CV (CNN+RF only)
--use_original_split true # use UNSW official train/test split
--resample over # handle imbalance on training set
```

## Outputs
Artifacts are saved under `runs/<timestamp>/`:
- `results.json`
- `paper_results.md`
- `run_metadata.json`
- `metrics_bar.png`, `roc_curves.png`, `time_bar.png`
