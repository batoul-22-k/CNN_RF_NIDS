# CNN+RF NIDS Project Guide (Step by Step)

## 1) What this project is

This project reproduces a network intrusion detection workflow inspired by the 2025 Scientific Reports paper:

- A 1D CNN is used as a feature extractor.
- A Random Forest (RF) is trained on the CNN embedding.
- Multiple baseline models are run for comparison.
- Results are exported in paper-style tables and plots.

Datasets supported:

- KDD99
- UNSW-NB15

---

## 2) Repository map

### Core entry points

- `scripts/run_kdd99.py`: Run KDD99 pipeline only.
- `scripts/run_unsw_nb15.py`: Run UNSW-NB15 pipeline only.
- `scripts/run_all.py`: Run KDD99 then UNSW in one command.

### Core logic (`src/`)

- `src/pipeline.py`: Main orchestration for train/eval/export.
- `src/data.py`: Data loading, label mapping, splitting, preprocessing, resampling.
- `src/model_cnn.py`: CNN architecture and embedding extraction.
- `src/model_rf.py`: RF training and optional grid search.
- `src/baselines.py`: Baseline models (RF, PCA+RF, GA+RF, C4.5-like, NBTree proxy, SVM).
- `src/metrics.py`: Accuracy/precision/recall/F1/confusion matrix/ROC-AUC.
- `src/plots.py`: Metric, ROC, and runtime plots.
- `src/utils_seed.py`: Reproducibility and version metadata.

### Other important files

- `README.md`: Setup and basic usage.
- `tips_to_match_paper.md`: Practical guidance to reproduce paper-like results.
- `tests/test_smoke.py`: Smoke tests for preprocessing and CNN+RF flow.
- `requirements.txt`: Dependencies.

---

## 3) End-to-end execution flow

This is the exact high-level sequence when you run a script.

1. Parse CLI arguments in `scripts/run_*.py`.
2. Create a timestamped run directory under `runs/`.
3. Call `run_single_split(...)` or `run_crossval(...)` in `src/pipeline.py`.
4. Set random seeds and save `run_metadata.json`.
5. Load dataset (`load_kdd99` or `load_unsw_nb15`).
6. Split data.
7. If no original test split is used, do stratified 80/10/10 split.
8. If UNSW original split is enabled, keep original test and split train into train/val.
9. Fit preprocessing on train only.
10. One-hot encode categorical features.
11. Min-max scale all encoded features.
12. Transform val and test with the same fitted preprocessors.
13. Optionally rebalance training data (`none`, `over`, `under`).
14. Train CNN and extract embedding vectors.
15. Train RF on embeddings and evaluate on test.
16. Run baseline models and evaluate on test.
17. Aggregate results and feature counts.
18. Save JSON and markdown reports.
19. Save plots (metrics bar, ROC curves, runtime bar).

---

## 4) Script-level behavior

### `scripts/run_kdd99.py`

- Supports single split and k-fold CV.
- Default `feature_k=18`.
- Calls:
  - `run_single_split(dataset='kdd99', ...)` or
  - `run_crossval(dataset='kdd99', ...)`.

### `scripts/run_unsw_nb15.py`

- Supports single split and k-fold CV.
- Default `feature_k=22`.
- Supports `--use_original_split true|false`.
- Calls pipeline with `dataset='unsw'`.

### `scripts/run_all.py`

- Creates one base run folder with two subfolders:
  - `kdd99/`
  - `unsw_nb15/`
- Runs two single-split jobs sequentially with fixed `feature_k`:
  - KDD99 with 18
  - UNSW with 22

---

## 5) Data handling details (`src/data.py`)

### KDD99 loading

`load_kdd99(...)` does:

1. Search for local file names:
   - `kddcup.data`
   - `kddcup.data_10_percent`
   - `kdd99.csv`
   - `kddcup99.csv`
2. If not found, fallback to `sklearn.datasets.fetch_kddcup99`.
3. Decode byte columns and labels.
4. Optionally remove duplicate rows.
5. Map labels:
   - Binary: `normal -> 0`, all attacks -> `1`.
   - Multi-class: maps attacks into `dos/probe/r2l/u2r/other`.
6. Optional stratified downsampling with `max_rows`.
7. Return `df`, `y`, and metadata (categorical/numeric columns and label mapping).

### UNSW-NB15 loading

`load_unsw_nb15(...)` does:

1. Find split files:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
2. It now auto-discovers these files in:
   - `data_dir` directly
   - `data_dir/unsw-nb15`
   - `data_dir/UNSW-NB15`
   - `data_dir/unsw_nb15`
   - or recursively under `data_dir`
3. Read train/test CSVs.
4. If `use_original_split` is false, concatenate train+test and later do stratified split.
5. Drop `id` column if present.
6. Label handling:
   - Binary mode uses `label`.
   - Multi-class mode prefers `attack_cat` (with `LabelEncoder`), drops `attack_cat` and `label`.
7. Optional downsampling with `max_rows`.
8. Return `df`, `y`, and metadata (feature types + optional original test set).

### Splitting and preprocessing

- `stratified_split`: 80/10/10 split using `StratifiedShuffleSplit`.
- `stratified_sample`: class-preserving downsampling if `max_rows` is set.
- `Preprocessor`:
  - One-hot encode categoricals.
  - Concatenate numeric + categorical.
  - Min-max scale.
  - Track `feature_count_` after encoding.

### Resampling

`balance_resample(X, y, method, seed)`:

- `none`: unchanged.
- `over`: oversample minority classes to majority size.
- `under`: undersample majority classes to minority size.

---

## 6) CNN feature extractor (`src/model_cnn.py`)

#### Input shape

- Input matrix after preprocessing: `(N, D)`.
- CNN reshape: `(N, D, 1)`.

#### Architecture

1. `Conv1D(32, kernel_size=3, relu, same)`
2. `MaxPool1D(2)`
3. `Conv1D(64, kernel_size=3, relu, same)`
4. `MaxPool1D(2)`
5. `Flatten`
6. `Dense(embedding_dim, relu, name='embedding')`
7. Output head:
   - Binary: `Dense(1, sigmoid)` + `binary_crossentropy`
   - Multi-class: `Dense(num_classes, softmax)` + `sparse_categorical_crossentropy`

#### Training

- Optimizer: Adam (`lr=1e-3`)
- Defaults: `epochs=100`, `batch_size=128`, `patience=10`
- Early stopping on `val_loss`, restore best weights.

#### Embedding extraction

- `build_embedding_model(model)` returns sub-model outputting the `embedding` layer.
- `extract_embeddings` runs `predict` on train/val/test.

---

## 7) RF model (`src/model_rf.py`)

- Default RF params:
  - `n_estimators=50`
  - `max_depth=20`
  - `max_leaf_nodes=1500`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `criterion='gini'`
- `train_rf_model(...)`:
  - If `use_gridsearch=False`: train RF directly.
  - If `use_gridsearch=True`: `GridSearchCV(cv=3, n_jobs=-1)` over:
    - `n_estimators: [50, 100]`
    - `max_depth: [10, 20]`
    - `max_leaf_nodes: [500, 1500]`

---

## 8) Baselines (`src/baselines.py`)

`run_all_baselines(...)` runs:

1. `RF (all features)`
2. `PCA+RF`
3. `GA+RF`
4. `C4.5-like` (entropy Decision Tree)
5. `NBTree-proxy` (tree leaves one-hot + GaussianNB)
6. `SVM (RBF)` with `probability=True`

### Runtime note

Two baselines are usually the slowest on larger datasets:

- `GA+RF` (many repeated RF fits inside genetic search)
- `SVM (RBF)` (kernel SVM scaling cost)

---

## 9) Metrics and plots

### Metrics (`src/metrics.py`)

- Always computed:
  - `accuracy`
  - `precision`
  - `recall`
  - `f1`
  - `confusion_matrix`
- Binary mode with probabilities also adds:
  - `roc_auc`
  - `roc_curve` (`fpr`, `tpr`)

### Plots (`src/plots.py`)

- `metrics_bar.png`: grouped bars for accuracy/precision/recall/F1.
- `roc_curves.png`: ROC per model (only if curves available).
- `time_bar.png`: train + inference seconds per model.

---

## 10) Pipeline orchestration (`src/pipeline.py`)

### `run_single_split(...)`

1. Set seed and save metadata.
2. Load selected dataset.
3. Create train/val/test split depending on mode.
4. Preprocess train/val/test.
5. Optional resampling on training set.
6. Train and evaluate `CNN+RF`.
7. Train and evaluate all baselines.
8. Build `feature_counts`:
   - `raw`: original feature count
   - `encoded`: post one-hot feature count
   - `reduced`: embedding size (`feature_k`)
9. Save:
   - `results.json`
   - `paper_results.md`
   - plots

### `run_crossval(...)`

- Uses `StratifiedKFold`.
- Runs only `CNN+RF` per fold.
- Saves:
  - `cv_results.json` with per-fold and mean metrics
  - `paper_results.md`
  - `metrics_bar.png` for mean result

---

## 11) Reproducibility (`src/utils_seed.py`)

- Sets:
  - `PYTHONHASHSEED`
  - `TF_DETERMINISTIC_OPS=1`
  - `TF_CUDNN_DETERMINISTIC=1`
- Seeds:
  - Python `random`
  - NumPy
  - TensorFlow
- Tries enabling TensorFlow op determinism.
- Saves `run_metadata.json` with versions:
  - Python, platform, numpy, pandas, sklearn, tensorflow

---

## 12) Outputs produced per run

For a single split run directory:

- `results.json`: raw machine-readable metrics and model metadata.
- `paper_results.md`: table formatted for reporting.
- `run_metadata.json`: seed + package versions.
- `metrics_bar.png`
- `roc_curves.png`
- `time_bar.png`

For `run_all.py`, outputs are separated into:

- `runs/<timestamp>/kdd99/...`
- `runs/<timestamp>/unsw_nb15/...`

---

## 13) Tests

`tests/test_smoke.py` includes:

1. `test_preprocessor_smoke`
   - Verifies preprocessing pipeline shape behavior.
2. `test_cnn_rf_smoke`
   - Trains tiny CNN for 1 epoch on random data.
   - Extracts embeddings.
   - Trains RF.
   - Verifies metrics output includes accuracy.

This is a sanity test suite, not a full correctness/performance benchmark.

---

## 14) Typical commands

### Setup

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Reproduce defaults

```bash
python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18
python scripts/run_unsw_nb15.py --data_dir data --binary true --feature_k 22
python scripts/run_all.py --data_dir data --binary true
```

### Faster smoke runs

```bash
python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18 --max_rows 50000
python scripts/run_unsw_nb15.py --data_dir data --binary true --feature_k 22 --max_rows 50000
```

---

## 15) Common warnings/errors and what they mean

### TensorFlow oneDNN info logs

- Message about oneDNN custom ops and floating-point round-off.
- Usually informational, not a failure.

### TensorFlow `use_unbounded_threadpool` NodeDef warning

- Indicates unknown attribute in graph nodes, usually ignored by runtime.
- Often noisy but non-fatal if run completes.

### Matplotlib `set_ticklabels` warning

- Previously triggered in `plot_time_bars`.
- Project code now uses safer tick label rotation handling.

### `UNSW-NB15 CSV files not found in data_dir`

- Means split CSV files were not found.
- Loader now searches common UNSW subfolders under the provided `data_dir`.

---

## 16) Performance expectations

Long runtimes are normal on full-size runs because:

1. CNN training can run up to 100 epochs.
2. `GA+RF` performs many repeated RF fits.
3. `SVM (RBF)` is expensive on larger sample sizes.
4. `run_all.py` trains both datasets sequentially.

Use `--max_rows` for faster checks before full experiments.

---

## 17) How to extend the project safely

1. Add a new model in `src/baselines.py` or alongside `model_cnn.py`/`model_rf.py`.
2. Plug it into `run_all_baselines` or `run_single_split`.
3. Return metrics in the same schema as existing models.
4. Add at least one smoke test in `tests/test_smoke.py`.
5. Keep seed handling through `set_global_seed`.
6. Keep output contracts (`results.json`, `paper_results.md`) stable.

---

## 18) Key design choices summary

1. CNN is used for feature learning, not final classification.
2. RF is the final classifier on learned embeddings.
3. Preprocessing is strictly fit on training data.
4. Baselines are included to compare feature-selection/classifier strategies.
5. Reproducibility metadata is persisted for each run.
6. Artifacts are organized by timestamped run directories for traceability.






# MY NOTES
is 50000 max rows enough? shortly answer

For a quick, reliable comparison run: yes, --max_rows 50000 is enough.
For final paper-level results: no, use full data (or much larger).


how much larger?

I’ll give you concrete numbers by checking the local dataset sizes quickly, then recommend a practical target above 50k.

Ran .\.venv\Scripts\python -c "import pandas as pd; a='data/unsw-nb15/UNSW_NB15_training-set.csv'; b='data/unsw-nb15/UNSW_NB15_testing-set.csv'; print('unsw_train',sum(1 for _ in open(a,encoding='utf-8'))-1); print('unsw_test',sum(1 for _ in open(b,encoding='utf-8'))-1);" for 23s


Ran .\.venv\Scripts\python -c "from sklearn.datasets import fetch_kddcup99; b=fetch_kddcup99(data_home='data', percent10=False); print('kdd_rows', b.data.shape[0])" for 8s


Worked for 1m 5s
Good target: 3-5x larger than 50k.

UNSW-NB15: use full data, about 257,673 rows (train+test).
KDD99: start around 200k-500k for near-final runs; full is about 4,898,431 rows (very slow).