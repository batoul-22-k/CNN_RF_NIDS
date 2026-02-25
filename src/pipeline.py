import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .baselines import run_all_baselines
from .data import Preprocessor, balance_resample, load_kdd99, load_unsw_nb15, stratified_split
from .logging_utils import get_logger
from .metrics import compute_metrics
from .model_cnn import build_cnn_model, build_embedding_model, extract_embeddings, reshape_for_cnn, train_cnn_model
from .model_rf import DEFAULT_RF_PARAMS, train_rf_model
from .plots import plot_metric_bars, plot_roc_curves, plot_time_bars
from .utils_seed import save_run_metadata, set_global_seed

logger = get_logger(__name__)


def make_run_dir(base_dir: str = 'runs', tag: str = '') -> str:
	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	name = ts if not tag else f'{ts}_{tag}'
	run_dir = os.path.join(base_dir, name)
	os.makedirs(run_dir, exist_ok=True)
	logger.info('Created run directory: %s', run_dir)
	return run_dir


def _save_json(path: str, obj: dict) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(obj, f, indent=2)


def _save_text(path: str, text: str) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		f.write(text)


def _render_paper_results(
	dataset_name: str,
	split_desc: str,
	feature_counts: Dict,
	results: List[Dict],
) -> str:
	lines = []
	lines.append('# Paper-style Results')
	lines.append(f'Dataset: {dataset_name}')
	lines.append(f'Split: {split_desc}')
	raw = feature_counts['raw']
	encoded = feature_counts['encoded']
	reduced = feature_counts['reduced']
	lines.append(f'Features: raw={raw}, encoded={encoded}, reduced={reduced}')
	lines.append('')
	lines.append('|Model|Accuracy|Precision|Recall|F1|ROC_AUC|Train_time_s|Infer_time_s|')
	lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
	for r in results:
		name = r['name']
		acc = r.get('accuracy', 0)
		prec = r.get('precision', 0)
		rec = r.get('recall', 0)
		f1 = r.get('f1', 0)
		roc_auc = r.get('roc_auc', float('nan'))
		train_time = r.get('train_time', 0)
		infer_time = r.get('infer_time', 0)
		lines.append(
			f'|{name}|{acc:.4f}|{prec:.4f}|{rec:.4f}|{f1:.4f}|{roc_auc:.4f}|{train_time:.2f}|{infer_time:.2f}|'
		)
	return '\n'.join(lines)


def _run_cnn_rf(
	X_train,
	y_train,
	X_val,
	y_val,
	X_test,
	y_test,
	feature_k,
	average,
	use_gridsearch,
	seed,
):
	logger.info('Running CNN+RF pipeline...')
	num_classes = len(np.unique(y_train))
	X_train_cnn = reshape_for_cnn(X_train)
	X_val_cnn = reshape_for_cnn(X_val)
	X_test_cnn = reshape_for_cnn(X_test)

	model = build_cnn_model(
		n_features=X_train.shape[1],
		num_classes=num_classes,
		embedding_dim=feature_k,
		lr=1e-3,
	)
	model, history, cnn_train_time = train_cnn_model(
		model, X_train_cnn, y_train, X_val_cnn, y_val, epochs=100, batch_size=128, patience=10
	)
	logger.info('CNN training stage done.')
	embed_model = build_embedding_model(model)

	t_embed = time.perf_counter()
	X_train_emb = extract_embeddings(embed_model, X_train_cnn)
	X_val_emb = extract_embeddings(embed_model, X_val_cnn)
	X_test_emb = extract_embeddings(embed_model, X_test_cnn)
	embed_time = time.perf_counter() - t_embed
	logger.info('Embedding extraction done in %.2f s', embed_time)

	rf, rf_params = train_rf_model(
		X_train_emb,
		y_train,
		DEFAULT_RF_PARAMS,
		use_gridsearch=use_gridsearch,
		seed=seed,
	)

	t_inf = time.perf_counter()
	y_pred = rf.predict(X_test_emb)
	y_prob = rf.predict_proba(X_test_emb)[:, 1] if average == 'binary' else None
	infer_time = time.perf_counter() - t_inf

	metrics = compute_metrics(y_test, y_pred, y_prob, average=average)
	result = {
		'name': 'CNN+RF',
		'train_time': cnn_train_time + embed_time,
		'infer_time': infer_time,
		'rf_params': rf_params,
		'cnn_history': history,
	}
	result.update(metrics)
	logger.info('CNN+RF evaluation complete: f1=%.4f acc=%.4f', result['f1'], result['accuracy'])
	return result


def run_single_split(
	dataset: str,
	data_dir: str,
	binary: bool,
	feature_k: int,
	seed: int,
	max_rows: int,
	resample: str,
	use_gridsearch: bool,
	use_original_split: bool,
	out_dir: str,
):
	logger.info(
		'Starting single split run (dataset=%s, binary=%s, feature_k=%d, seed=%d, max_rows=%s, resample=%s)',
		dataset,
		binary,
		feature_k,
		seed,
		max_rows,
		resample,
	)
	set_global_seed(seed)
	save_run_metadata(out_dir, seed)

	if dataset == 'kdd99':
		df, y, meta = load_kdd99(data_dir=data_dir, binary=binary, max_rows=max_rows, seed=seed)
	elif dataset in ('unsw', 'unsw_nb15'):
		df, y, meta = load_unsw_nb15(
			data_dir=data_dir, binary=binary, max_rows=max_rows, seed=seed, use_original_split=use_original_split
		)
	else:
		raise ValueError('Unknown dataset.')
	logger.info('Dataset loaded: rows=%d, columns=%d', len(df), df.shape[1])

	if meta.get('original_test') is None:
		df_train, df_val, df_test, y_train, y_val, y_test = stratified_split(df, y, seed)
	else:
		df_train = df
		y_train = y
		df_test = meta['original_test']
		if binary:
			y_test = df_test['label'].to_numpy()
			df_test = df_test.drop(columns=['label'])
		else:
			if 'attack_cat' in df_test.columns:
				y_test_raw = df_test['attack_cat'].astype(str).to_numpy()
				mapping = meta['label_mapping']
				missing = [v for v in np.unique(y_test_raw) if v not in mapping]
				if missing:
					raise ValueError(f'Unseen attack_cat in test set: {missing}')
				y_test = np.array([mapping[v] for v in y_test_raw])
				df_test = df_test.drop(columns=['attack_cat', 'label'])
			else:
				y_test = df_test['label'].to_numpy()
				df_test = df_test.drop(columns=['label'])

		splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
		train_idx, val_idx = next(splitter.split(df_train, y_train))
		df_val = df_train.iloc[val_idx].reset_index(drop=True)
		y_val = y_train[val_idx]
		df_train = df_train.iloc[train_idx].reset_index(drop=True)
		y_train = y_train[train_idx]
	logger.info(
		'Splits prepared: train=%d, val=%d, test=%d',
		len(df_train),
		len(df_val),
		len(df_test),
	)

	pre = Preprocessor(meta['categorical_cols'], meta['numeric_cols'])
	X_train = pre.fit(df_train).transform(df_train)
	X_val = pre.transform(df_val)
	X_test = pre.transform(df_test)
	logger.info('Preprocessing complete: encoded_features=%d', pre.feature_count_)

	X_train, y_train = balance_resample(X_train, y_train, resample, seed)
	classes, counts = np.unique(y_train, return_counts=True)
	logger.info('Resampling complete: class_distribution=%s', dict(zip(classes.tolist(), counts.tolist())))

	average = 'binary' if binary else 'weighted'

	cnn_rf_result = _run_cnn_rf(
		X_train, y_train, X_val, y_val, X_test, y_test, feature_k, average, use_gridsearch, seed
	)

	baseline_results = run_all_baselines(
		X_train, y_train, X_val, y_val, X_test, y_test, average, seed, feature_k
	)

	results = [cnn_rf_result] + baseline_results

	feature_counts = {
		'raw': meta['raw_feature_count'],
		'encoded': pre.feature_count_,
		'reduced': feature_k,
	}

	_save_json(os.path.join(out_dir, 'results.json'), {'results': results, 'feature_counts': feature_counts})
	report = _render_paper_results(dataset, '80/10/10 stratified', feature_counts, results)
	_save_text(os.path.join(out_dir, 'paper_results.md'), report)

	plot_metric_bars(results, os.path.join(out_dir, 'metrics_bar.png'))
	plot_roc_curves(results, os.path.join(out_dir, 'roc_curves.png'))
	plot_time_bars(results, os.path.join(out_dir, 'time_bar.png'))
	logger.info('Single split run completed. Outputs saved to %s', out_dir)

	return {'results': results, 'feature_counts': feature_counts}


def run_crossval(
	dataset: str,
	data_dir: str,
	binary: bool,
	feature_k: int,
	seed: int,
	max_rows: int,
	resample: str,
	use_gridsearch: bool,
	use_original_split: bool,
	out_dir: str,
	cv_folds: int,
):
	logger.info(
		'Starting cross-validation run (dataset=%s, folds=%d, feature_k=%d, seed=%d)',
		dataset,
		cv_folds,
		feature_k,
		seed,
	)
	set_global_seed(seed)
	save_run_metadata(out_dir, seed)

	if dataset == 'kdd99':
		df, y, meta = load_kdd99(data_dir=data_dir, binary=binary, max_rows=max_rows, seed=seed)
	elif dataset in ('unsw', 'unsw_nb15'):
		df, y, meta = load_unsw_nb15(
			data_dir=data_dir, binary=binary, max_rows=max_rows, seed=seed, use_original_split=use_original_split
		)
	else:
		raise ValueError('Unknown dataset.')
	logger.info('Dataset loaded for CV: rows=%d, columns=%d', len(df), df.shape[1])

	skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
	average = 'binary' if binary else 'weighted'
	fold_results = []

	for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
		logger.info('Starting fold %d/%d', fold, cv_folds)
		df_train = df.iloc[train_idx].reset_index(drop=True)
		y_train = y[train_idx]
		df_val = df.iloc[val_idx].reset_index(drop=True)
		y_val = y[val_idx]

		pre = Preprocessor(meta['categorical_cols'], meta['numeric_cols'])
		X_train = pre.fit(df_train).transform(df_train)
		X_val = pre.transform(df_val)

		X_train, y_train = balance_resample(X_train, y_train, resample, seed + fold)

		result = _run_cnn_rf(
			X_train, y_train, X_val, y_val, X_val, y_val, feature_k, average, use_gridsearch, seed + fold
		)
		result['name'] = f'CNN+RF_fold{fold}'
		fold_results.append(result)
		logger.info('Finished fold %d/%d', fold, cv_folds)

	mean_result = {'name': 'CNN+RF_CV_mean'}
	for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'train_time', 'infer_time']:
		vals = [r.get(k) for r in fold_results if k in r]
		if vals:
			mean_result[k] = float(np.mean(vals))

	feature_counts = {
		'raw': meta['raw_feature_count'],
		'encoded': pre.feature_count_,
		'reduced': feature_k,
	}

	_save_json(os.path.join(out_dir, 'cv_results.json'), {'folds': fold_results, 'mean': mean_result})
	report = _render_paper_results(dataset, f'{cv_folds}-fold stratified CV (CNN+RF only)', feature_counts, [mean_result])
	_save_text(os.path.join(out_dir, 'paper_results.md'), report)

	plot_metric_bars([mean_result], os.path.join(out_dir, 'metrics_bar.png'))
	logger.info('Cross-validation run completed. Outputs saved to %s', out_dir)

	return {'results': fold_results, 'mean': mean_result, 'feature_counts': feature_counts}
