import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from .logging_utils import get_logger

logger = get_logger(__name__)

KDD_FEATURE_NAMES = [
	'duration',
	'protocol_type',
	'service',
	'flag',
	'src_bytes',
	'dst_bytes',
	'land',
	'wrong_fragment',
	'urgent',
	'hot',
	'num_failed_logins',
	'logged_in',
	'num_compromised',
	'root_shell',
	'su_attempted',
	'num_root',
	'num_file_creations',
	'num_shells',
	'num_access_files',
	'num_outbound_cmds',
	'is_host_login',
	'is_guest_login',
	'count',
	'srv_count',
	'serror_rate',
	'srv_serror_rate',
	'rerror_rate',
	'srv_rerror_rate',
	'same_srv_rate',
	'diff_srv_rate',
	'srv_diff_host_rate',
	'dst_host_count',
	'dst_host_srv_count',
	'dst_host_same_srv_rate',
	'dst_host_diff_srv_rate',
	'dst_host_same_src_port_rate',
	'dst_host_srv_diff_host_rate',
	'dst_host_serror_rate',
	'dst_host_srv_serror_rate',
	'dst_host_rerror_rate',
	'dst_host_srv_rerror_rate',
]

KDD_CATEGORICAL = ['protocol_type', 'service', 'flag']

KDD_ATTACK_CATEGORIES = {
	'dos': {
		'back',
		'land',
		'neptune',
		'pod',
		'smurf',
		'teardrop',
		'apache2',
		'udpstorm',
		'processtable',
		'mailbomb',
	},
	'probe': {
		'satan',
		'ipsweep',
		'nmap',
		'portsweep',
		'mscan',
		'saint',
	},
	'r2l': {
		'guess_passwd',
		'ftp_write',
		'imap',
		'phf',
		'multihop',
		'warezmaster',
		'warezclient',
		'spy',
		'xlock',
		'xsnoop',
		'snmpguess',
		'snmpgetattack',
		'httptunnel',
		'sendmail',
		'named',
	},
	'u2r': {
		'buffer_overflow',
		'loadmodule',
		'perl',
		'rootkit',
		'ps',
		'sqlattack',
		'xterm',
	},
}


def _decode_if_bytes(x):
	if isinstance(x, (bytes, bytearray)):
		return x.decode('utf-8')
	return x


def _clean_labels(labels: pd.Series) -> pd.Series:
	labels = labels.apply(_decode_if_bytes).astype(str)
	labels = labels.str.strip().str.replace('.', '', regex=False).str.lower()
	return labels


def _map_kdd99_labels(labels: pd.Series, binary: bool) -> Tuple[np.ndarray, Dict[str, int]]:
	clean = _clean_labels(labels)
	if binary:
		y = (clean != 'normal').astype(int).to_numpy()
		return y, {'normal': 0, 'attack': 1}
	categories = []
	for lbl in clean:
		cat = 'other'
		if lbl == 'normal':
			cat = 'normal'
		else:
			for cname, members in KDD_ATTACK_CATEGORIES.items():
				if lbl in members:
					cat = cname
					break
		categories.append(cat)
	le = LabelEncoder()
	y = le.fit_transform(categories)
	mapping = {name: idx for idx, name in enumerate(le.classes_)}
	return y, mapping


def _find_kdd_local_file(data_dir: str) -> str:
	if not data_dir:
		return ''
	candidates = [
		'kddcup.data',
		'kddcup.data_10_percent',
		'kdd99.csv',
		'kddcup99.csv',
	]
	for name in candidates:
		path = os.path.join(data_dir, name)
		if os.path.exists(path):
			return path
	return ''


def _find_unsw_split_files(data_dir: str) -> Tuple[str, str]:
	train_name = 'UNSW_NB15_training-set.csv'
	test_name = 'UNSW_NB15_testing-set.csv'
	if not data_dir:
		return '', ''

	candidate_dirs = [
		data_dir,
		os.path.join(data_dir, 'unsw-nb15'),
		os.path.join(data_dir, 'UNSW-NB15'),
		os.path.join(data_dir, 'unsw_nb15'),
	]
	for base in candidate_dirs:
		train_path = os.path.join(base, train_name)
		test_path = os.path.join(base, test_name)
		if os.path.exists(train_path) and os.path.exists(test_path):
			return train_path, test_path

	found = {}
	for root, _, files in os.walk(data_dir):
		for fname in files:
			lower = fname.lower()
			if lower == train_name.lower() and 'train' not in found:
				found['train'] = os.path.join(root, fname)
			elif lower == test_name.lower() and 'test' not in found:
				found['test'] = os.path.join(root, fname)
		if 'train' in found and 'test' in found:
			return found['train'], found['test']

	return '', ''


def stratified_sample(
	df: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
	if max_rows is None or max_rows >= len(df):
		return df, y
	logger.info('Applying stratified sample: %d -> %d rows', len(df), max_rows)
	splitter = StratifiedShuffleSplit(
		n_splits=1, train_size=max_rows, random_state=seed
	)
	idx, _ = next(splitter.split(df, y))
	sampled_df = df.iloc[idx].reset_index(drop=True)
	sampled_y = y[idx]
	logger.info('Stratified sample complete: %d rows', len(sampled_df))
	return sampled_df, sampled_y


def stratified_split(
	df: pd.DataFrame,
	y: np.ndarray,
	seed: int,
	val_size: float = 0.1,
	test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
	splitter = StratifiedShuffleSplit(
		n_splits=1, test_size=val_size + test_size, random_state=seed
	)
	train_idx, temp_idx = next(splitter.split(df, y))
	df_train = df.iloc[train_idx].reset_index(drop=True)
	y_train = y[train_idx]

	df_temp = df.iloc[temp_idx].reset_index(drop=True)
	y_temp = y[temp_idx]

	splitter2 = StratifiedShuffleSplit(
		n_splits=1,
		test_size=test_size / (val_size + test_size),
		random_state=seed,
	)
	val_idx, test_idx = next(splitter2.split(df_temp, y_temp))
	df_val = df_temp.iloc[val_idx].reset_index(drop=True)
	y_val = y_temp[val_idx]
	df_test = df_temp.iloc[test_idx].reset_index(drop=True)
	y_test = y_temp[test_idx]
	return df_train, df_val, df_test, y_train, y_val, y_test


class Preprocessor:
	def __init__(self, categorical_cols: List[str], numeric_cols: List[str]):
		self.categorical_cols = categorical_cols
		self.numeric_cols = numeric_cols
		try:
			# scikit-learn >= 1.2
			self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
		except TypeError:
			# scikit-learn < 1.2
			self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
		self.scaler = MinMaxScaler()
		self.feature_count_ = 0

	def fit(self, df: pd.DataFrame) -> 'Preprocessor':
		X_num = df[self.numeric_cols].astype(float).to_numpy() if self.numeric_cols else None
		if self.categorical_cols:
			X_cat = self.ohe.fit_transform(df[self.categorical_cols])
			if X_num is None:
				X_all = X_cat
			else:
				X_all = np.hstack([X_num, X_cat])
		else:
			X_all = X_num
		self.scaler.fit(X_all)
		self.feature_count_ = X_all.shape[1]
		return self

	def transform(self, df: pd.DataFrame) -> np.ndarray:
		X_num = df[self.numeric_cols].astype(float).to_numpy() if self.numeric_cols else None
		if self.categorical_cols:
			X_cat = self.ohe.transform(df[self.categorical_cols])
			if X_num is None:
				X_all = X_cat
			else:
				X_all = np.hstack([X_num, X_cat])
		else:
			X_all = X_num
		return self.scaler.transform(X_all)


def balance_resample(X: np.ndarray, y: np.ndarray, method: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	if method == 'none':
		return X, y
	rng = np.random.default_rng(seed)
	classes, counts = np.unique(y, return_counts=True)
	target = counts.max() if method == 'over' else counts.min()
	indices = []
	for cls, cnt in zip(classes, counts):
		cls_idx = np.where(y == cls)[0]
		if method == 'over':
			if cnt < target:
				extra = rng.choice(cls_idx, size=target - cnt, replace=True)
				indices.append(extra)
			indices.append(cls_idx)
		else:
			if cnt > target:
				keep = rng.choice(cls_idx, size=target, replace=False)
				indices.append(keep)
			else:
				indices.append(cls_idx)
	new_idx = np.concatenate(indices)
	rng.shuffle(new_idx)
	return X[new_idx], y[new_idx]


def load_kdd99(
	data_dir: str = '',
	binary: bool = True,
	max_rows: int = None,
	seed: int = 42,
	remove_duplicates: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
	logger.info(
		'Loading KDD99 (binary=%s, max_rows=%s, remove_duplicates=%s)',
		binary,
		max_rows,
		remove_duplicates,
	)
	local_path = _find_kdd_local_file(data_dir)
	if local_path:
		logger.info('Using local KDD99 file: %s', local_path)
		df = pd.read_csv(local_path, header=None)
		if df.shape[1] == len(KDD_FEATURE_NAMES) + 1:
			df.columns = KDD_FEATURE_NAMES + ['label']
		elif df.shape[1] == len(KDD_FEATURE_NAMES) + 2:
			df.columns = KDD_FEATURE_NAMES + ['label', 'difficulty']
			df = df.drop(columns=['difficulty'])
		else:
			raise ValueError('Unexpected KDD99 column count in local file.')
		labels = df.pop('label')
	else:
		from sklearn.datasets import fetch_kddcup99

		logger.info('Local KDD99 file not found. Fetching via sklearn...')
		try:
			bunch = fetch_kddcup99(data_home=data_dir, percent10=False)
		except TypeError:
			bunch = fetch_kddcup99(data_home=data_dir)
		df = pd.DataFrame(bunch.data, columns=KDD_FEATURE_NAMES)
		labels = pd.Series(bunch.target)

	for col in df.columns:
		if df[col].dtype == object:
			df[col] = df[col].apply(_decode_if_bytes)

	labels = labels.apply(_decode_if_bytes)

	if remove_duplicates:
		before = len(df)
		df = df.drop_duplicates()
		labels = labels.loc[df.index]
		df = df.reset_index(drop=True)
		labels = labels.reset_index(drop=True)
		logger.info('Removed duplicate rows: %d', before - len(df))

	y, mapping = _map_kdd99_labels(labels, binary=binary)

	df, y = stratified_sample(df, y, max_rows, seed)

	numeric_cols = [c for c in KDD_FEATURE_NAMES if c not in KDD_CATEGORICAL]
	categorical_cols = KDD_CATEGORICAL

	meta = {
		'dataset': 'kdd99',
		'categorical_cols': categorical_cols,
		'numeric_cols': numeric_cols,
		'label_mapping': mapping,
		'raw_feature_count': len(KDD_FEATURE_NAMES),
	}
	logger.info(
		'KDD99 loaded: rows=%d, features=%d, classes=%d',
		len(df),
		df.shape[1],
		len(np.unique(y)),
	)
	return df, y, meta


def load_unsw_nb15(
	data_dir: str,
	binary: bool = True,
	max_rows: int = None,
	seed: int = 42,
	use_original_split: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
	logger.info(
		'Loading UNSW-NB15 (binary=%s, max_rows=%s, use_original_split=%s)',
		binary,
		max_rows,
		use_original_split,
	)
	train_path, test_path = _find_unsw_split_files(data_dir)
	if not train_path or not test_path:
		raise FileNotFoundError(
			f'UNSW-NB15 CSV files not found under "{data_dir}". '
			'Expected UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv.'
		)
	logger.info('Using UNSW train=%s', train_path)
	logger.info('Using UNSW test=%s', test_path)

	df_train = pd.read_csv(train_path)
	df_test = pd.read_csv(test_path)
	logger.info('Loaded UNSW files: train_rows=%d, test_rows=%d', len(df_train), len(df_test))

	if use_original_split:
		df = df_train.copy()
		df_test_final = df_test.copy()
	else:
		df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
		df_test_final = None

	for drop_col in ['id']:
		if drop_col in df.columns:
			df = df.drop(columns=[drop_col])
		if df_test_final is not None and drop_col in df_test_final.columns:
			df_test_final = df_test_final.drop(columns=[drop_col])

	if binary:
		y = df['label'].to_numpy()
		label_mapping = {'normal': 0, 'attack': 1}
		df = df.drop(columns=['label'])
	else:
		if 'attack_cat' in df.columns:
			categories = df['attack_cat'].astype(str)
			le = LabelEncoder()
			y = le.fit_transform(categories)
			label_mapping = {name: idx for idx, name in enumerate(le.classes_)}
			df = df.drop(columns=['attack_cat', 'label'])
		else:
			y = df['label'].to_numpy()
			label_mapping = {'normal': 0, 'attack': 1}
			df = df.drop(columns=['label'])

	if df_test_final is None:
		df, y = stratified_sample(df, y, max_rows, seed)
	else:
		if max_rows is not None and max_rows < len(df):
			df, y = stratified_sample(df, y, max_rows, seed)

	categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
	numeric_cols = [c for c in df.columns if c not in categorical_cols]

	meta = {
		'dataset': 'unsw_nb15',
		'categorical_cols': categorical_cols,
		'numeric_cols': numeric_cols,
		'label_mapping': label_mapping,
		'raw_feature_count': df.shape[1],
		'use_original_split': use_original_split,
		'original_test': df_test_final,
	}
	logger.info(
		'UNSW-NB15 loaded: rows=%d, features=%d, classes=%d',
		len(df),
		df.shape[1],
		len(np.unique(y)),
	)
	return df, y, meta
