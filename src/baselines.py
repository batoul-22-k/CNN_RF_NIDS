import time
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .logging_utils import get_logger
from .metrics import compute_metrics
from .model_rf import DEFAULT_RF_PARAMS

logger = get_logger(__name__)


def _eval_classifier(name: str, clf, X_test, y_test, average: str):
	t0 = time.perf_counter()
	y_pred = clf.predict(X_test)
	infer_time = time.perf_counter() - t0
	y_prob = None
	if average == 'binary' and hasattr(clf, 'predict_proba'):
		y_prob = clf.predict_proba(X_test)[:, 1]
	metrics = compute_metrics(y_test, y_pred, y_prob, average=average)
	metrics['name'] = name
	metrics['infer_time'] = infer_time
	logger.info('%s complete: f1=%.4f acc=%.4f', name, metrics['f1'], metrics['accuracy'])
	return metrics


def run_rf_baseline(X_train, y_train, X_test, y_test, average, seed):
	t0 = time.perf_counter()
	clf = RandomForestClassifier(random_state=seed, **DEFAULT_RF_PARAMS)
	clf.fit(X_train, y_train)
	train_time = time.perf_counter() - t0
	res = _eval_classifier('RF (all features)', clf, X_test, y_test, average)
	res['train_time'] = train_time
	return res


def run_pca_rf_baseline(X_train, y_train, X_test, y_test, average, seed, feature_k):
	t0 = time.perf_counter()
	pca = PCA(n_components=feature_k, random_state=seed)
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca = pca.transform(X_test)
	clf = RandomForestClassifier(random_state=seed, **DEFAULT_RF_PARAMS)
	clf.fit(X_train_pca, y_train)
	train_time = time.perf_counter() - t0
	res = _eval_classifier('PCA+RF', clf, X_test_pca, y_test, average)
	res['train_time'] = train_time
	return res


def _genetic_select(
	X_train,
	y_train,
	X_val,
	y_val,
	k: int,
	seed: int,
	average: str,
	population: int = 12,
	generations: int = 6,
	mutation_rate: float = 0.1,
):
	rng = np.random.default_rng(seed)
	n_features = X_train.shape[1]

	def random_mask():
		idx = rng.choice(n_features, size=k, replace=False)
		mask = np.zeros(n_features, dtype=bool)
		mask[idx] = True
		return mask

	def fitness(mask):
		clf = RandomForestClassifier(random_state=seed, **DEFAULT_RF_PARAMS)
		clf.fit(X_train[:, mask], y_train)
		y_pred = clf.predict(X_val[:, mask])
		return compute_metrics(y_val, y_pred, average=average)['f1']

	population_masks = [random_mask() for _ in range(population)]
	scores = [fitness(m) for m in population_masks]

	for _ in range(generations):
		ranked = sorted(zip(population_masks, scores), key=lambda x: x[1], reverse=True)
		survivors = [m for m, _ in ranked[: population // 2]]
		new_pop = survivors.copy()

		while len(new_pop) < population:
			parent1 = survivors[rng.integers(0, len(survivors))]
			parent2 = survivors[rng.integers(0, len(survivors))]
			cut = rng.integers(1, n_features - 1)
			child = np.concatenate([parent1[:cut], parent2[cut:]]).astype(bool)

			if child.sum() > k:
				off = rng.choice(np.where(child)[0], size=child.sum() - k, replace=False)
				child[off] = False
			elif child.sum() < k:
				on = rng.choice(np.where(~child)[0], size=k - child.sum(), replace=False)
				child[on] = True

			if rng.random() < mutation_rate:
				on = rng.choice(np.where(child)[0])
				off = rng.choice(np.where(~child)[0])
				child[on] = False
				child[off] = True

			new_pop.append(child)

		population_masks = new_pop
		scores = [fitness(m) for m in population_masks]

	best_mask = population_masks[int(np.argmax(scores))]
	return best_mask


def run_ga_rf_baseline(X_train, y_train, X_val, y_val, X_test, y_test, average, seed, feature_k):
	t0 = time.perf_counter()
	mask = _genetic_select(X_train, y_train, X_val, y_val, feature_k, seed, average)
	clf = RandomForestClassifier(random_state=seed, **DEFAULT_RF_PARAMS)
	clf.fit(X_train[:, mask], y_train)
	train_time = time.perf_counter() - t0
	res = _eval_classifier('GA+RF', clf, X_test[:, mask], y_test, average)
	res['train_time'] = train_time
	return res


def run_c45_baseline(X_train, y_train, X_test, y_test, average, seed):
	t0 = time.perf_counter()
	clf = DecisionTreeClassifier(criterion='entropy', random_state=seed)
	clf.fit(X_train, y_train)
	train_time = time.perf_counter() - t0
	res = _eval_classifier('C4.5-like', clf, X_test, y_test, average)
	res['train_time'] = train_time
	return res


def run_nbtree_proxy_baseline(X_train, y_train, X_test, y_test, average, seed):
	t0 = time.perf_counter()
	tree = DecisionTreeClassifier(max_depth=8, random_state=seed)
	tree.fit(X_train, y_train)

	leaf_train = tree.apply(X_train).reshape(-1, 1)
	leaf_test = tree.apply(X_test).reshape(-1, 1)

	try:
		# scikit-learn >= 1.2
		ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
	except TypeError:
		# scikit-learn < 1.2
		ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
	leaf_train_enc = ohe.fit_transform(leaf_train)
	leaf_test_enc = ohe.transform(leaf_test)

	X_train_nb = np.hstack([X_train, leaf_train_enc])
	X_test_nb = np.hstack([X_test, leaf_test_enc])

	nb = GaussianNB()
	nb.fit(X_train_nb, y_train)
	train_time = time.perf_counter() - t0

	res = _eval_classifier('NBTree-proxy', nb, X_test_nb, y_test, average)
	res['train_time'] = train_time
	return res


def run_svm_baseline(X_train, y_train, X_test, y_test, average, seed):
	t0 = time.perf_counter()
	scaler = StandardScaler()
	X_train_s = scaler.fit_transform(X_train)
	X_test_s = scaler.transform(X_test)

	clf = SVC(kernel='rbf', probability=True, random_state=seed, gamma='scale')
	clf.fit(X_train_s, y_train)
	train_time = time.perf_counter() - t0

	res = _eval_classifier('SVM (RBF)', clf, X_test_s, y_test, average)
	res['train_time'] = train_time
	return res


def run_all_baselines(
	X_train,
	y_train,
	X_val,
	y_val,
	X_test,
	y_test,
	average: str,
	seed: int,
	feature_k: int,
) -> List[Dict]:
	logger.info('Running baselines...')
	results = []
	results.append(run_rf_baseline(X_train, y_train, X_test, y_test, average, seed))
	results.append(run_pca_rf_baseline(X_train, y_train, X_test, y_test, average, seed, feature_k))
	results.append(run_ga_rf_baseline(X_train, y_train, X_val, y_val, X_test, y_test, average, seed, feature_k))
	results.append(run_c45_baseline(X_train, y_train, X_test, y_test, average, seed))
	results.append(run_nbtree_proxy_baseline(X_train, y_train, X_test, y_test, average, seed))
	results.append(run_svm_baseline(X_train, y_train, X_test, y_test, average, seed))
	logger.info('Finished baselines (%d models).', len(results))
	return results
