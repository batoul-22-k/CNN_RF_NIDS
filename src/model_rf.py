from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from .logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_RF_PARAMS = {
	'n_estimators': 50,
	'max_depth': 20,
	'max_leaf_nodes': 1500,
	'min_samples_split': 2,
	'min_samples_leaf': 1,
	'criterion': 'gini',
}


def train_rf_model(
	X_train: np.ndarray,
	y_train: np.ndarray,
	params: Dict,
	use_gridsearch: bool,
	seed: int,
) -> Tuple[RandomForestClassifier, Dict]:
	if not use_gridsearch:
		logger.info('Training RF with fixed params: %s', params)
		clf = RandomForestClassifier(random_state=seed, **params)
		clf.fit(X_train, y_train)
		return clf, params

	logger.info('Training RF with GridSearchCV...')
	grid = {
		'n_estimators': [50, 100],
		'max_depth': [10, 20],
		'max_leaf_nodes': [500, 1500],
	}
	base = RandomForestClassifier(
		random_state=seed,
		min_samples_split=params.get('min_samples_split', 2),
		min_samples_leaf=params.get('min_samples_leaf', 1),
		criterion=params.get('criterion', 'gini'),
	)
	gs = GridSearchCV(base, grid, cv=3, n_jobs=-1)
	gs.fit(X_train, y_train)
	best = gs.best_estimator_
	best_params = gs.best_params_
	logger.info('GridSearch best params: %s', best_params)
	return best, best_params
