from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)

from .logging_utils import get_logger

logger = get_logger(__name__)


def compute_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_prob: Optional[np.ndarray] = None,
	average: str = 'binary',
) -> Dict:
	metrics = {
		'accuracy': float(accuracy_score(y_true, y_pred)),
		'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
		'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
		'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
		'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
	}
	if y_prob is not None and average == 'binary':
		auc = roc_auc_score(y_true, y_prob)
		fpr, tpr, _ = roc_curve(y_true, y_prob)
		metrics['roc_auc'] = float(auc)
		metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
	logger.debug(
		'Computed metrics (average=%s): acc=%.4f f1=%.4f',
		average,
		metrics['accuracy'],
		metrics['f1'],
	)
	return metrics
