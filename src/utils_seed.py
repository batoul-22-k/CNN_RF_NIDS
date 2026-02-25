import json
import os
import platform
import random
import sys

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from .logging_utils import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int) -> None:
	logger.info('Setting global seed: %d', seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
	os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	try:
		tf.config.experimental.enable_op_determinism()
	except Exception:
		pass


def get_versions() -> dict:
	return {
		'python': sys.version.replace('\n', ' '),
		'platform': platform.platform(),
		'numpy': np.__version__,
		'pandas': pd.__version__,
		'sklearn': sklearn.__version__,
		'tensorflow': tf.__version__,
	}


def save_run_metadata(run_dir: str, seed: int) -> dict:
	meta = {'seed': seed, 'versions': get_versions()}
	path = os.path.join(run_dir, 'run_metadata.json')
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(meta, f, indent=2)
	logger.info('Saved run metadata: %s', path)
	return meta
