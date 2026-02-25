import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.logging_utils import configure_logging, get_logger
from src.pipeline import make_run_dir, run_crossval, run_single_split


def str2bool(x):
	return str(x).lower() in ('true', '1', 'yes', 'y')


def main():
	configure_logging()
	logger = get_logger(__name__)
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='kdd99', choices=['kdd99'])
	parser.add_argument('--data_dir', default='')
	parser.add_argument('--binary', type=str2bool, default=True)
	parser.add_argument('--cv_folds', type=int, default=0)
	parser.add_argument('--use_gridsearch', type=str2bool, default=False)
	parser.add_argument('--feature_k', type=int, default=18)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--max_rows', type=int, default=None)
	parser.add_argument('--resample', choices=['none', 'over', 'under'], default='none')
	args = parser.parse_args()
	logger.info('Starting run_kdd99 with args: %s', vars(args))

	run_dir = make_run_dir()
	log_file = os.path.join(run_dir, 'run.log')
	configure_logging(log_file=log_file)
	logger.info('Writing logs to %s', log_file)
	if args.cv_folds and args.cv_folds > 1:
		run_crossval(
			dataset='kdd99',
			data_dir=args.data_dir,
			binary=args.binary,
			feature_k=args.feature_k,
			seed=args.seed,
			max_rows=args.max_rows,
			resample=args.resample,
			use_gridsearch=args.use_gridsearch,
			use_original_split=False,
			out_dir=run_dir,
			cv_folds=args.cv_folds,
		)
	else:
		run_single_split(
			dataset='kdd99',
			data_dir=args.data_dir,
			binary=args.binary,
			feature_k=args.feature_k,
			seed=args.seed,
			max_rows=args.max_rows,
			resample=args.resample,
			use_gridsearch=args.use_gridsearch,
			use_original_split=False,
			out_dir=run_dir,
		)
	logger.info('run_kdd99 completed. Outputs: %s', run_dir)


if __name__ == '__main__':
	main()
