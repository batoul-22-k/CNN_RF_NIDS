import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.logging_utils import configure_logging, get_logger
from src.pipeline import make_run_dir, run_single_split


def str2bool(x):
	return str(x).lower() in ('true', '1', 'yes', 'y')


def main():
	configure_logging()
	logger = get_logger(__name__)
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', required=True)
	parser.add_argument('--binary', type=str2bool, default=True)
	parser.add_argument('--use_gridsearch', type=str2bool, default=False)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--max_rows', type=int, default=None)
	parser.add_argument('--resample', choices=['none', 'over', 'under'], default='none')
	parser.add_argument('--use_original_split', type=str2bool, default=False)
	args = parser.parse_args()
	logger.info('Starting run_all with args: %s', vars(args))

	base_dir = make_run_dir()
	log_file = os.path.join(base_dir, 'run.log')
	configure_logging(log_file=log_file)
	logger.info('Writing logs to %s', log_file)
	kdd_dir = os.path.join(base_dir, 'kdd99')
	unsw_dir = os.path.join(base_dir, 'unsw_nb15')
	os.makedirs(kdd_dir, exist_ok=True)
	os.makedirs(unsw_dir, exist_ok=True)

	run_single_split(
		dataset='kdd99',
		data_dir=args.data_dir,
		binary=args.binary,
		feature_k=18,
		seed=args.seed,
		max_rows=args.max_rows,
		resample=args.resample,
		use_gridsearch=args.use_gridsearch,
		use_original_split=False,
		out_dir=kdd_dir,
	)

	run_single_split(
		dataset='unsw',
		data_dir=args.data_dir,
		binary=args.binary,
		feature_k=22,
		seed=args.seed,
		max_rows=args.max_rows,
		resample=args.resample,
		use_gridsearch=args.use_gridsearch,
		use_original_split=args.use_original_split,
		out_dir=unsw_dir,
	)
	logger.info('run_all completed. Outputs: %s', base_dir)


if __name__ == '__main__':
	main()
