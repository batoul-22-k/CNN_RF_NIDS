import logging
import os
from typing import Optional

_FORMAT = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
_DATE_FORMAT = '%H:%M:%S'


def configure_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
	level_name = (level or os.getenv('NIDS_LOG_LEVEL', 'INFO')).upper()
	level_value = getattr(logging, level_name, logging.INFO)
	root = logging.getLogger()
	root.setLevel(level_value)

	formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

	if not root.handlers:
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		root.addHandler(stream_handler)

	if log_file:
		log_path = os.path.abspath(log_file)
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		file_handler_exists = any(
			isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == log_path
			for h in root.handlers
		)
		if not file_handler_exists:
			file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
			file_handler.setFormatter(formatter)
			root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
	configure_logging()
	return logging.getLogger(name)
