import matplotlib.pyplot as plt

from .logging_utils import get_logger

logger = get_logger(__name__)


def plot_metric_bars(results, out_path: str) -> None:
	logger.info('Generating metric bar plot: %s', out_path)
	names = [r['name'] for r in results]
	metrics = ['accuracy', 'precision', 'recall', 'f1']

	fig, ax = plt.subplots(figsize=(10, 5))
	width = 0.18
	x = list(range(len(names)))

	for i, m in enumerate(metrics):
		vals = [r.get(m, 0.0) for r in results]
		ax.bar([v + i * width for v in x], vals, width=width, label=m)

	ax.set_xticks([v + width * 1.5 for v in x])
	ax.set_xticklabels(names, rotation=30, ha='right')
	ax.set_ylim(0, 1.0)
	ax.set_ylabel('Score')
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def plot_roc_curves(results, out_path: str) -> None:
	logger.info('Generating ROC curve plot: %s', out_path)
	fig, ax = plt.subplots(figsize=(6, 6))
	plotted = False
	for r in results:
		roc = r.get('roc_curve')
		if not roc:
			continue
		ax.plot(roc['fpr'], roc['tpr'], label=r['name'])
		plotted = True

	if plotted:
		ax.plot([0, 1], [0, 1], 'k--')
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('ROC Curves')
		ax.legend()
		fig.tight_layout()
		fig.savefig(out_path)
	else:
		logger.info('Skipping ROC plot save because no ROC data was available.')
	plt.close(fig)


def plot_time_bars(results, out_path: str) -> None:
	logger.info('Generating time bar plot: %s', out_path)
	names = [r['name'] for r in results]
	times = [r.get('train_time', 0.0) + r.get('infer_time', 0.0) for r in results]

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.bar(names, times)
	ax.set_ylabel('Seconds')
	ax.set_title('Train + Inference Time')
	ax.tick_params(axis='x', labelrotation=30)
	plt.setp(ax.get_xticklabels(), ha='right')
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)
