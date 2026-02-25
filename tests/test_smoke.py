import numpy as np
import pandas as pd

from src.data import Preprocessor
from src.model_cnn import build_cnn_model, build_embedding_model, extract_embeddings, reshape_for_cnn, train_cnn_model
from src.model_rf import train_rf_model
from src.metrics import compute_metrics


def test_preprocessor_smoke():
	df = pd.DataFrame(
		{
			'num1': [1, 2, 3, 4],
			'num2': [0.1, 0.2, 0.3, 0.4],
			'cat': ['a', 'b', 'a', 'c'],
		}
	)
	pre = Preprocessor(categorical_cols=['cat'], numeric_cols=['num1', 'num2'])
	X = pre.fit(df).transform(df)
	assert X.shape[0] == 4
	assert X.shape[1] >= 3


def test_cnn_rf_smoke():
	rng = np.random.default_rng(0)
	X = rng.normal(size=(120, 16)).astype(np.float32)
	y = rng.integers(0, 2, size=120)

	X_train, X_val, X_test = X[:80], X[80:100], X[100:]
	y_train, y_val, y_test = y[:80], y[80:100], y[100:]

	model = build_cnn_model(n_features=16, num_classes=2, embedding_dim=8, lr=1e-3)
	model, _, _ = train_cnn_model(
		model,
		reshape_for_cnn(X_train),
		y_train,
		reshape_for_cnn(X_val),
		y_val,
		epochs=1,
		batch_size=16,
		patience=1,
	)

	emb = build_embedding_model(model)
	X_train_emb = extract_embeddings(emb, reshape_for_cnn(X_train))
	X_test_emb = extract_embeddings(emb, reshape_for_cnn(X_test))

	rf, _ = train_rf_model(X_train_emb, y_train, params={}, use_gridsearch=False, seed=0)
	y_pred = rf.predict(X_test_emb)
	metrics = compute_metrics(y_test, y_pred, average='binary')
	assert 'accuracy' in metrics
