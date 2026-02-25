import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from .logging_utils import get_logger

logger = get_logger(__name__)


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
	X_reshaped = X.astype(np.float32).reshape((X.shape[0], X.shape[1], 1))
	logger.debug('Reshaped features for CNN: %s -> %s', X.shape, X_reshaped.shape)
	return X_reshaped


def build_cnn_model(
	n_features: int, num_classes: int, embedding_dim: int, lr: float = 1e-3
) -> Model:
	logger.info(
		'Building CNN model (features=%d, classes=%d, embedding_dim=%d, lr=%s)',
		n_features,
		num_classes,
		embedding_dim,
		lr,
	)
	inputs = layers.Input(shape=(n_features, 1))
	x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
	x = layers.MaxPool1D(pool_size=2)(x)
	x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
	x = layers.MaxPool1D(pool_size=2)(x)
	x = layers.Flatten()(x)
	embedding = layers.Dense(embedding_dim, activation='relu', name='embedding')(x)
	if num_classes == 2:
		outputs = layers.Dense(1, activation='sigmoid')(embedding)
		loss = 'binary_crossentropy'
	else:
		outputs = layers.Dense(num_classes, activation='softmax')(embedding)
		loss = 'sparse_categorical_crossentropy'
	model = Model(inputs, outputs)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['accuracy'])
	return model


def train_cnn_model(
	model: Model,
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	epochs: int = 100,
	batch_size: int = 128,
	patience: int = 10,
) -> Tuple[Model, dict, float]:
	logger.info(
		'Training CNN (epochs=%d, batch_size=%d, patience=%d, train_shape=%s, val_shape=%s)',
		epochs,
		batch_size,
		patience,
		X_train.shape,
		X_val.shape,
	)
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor='val_loss', patience=patience, restore_best_weights=True
		)
	]
	t0 = time.perf_counter()
	history = model.fit(
		X_train,
		y_train,
		validation_data=(X_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=callbacks,
		verbose=0,
	)
	train_time = time.perf_counter() - t0
	logger.info('CNN training complete in %.2f s', train_time)
	return model, history.history, train_time


def build_embedding_model(model: Model) -> Model:
	return Model(inputs=model.input, outputs=model.get_layer('embedding').output)


def extract_embeddings(embedding_model: Model, X: np.ndarray) -> np.ndarray:
	logger.debug('Extracting embeddings for shape=%s', X.shape)
	emb = embedding_model.predict(X, verbose=0)
	logger.debug('Embedding shape=%s', emb.shape)
	return emb
