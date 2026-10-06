import io
import contextlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, regularizers

from source.neural_network.metrics.RMSE import RMSE
from source.neural_network.architecture_logger import ArchitectureLogger


class NeuralNetwork:
    def __init__(self):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(42,)),
            layers.Dense(1, activation="sigmoid")
        ])

        self.logger = ArchitectureLogger()
        self._compile_model()

    def _compile_model(self):
        self.model.compile(
            tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="mse",
            metrics=[RMSE()]
        )

    def fit(self, X, y):
        """Функция для обучения (X - входные данные, y - выходные)"""
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')

        return self.model.fit(
            X, y,
            epochs=30,
            batch_size=32,
            verbose=1
        )
    
    def validate(self, X, y):
        """
        Функция для валидации модели.
        Возвращает словарь с RMSE и MSE (loss).
        """
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')

        results = self.model.evaluate(X, y, verbose=0, return_dict=True)

        report = {
            'loss_mse': float(results.get('loss', float('nan'))),
            'rmse': float(results.get('RMSE', results.get('rmse', float('nan')))),
        }
        self.logger.update(architecture_str=str(self), rmse=report['rmse'])
        return report

    def predict(self, value):
        """Функция для предсказания модели"""
        arr = np.asarray(value, dtype='float32').reshape(1, -1)
        preds = self.model.predict(arr, verbose=0)
        return preds[0].tolist()

    def __str__(self):
        """Функция для вывода информации об архитектуре нейросети"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model.summary()
        return buf.getvalue()
