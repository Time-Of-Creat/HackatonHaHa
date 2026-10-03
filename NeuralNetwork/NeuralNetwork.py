import io
import contextlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

from .Metrics.RMSE import RMSE


class NeuralNetwork:
	def __init__(self):
		self.model = tf.keras.Sequential([
            # Вход
            layers.Input(shape=(1,)),
            
            # Внутренняя архитектура нейронки
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Выход
            layers.Dense(10)
        ])
        
        self._compile_model()
    
    def _compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[RMSE()]
        )
    
    def fit(self, X, y):
        """Функция для обучения (X - входные данные, y - выходные)"""
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
        
        return self.model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            verbose=1
        )
    
    def predict(self, value):
        """Функция для предсказания модели"""
        arr = np.asarray(value, dtype='float32')
        preds = self.model.predict(arr, verbose=0)
        return preds[0].tolist()
    
    def __str__(self):
        """Функция для вывода информации об архитектуре нейросети"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model.summary()
        return buf.getvalue()
