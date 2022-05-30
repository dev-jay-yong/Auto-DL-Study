import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ExplainKeras:
    def __init__(self):
        pass

    def explain_keras(self):
        pass

    def explain_hyper_params(self):
        pass


class KerasModel:
    def __init__(self):
        pass

    def train_keras(self, data: pd.DataFrame, predict_column_name: str, hyper_params: dict):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model


if __name__ == '__main__':
    data = tf.keras.datasets.fashion_mnist
    print(data)
