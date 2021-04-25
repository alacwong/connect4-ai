from tensorflow import keras
from src.constants import col, row
import time
import os
import numpy as np
from src.constants import device
import tensorflow as tf
from src.ml.model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_cnn_policy_model():
    """
    CNN architecture for policy model
    """
    with tf.device(device):
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(col, row, 1)),
                keras.layers.Conv2D(32, (3, 3), input_shape=(col, row), activation='relu'),
                keras.layers.Conv2D(32, (3, 3), input_shape=(col, row), activation='relu'),
                keras.layers.Conv2D(32, (3, 3), input_shape=(col, row), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(col * row, activation='relu'),
                keras.layers.Dense(col, activation='softmax')
            ]
        )
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_cnn_value_model():
    """
    CNN architecture for value model
    """
    with tf.device(device):
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(col, row, 1)),
                keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                keras.layers.Flatten(),
                keras.layers.Dense(col * row, activation='relu'),
                keras.layers.Dense(1, activation='tanh')
            ]
        )
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def get_policy_network(from_file=False):
    """
    policy architecture
    :return:
    """

    path = 'generated/initial/policy'

    if from_file:
        return tf.keras.models.load_model(path)

    with tf.device(device):
        model = keras.Sequential(
            [
                keras.Input(shape=(col * row), dtype='int64'),
                keras.layers.Dense((col * row), activation='relu'),
                keras.layers.Dense((col * row), activation='relu'),
                keras.layers.Dense(row, activation='softmax')
            ]
        )

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.save(path)
    return model


def get_value_network(from_file=False):
    """
    value network architecture
    :return:
    """

    path = 'generated/initial/value'

    if from_file:
        return tf.keras.models.load_model(path)

    with tf.device(device):
        model = keras.Sequential(
            [
                keras.Input(shape=(col * row), dtype='int64'),
                keras.layers.Dense(col * row, activation='relu'),
                keras.layers.Dense(col * row, activation='relu'),
                keras.layers.Dense(1, activation='tanh')
            ]
        )

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    np.random.seed(0)
    test_case = [
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
        np.random.randint(2, size=42),
    ]

    raw_model = get_value_network()
    lite_model = Model.from_keras(raw_model)
    ans = []
    start = time.time()
    for test in test_case:
        ans.append(raw_model(test.reshape(1, 42)))
    old = time.time() - start

    lite_mode_pred = []
    start = time.time()
    for test in test_case:
        lite_mode_pred.append(lite_model.predict(test.reshape(1, 42)))
    lite_model_time = time.time() - start

    for i in range(len(ans)):
        print(f'Raw: {ans[i]} Lite: {lite_mode_pred[i]}')

    print(f'Raw: {old} Lite: {lite_model_time} s')
