from tensorflow import keras
from src.constants import col, row
import time
import os
import numpy as np
from src.constants import device
import tensorflow as tf


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


def get_policy_network():
    """
    policy architecture
    :return:
    """
    
    with tf.device(device):
        model = keras.Sequential(
            [
                keras.Input((col * row)),
                keras.layers.Dense((col * row), activation='relu'),
                keras.layers.Dense((col * row), activation='relu'),
                keras.layers.Dense(row, activation='softmax')
            ]
        )

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_value_network():
    """
    value network architecture
    :return:
    """
    with tf.device(device):
        model = keras.Sequential(
            [
                keras.Input(shape=(col * row)),
                keras.layers.Dense(col * row, activation='relu'),
                keras.layers.Dense(col * row, activation='relu'),
                keras.layers.Dense(1, activation='tanh')
            ]
        )

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    my_model = get_value_network()
    start = time.time()
    board = np.zeros((1, 42))
    pred = my_model(board, training=False)
    print(pred.numpy()[0][0])
    print(f'{time.time() - start} s')
