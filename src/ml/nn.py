from tensorflow import keras
from src.constants import col, row
import time
import os
import numpy as np
from src.constants import device
import tensorflow as tf
from src.ml.model import Model
import pickle
from src.ml.train_util import preprocess_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_cnn_policy_model():
    """
    CNN architecture for policy model
    """
    with tf.device(device):
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(col, row, 1)),
                keras.layers.Conv2D(9, (3, 3), input_shape=(col, row), activation='relu'),
                keras.layers.Conv2D(9, (3, 3), input_shape=(col, row), activation='relu'),
                keras.layers.Conv2D(9, (3, 3), input_shape=(col, row), activation='relu'),
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
                keras.layers.Conv2D(9, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(9, (3, 3), activation='relu', padding='same'),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1, activation='tanh')
            ]
        )
        model.compile(loss='mse', optimizer='adam')
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

        model.compile(loss='binary_crossentropy', optimizer='adam')
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

        model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    testing = False

    # Test model efficiency with lite-model
    if testing:
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

        print('stop')
        start = time.time()
        for test in test_case:
            lite_mode_pred.append(lite_model.predict(test.reshape(1, 42)))
        lite_model_cache_time = time.time() - start

        for i in range(len(ans)):
            print(f'Raw: {ans[i]} Lite: {lite_mode_pred[i]}')

        print(f'Raw: {old} Lite: {lite_model_time} s, {lite_model_cache_time}')

    # See how well model fits self-play data
    # with open('generated/datasets/large-data.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     priors = data['priors']
    #     states = data['states']
    #     values = data['values']
    #
    # print(states.shape)
    # start = time.time()
    # states, priors, values = preprocess_data(states, priors, values)
    # with open('generated/datasets/cleaned_data.pkl', 'wb+') as f:
    #     pickle.dump(
    #         {
    #             'priors': priors,
    #             'states': states,
    #             'values': values
    #         }, f
    #     )
    # print(f'Aggregate and process data in {time.time() - start}s')
    # print(states.shape)

    # See how well model performs on cleaned data
    with open('generated/datasets/cleaned_data.pkl', 'rb+')as f:
        data = pickle.load(f)
        priors = data['priors']
        states = data['states']
        values = data['values']

    value_network = get_cnn_value_model()
    print(value_network.summary())
    N, D = states.shape
    states = states.reshape((N, col, row, 1))
    # policy_network = get_policy_network()

    #
    value_network.fit(states, values, epochs=10, validation_split=0.3)
    # policy_network.fit(states, priors, epochs=10, validation_split=0.3)
