"""
Train neurual network
"""

from tensorflow import keras
from constants import col, row


def get_policy_network():
    """
    policy architecture
    :return:
    """

    model = keras.Sequential(
        [
            keras.Input((col * row)),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((row,), activation='softmax')
        ]
    )

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_value_network():
    """
    value network architecture
    :return:
    """

    model = keras.Sequential(
        [
            keras.Input((col * row)),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((1,), activation='sigmoid')
        ]
    )

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
