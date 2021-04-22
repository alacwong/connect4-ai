"""
Different neural network models
"""

from abc import ABC
import numpy as np
from src.constants import row, col
from src.ml.nn import get_value_network, get_policy_network
from abc import abstractmethod


# Model interfaces

class ValueModel(ABC):

    @abstractmethod
    def compute_value(self, state: np.ndarray) -> float:
        """evaluate board state"""
        pass


class PolicyModel(ABC):

    @abstractmethod
    def compute_policy(self, state: np.ndarray, valid_actions) -> np.array:
        """Compute policy distribution of actions from state """
        pass


# Model Implementations

class MockValueModel(ValueModel):

    def compute_value(self, state) -> float:
        return 0


class MockPolicyModel(PolicyModel):

    def compute_policy(self, state, valid_actions) -> np.array:
        """
        Assume uniform
        :param valid_actions:
        :param state:
        :return:
        """

        dist = np.array([1 / row for _ in range(row)])
        for i in range(row):
            if i not in valid_actions:
                dist[i] = 0

        return dist / np.sum(dist)


class AlphaValueModel(ValueModel):

    def __init__(self, network=None):
        # initial network, load from python,
        # otherwise load from serialized file
        if not network:
            self.network = get_value_network()

    def compute_value(self, state) -> float:
        return self.network(state.reshape((1, col * row)), training=False).numpy()[0][0]


class AlphaPolicyModel(PolicyModel):

    def __init__(self, network=None):
        # initial network, load from python,
        # otherwise load from serialized file
        if not network:
            self.network = get_policy_network()

    def compute_policy(self, state: np.ndarray, valid_actions) -> np.array:
        dist = self.network(state.reshape((1, col * row))).numpy()[0]
        for i in range(row):
            if i not in valid_actions:
                dist[i] = 0
        return dist / np.sum(dist)
