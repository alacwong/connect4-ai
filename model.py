"""
Different neural network models
"""

from abc import ABC
import numpy as np
from constants import row, col
from train import get_value_network, get_policy_network


# Model interfaces

class ValueModel(ABC):

    def compute_value(self, state: np.ndarray) -> float:
        """evaluate board state"""
        pass


class PolicyModel(ABC):

    def compute_policy(self, state: np.ndarray) -> np.array:
        """Compute policy distribution of actions from state """
        pass


# Model Implementations

class MockValueModel(ValueModel):

    def compute_value(self, state) -> float:
        return 0


class MockPolicyModel(PolicyModel):

    def compute_policy(self, state) -> np.array:
        """
        Assume uniform
        :param state:
        :return:
        """

        return np.array([1 / row for _ in range(row)])


class AlphaValueModel(ValueModel):

    def __init__(self, network=None):
        # initial network, load from python,
        # otherwise load from serialized file
        if not network:
            self.network = get_value_network()

    def compute_value(self, state) -> float:
        return self.network.predict(state.reshape((1, col * row)))[0]


class AlphaPolicyModel(PolicyModel):

    def __init__(self, network=None):
        # initial network, load from python,
        # otherwise load from serialized file
        if not network:
            self.network = get_policy_network()

    def compute_value(self, state) -> np.ndarray:
        return self.network.predict(state.reshape((1, col * row)))[0]
