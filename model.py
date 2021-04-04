"""
Different neural network models
"""

from abc import ABC
import numpy as np
from constants import row


# Model interfaces

class ValueModel(ABC):

    def compute_value(self, state) -> float:
        """evaluate board state"""
        pass


class PolicyModel(ABC):

    def compute_policy(self, state) -> np.array:
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
