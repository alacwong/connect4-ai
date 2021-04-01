"""
Different neural network model
"""

from abc import ABC
import numpy as np


class ValueModel(ABC):

    def compute_value(self, state) -> float:
        """evaluate board state"""
        pass


class PolicyModel(ABC):

    def compute_policy(self, state) -> np.array:
        """Compute policy distrbution of actions from state """
        pass
