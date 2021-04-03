"""
Different neural network models
"""

from abc import ABC
import numpy as np


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
