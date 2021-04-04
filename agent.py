"""
Agent class for different nn models
"""

from abc import ABC
import numpy as np
from util import get_stack
from constants import row, col


class Agent(ABC):

    def play(self) -> int:
        """
        select action
        :return:
        """
        pass

    def update_board(self, board):
        """
        update agent's board state
        :return:
        """
        pass


class RandomAgent(Agent):
    """
    agent plays randomly
    """

    def play(self) -> int:
        """
        select action randomly
        :return:
        """

        stack = get_stack(self.board)

        actions = []
        for i in range(row):
            if stack[i] < col:
                actions.append(i)

        return np.random.choice(np.array(actions))

    def update_board(self, board):
        self.board = board

    def __init__(self, board=None):
        self.board = board


class MCTSAgent(Agent):
    """
    Agent plays using mcts guided by policy and value network
    """


class MiniMaxAgent(Agent):
    """
    Agent plays using minimax with alpha beta pruning
    using value network as a partial configuration function
    """
