"""
Agent class for different nn models
"""

from abc import ABC
import numpy as np
from util import get_stack
from constants import row, col
from node import Node
from mcts import monte_carlo_tree_search


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

    def update_board(self, action):
        count = np.count_nonzero(action)
        self.board[action][col - count] = -1 * self.player

    def __init__(self, board, player):
        self.board = board
        self.player = player


class MCTSAgent(Agent):
    """
    Agent plays using mcts guided by policy and value network
    """

    def __init__(self, board, player):
        self.tree = Node(board=board, action_id=0)
        self.root = self.tree
        self.board = board
        self.player = player

    def play(self) -> int:
        """
        :return:
        """
        node = monte_carlo_tree_search(self.root)
        self.root = node
        return node.action_id

    def update_board(self, action):
        """
        Update current board position
        :param action:
        :return:
        """

        # traverse tree
        for child in self.root:
            if child.action_id == action:
                self.root = child


class MiniMaxAgent(Agent):
    """
    Agent plays using minimax with alpha beta pruning
    using value network as a partial configuration function
    """
