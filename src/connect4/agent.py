"""
Agent class for different nn models
"""

from abc import ABC, abstractmethod
import numpy as np
from connect4.util import Board
from mcts.node import Node
from mcts.mcts import monte_carlo_tree_search
from ml.model import ValueModel, PolicyModel


class Agent(ABC):

    @abstractmethod
    def play(self) -> int:
        """
        select action
        :return:
        """
        pass

    @abstractmethod
    def update_board(self, action: int):
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

        actions = self.board.get_valid_actions()
        action = np.random.choice(np.array(list(actions)))
        self.board = self.board.play_action(action)
        return action

    def update_board(self, action):
        self.board = self.board.play_action(action)

    def __init__(self, board: Board, player):
        self.board = board
        self.player = player


class MCTSAgent(Agent):
    """
    Agent plays using mcts guided by policy and value network
    """

    def __init__(self, board: Board, player: int, value_network: ValueModel, policy_network: PolicyModel):
        self.tree = Node(board=board, action_id=0, depth=0)
        self.root = self.tree
        self.board = board
        self.player = player
        self.value_model = value_network
        self.policy_model = policy_network

    def play(self) -> int:
        """
        :return:
        """
        node = monte_carlo_tree_search(self.root, self.value_model, self.policy_model)
        self.root = node
        return node.action_id

    def update_board(self, action):
        """
        Update current board position
        :param action:
        :return:
        """

        # traverse tree
        for child in self.root.children:
            if child.action_id == action:
                self.root = child


class MiniMaxAgent(Agent):
    """
    Agent plays using minimax with alpha beta pruning
    using value network as a partial configuration function
    """

    def update_board(self, action: int):
        pass

    def play(self) -> int:
        pass


class HumanAgent(Agent):
    """
    Agent plays with human input
    """

    def update_board(self, action: int):
        pass

    def play(self) -> int:
        pass
