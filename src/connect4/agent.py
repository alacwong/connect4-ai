"""
Agent class for different nn models
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from src.connect4.board import Board
from src.mcts.node import Node
from src.mcts.mcts import monte_carlo_tree_search
from src.ml.wrapper import ValueModel, PolicyModel
from src.constants import HUMAN, MCTS, MINIMAX, RANDOM
import uuid


class Agent(ABC):

    @abstractmethod
    def play(self) -> int:
        """
        select action
        :return:
        """
        pass

    @abstractmethod
    def update_state(self, action: int):
        """
        update agent's board state
        :return:
        """
        pass

    @abstractmethod
    def get_agent_type(self):
        """
        get agent's unique id
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset agent's memory to initial values
        """
        pass

    @abstractmethod
    def copy(self) -> Agent:
        """
        Copies agent
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

    def update_state(self, action):
        self.board = self.board.play_action(action)

    def __init__(self):
        self.board = Board.empty()

    def get_agent_type(self):
        """
        get agent's unique id
        """
        return 'random'

    def reset(self):
        """
        Reset agent's memory to initial values
        """
        self.board = Board.empty()

    def copy(self) -> Agent:
        """
        Copies agent
        """
        return RandomAgent()


class MCTSAgent(Agent):
    """
    Agent plays using mcts guided by policy and value network
    """

    def __init__(self, value_network: ValueModel, policy_network: PolicyModel, agent_type: str):
        self.tree = Node(board=Board.empty(), action_id=0, depth=0)
        self.root = self.tree
        self.value_model = value_network
        self.policy_model = policy_network
        self.agent_type = agent_type
        self.agent_id = str(uuid.uuid4())

    def play(self) -> int:
        """
        :return:
        """
        node = monte_carlo_tree_search(self.root, self.value_model, self.policy_model)
        node.parent = None
        self.tree = node
        self.root = node
        return node.action_id

    def update_state(self, action):
        """
        Update current board position
        :param action:
        :return:
        """

        # traverse tree
        for child in self.root.children:
            if child.action_id == action:
                self.root = child

    def get_agent_type(self):
        """
        get agent's unique id
        """
        return self.agent_type

    def reset(self):
        """
        Reset agent's memory to initial values
        """
        self.tree = Node(board=Board.empty(), action_id=0, depth=0)
        self.root = self.tree

    def copy(self) -> Agent:
        """
        Copies agent
        """
        return MCTSAgent(
            value_network=self.value_model,
            policy_network=self.policy_model,
            agent_type=self.agent_type
        )


class MiniMaxAgent(Agent):
    """
    Agent plays using minimax with alpha beta pruning
    using value network as a partial configuration function
    """

    def update_state(self, action: int):
        pass

    def play(self) -> int:
        pass

    def get_agent_type(self):
        """
        get agent's unique id
        """
        pass

    def reset(self):
        """
        Reset agent's memory to initial values
        """
        pass

    def copy(self) -> Agent:
        """
        Copies agent
        """


class HumanAgent(Agent):
    """
    Agent plays with human input
    """

    def update_state(self, action: int):
        pass

    def play(self) -> int:
        pass

    def get_agent_type(self):
        """
        get agent's unique id
        """
        pass

    def reset(self):
        """
        Reset agent's memory to initial values
        """
        pass

    def copy(self) -> Agent:
        """
        Copies agent
        """


class QAgent(Agent):
    """
    Deep q learning agent
    """

    def play(self) -> int:
        pass

    def update_state(self, action: int):
        pass

    def get_agent_type(self):
        pass

    def reset(self):
        """
        Reset agent's memory to initial values
        """
        pass

    def copy(self) -> Agent:
        """
        Copies agent
        """


class AgentFactory:

    @staticmethod
    def get_agent(agent_code, **kwargs) -> Agent:
        """
        Factory method to get agent
        """

        if agent_code == MCTS:
            return MCTSAgent(**kwargs)
        elif agent_code == HUMAN:
            pass
        elif agent_code == RANDOM:
            pass
        elif agent_code == MINIMAX:
            pass
