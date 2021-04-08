from __future__ import annotations

from connect4.util import Board
from constants import simulation_constant


class Node:
    """
    Tree Node from monte carlo tree search
    """

    def __init__(self, board: Board, action_id, expected_reward=0, probability=0, parent=None, is_terminal=False):

        self.board = board
        self.action_id = action_id
        self.children = []  # states reaching by taking action a from current state
        self.visit_count = 0  # number of times visited during mcts
        self.expected_reward = expected_reward  # expected reward (initially based on value-network)
        self.simulated_reward = 0  # reward after simulating the world
        self.probability = probability  # probability of current node being played
        self.parent = parent
        self.is_terminal = is_terminal

    def ucb_score(self, exploration_factor: float):
        """
        Ucb score of node
        computed as Q(s, a) + U(s, a)
        Q(s, a) = (1 - alpha)( expected reward) + (alpha) (simulated reward) / visit_count
        U(s,a) = prior_probability / 1 + visit_count
        :return:
        """
        u = exploration_factor * self.probability / 1 + self.visit_count
        q = (1 - simulation_constant) * self.expected_reward + (
                (simulation_constant * self.simulated_reward) / (1 + self.visit_count)
        )
        return u + q

    def select_node(self, exploration_factor: float) -> Node:
        """
        returns descendant with best
        :return:
        """
        self.visit_count += 1

        if not self.children or self.is_terminal:  # terminal or leaf
            return self

        max_ucb = 0
        selected_child = None
        for child in self.children:  # traverse path of greatest ucb
            if child.ucb_score(exploration_factor) > max_ucb:
                selected_child = child
                max_ucb = child.ucb_score(exploration_factor)

        return selected_child.select_node(exploration_factor)

    def update_reward(self, reward=None):
        """
        Update reward recursively
        (early stops, if reward value remains the same, does not propagate)
        :param reward:
        :return:
        """

        if reward:
            if self.simulated_reward != reward:
                self.simulated_reward = reward
                if self.parent:
                    self.parent.update_reward()
        else:
            total_reward = 0
            for child in self.children:
                total_reward += child.simulated_reward

            # update if change
            if total_reward / len(self.children) != self.simulated_reward:
                self.simulated_reward = total_reward / len(self.children)
                if self.parent:
                    self.parent.update_reward()
