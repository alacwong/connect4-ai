from __future__ import annotations

from src.connect4.board import Board
import numpy as np


class Node:
    """
    Tree Node from monte carlo tree search
    """

    def __init__(self, board: Board, action_id, expected_reward=0, probability=0, parent=None, is_terminal=False, depth=0):

        self.board = board
        self.action_id = action_id
        self.children = []  # states reaching by taking action a from current state
        self.visit_count = 0  # number of times visited during mcts
        self.expected_reward = expected_reward  # expected reward (initially based on value-network)
        self.total_simulated_reward = 0  # reward after simulating the world
        self.probability = probability  # probability of current node being played
        self.parent = parent
        self.is_terminal = is_terminal
        self.depth = depth

    def ucb_score(self, exploration_factor: float, prior_factor):
        """
        Ucb score of node
        computed as Q(s, a) + U(s, a)
        Q(s, a) = (1 - alpha)( expected reward) + (alpha) (simulated reward) / visit_count
        U(s,a) = prior_probability / 1 + visit_count
        :return:
        """
        u = exploration_factor * np.sqrt(self.probability / (1 + self.visit_count))
        if self.visit_count:
            simulated_value = - 1 * self.total_simulated_reward
        else:
            simulated_value = 0
        q = (1 - prior_factor) * self.expected_reward + (
            (prior_factor * simulated_value)
        )
        return -1 * (u + q)

    def select_node(self, exploration_factor: float, prior_factor: float) -> Node:
        """
        returns descendant with best
        :return:
        """

        if not len(self.children) or self.is_terminal:  # terminal or leaf
            return self

        max_ucb = -1000
        selected_child = None
        for child in self.children:  # traverse path of greatest ucb
            ucb = child.ucb_score(exploration_factor, prior_factor)
            if ucb > max_ucb:
                selected_child = child
                max_ucb = ucb

        return selected_child.select_node(exploration_factor, prior_factor)

    def update_reward(self, reward):
        """
        Update reward recursively
        :param reward:
        :return:
        """
        self.visit_count += 1
        self.total_simulated_reward += reward
        if self.parent:
            self.parent.update_reward(-reward)

    def __str__(self):
        """
        pretty print node
        """
        children = ''
        for child in self.children:
            if child.visit_count:
                value = -1 * child.total_simulated_reward / child.visit_count
            else:
                value = 0
            children += f'{child.action_id}: [value: {value}, visited: {child.visit_count}]\n'
        return f'Value: {self.total_simulated_reward / (self.visit_count + 1)} Visited: {self.visit_count}\n{children}'
