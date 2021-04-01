from __future__ import annotations

import numpy as np
from constants import width, col


class Node:
    """
    Tree Node from monte carlo tree search
    """

    def __init__(self, expected_reward=0, probability=0, board=None, parent=None):
        if not board:
            self.board = np.zeros((width, col))

        self.children = []  # states reaching by taking action a from current state
        self.visit_count = 0  # number of times visited during mtcs
        self.expected_reward = expected_reward  # expected reward from being in current state
        self.probability = probability  # probability of current node being played
        self.parent = parent

    def _ucb_score(self):
        """
        Ucb score of node
        computed as Q(s, a) + U(s, a)
        Q(s, a) = (1 - alpha)( expected reward) + (alpha) (simulated reward) / visit_count
        U(s,a) = prior_probability / 1 + visit_count
        :return:
        """
        u = self.probability / 1 + self.visit_count
        q = 0
        return u + q

    def select_node(self) -> Node:
        """
        returns descendant with best
        :return:
        """
        self.visit_count += 1

        if not self.children:  # terminal
            return self

        max_ucb = 0
        selected_child = None
        for child in self.children:  # traverse path of greatest ucb
            if child.get_ucb() > max_ucb:
                selected_child = child
                max_ucb = child.get_ucb()

        return selected_child.select_node()
