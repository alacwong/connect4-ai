from src.constants import row, col
from src.mcts.node import Node
from collections import deque, defaultdict
import numpy as np


def record_tree(root: Node, min_leaf_value=10):
    """
    Convert tree root into training data
    and serialize as pkl
    :param root:
    :param min_leaf_value:
    :return:
    """

    if root.visit_count < min_leaf_value:
        return [], [], []

    priors = []
    value = []
    states = []

    q = deque([root])

    # use bfs to record state -> value pairs and state -> action distribution
    while q:
        node = q.pop()

        # add to q

        for child in node.children:
            if child.visit_count > min_leaf_value and not child.is_terminal:
                q.append(child)

        states.append(node.board.board.reshape(col * row))
        value.append(node.total_simulated_reward / node.visit_count)

        dist = [0] * row
        for child in node.children:
            dist[child.action_id] = child.visit_count
        priors.append(dist)

    return priors, value, states


def preprocess_data():
    """
    Pre-process data into format
    """
