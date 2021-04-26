from src.constants import row, col
from src.mcts.node import Node
from collections import deque, defaultdict


def record_tree(root: Node):
    """
    Convert tree root into training data
    and serialize as pkl
    :param root:
    :return:
    """
    priors = []
    value = []
    states = []

    q = deque([root])

    # use bfs to record state -> value pairs and state -> action distribution
    while q:
        node = q.pop()

        # add to q

        for child in node.children:
            if not child.is_terminal:
                q.append(child)

        states.append(node.board.board.reshape(col * row))
        value.append(node.total_simulated_reward)

        dist = [0] * row
        for child in node.children:
            dist[child.action_id] = child.visit_count
        priors.append(dist)

    return priors, value, states


def preprocess_data(states, priors, values):
    """
    Pre-process data into format
    Leaf nodes have high variance and do not necessarily reflect our true value/policy function
    to fix this, we can take an average among these leaf nodes
    """

    new_states, new_priors, new_values = [], [], []
    process_map = {}

    for i in range(states):
        print('i hope my neural network learns D:')
