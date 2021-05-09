from src.constants import row, col, min_leaves_aggregation
from src.mcts.node import Node
from collections import deque
import numpy as np


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


def preprocess_data(states: np.ndarray, priors: np.ndarray, values: np.ndarray):
    """
    Pre-process data into format
    Leaf nodes have high variance and do not necessarily reflect our true value/policy function
    to fix this, we can take an average among these leaf nodes
    """

    new_states, new_priors, new_values = [], [], []
    process_map = {}
    N, D = states.shape

    for i in range(N):
        total_visits = np.sum(priors[i])
        state = states[i].tobytes()
        if np.sum(total_visits) < min_leaves_aggregation:

            # Aggregate into statistics into process map
            if state in process_map:
                process_map[state]['priors'] += priors[i]
                process_map[state]['values'] += values[i]
                process_map[state]['visits'] += total_visits
            else:
                process_map[state] = {}
                process_map[state]['priors'] = priors[i]
                process_map[state]['values'] = values[i]
                process_map[state]['visits'] = total_visits

            # Add aggregated statistic into training data
            if process_map[state]['visits'] >= min_leaves_aggregation:
                new_states.append(
                    np.frombuffer(state).reshape(col * row).astype(int)
                )
                new_values.append(process_map[state]['values'] / process_map[state]['visits'])
                new_priors.append(process_map[state]['priors'] / process_map[state]['visits'])
                process_map[state]['priors'] = 0
                process_map[state]['values'] = 0
                process_map[state]['visits'] = 0
        else:
            new_states.append(states[i])
            new_priors.append(priors[i]/np.sum(priors[i]))
            new_values.append(values[i]/np.sum(priors[i]))

    return np.array(new_states), np.array(new_priors), np.array(new_values)
