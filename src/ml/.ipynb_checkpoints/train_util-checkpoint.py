from src.constants import row
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

    prior = defaultdict(list)
    value = defaultdict(list)

    q = deque([root])

    # use bfs to record state -> value pairs and state -> action distribution
    while q:
        node = q.pop()

        # add to q
        if not node.is_terminal:
            for child in node.children:
                if child.visit_count > min_leaf_value:
                    q.append(child)

        value[str(node.board.board)].append(node.total_simulated_reward/ node.visit_count)

        dist = [0] * row

        for child in node.children:
            dist[child.action_id] = child.visit_count

        prior[str(node.board)].append(np.array(dist))

    return prior, value
