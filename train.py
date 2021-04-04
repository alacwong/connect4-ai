"""
Train neural network

Plan for training networks (from mcts self play)

1. Play n games with m simulations.
2. For each game, record mcts distributions and value from tree.
3. Split n games into k batches.
4. Average state values for each batch.
5. Train networks on batches.
6. Iterate with updated networks.

(batches may not be necessary due to simulations averaging out values, will try without or with small batches)
"""

from tensorflow import keras
from constants import col, row
from node import Node
from collections import deque, defaultdict


def get_policy_network():
    """
    policy architecture
    :return:
    """

    model = keras.Sequential(
        [
            keras.Input((col * row)),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((row,), activation='softmax')
        ]
    )

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_value_network():
    """
    value network architecture
    :return:
    """

    model = keras.Sequential(
        [
            keras.Input((col * row)),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((col * row), activation='relu'),
            keras.layers.Dense((1,), activation='sigmoid')
        ]
    )

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def record_tree(root: Node):
    """
    Convert tree root into training data
    and serialize as pkl
    :param root:
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
                q.append(child)

        value[str(node.board)].append(node.simulated_reward)

        # need to update mcts to include action number in children
        # so we can get the distribution
