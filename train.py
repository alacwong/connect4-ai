"""
Train neural network

Plan for training networks (from mcts self play)

1. Randomly select previous agent to train
2. Play n games with m simulations per move
3. For each game, record mcts distributions and value from tree.
4. Split n games into k batches.
5. Average state values for each batch.
6. Train networks on batches.
7. Iterate with updated networks.

(batches may not be necessary due to simulations averaging out values, will try without or with small batches)
"""

from tensorflow import keras
from constants import col, row
from node import Node
from collections import deque, defaultdict
import numpy as np


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
            keras.layers.Dense(row, activation='softmax')
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
            keras.Input(shape=(col * row)),
            keras.layers.Dense(col * row, activation='relu'),
            keras.layers.Dense(col * row, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
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

        dist = [0] * row

        for child in node.children:
            dist[child.action_id] = child.visit_count

        prior[str(node.board)].append(np.array(dist))


if __name__ == '__main__':
    value_network = get_value_network()
    # print(value_network.summary())
    y = np.zeros(col * row).reshape(1, 42)
    y[0][1] = -1
    x = get_policy_network().predict(
        y
    )
    print(x)
