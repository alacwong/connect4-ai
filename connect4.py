"""
Connect 4 board logic
"""
from node import Node
from constants import row, col, directions
import numpy as np
from model import PolicyModel, ValueModel


def update_board(board, stack, action, player):
    """Update board position"""

    board[action][stack[action]] = player
    stack[action] += 1
    pos_x, pos_y = (action, stack[action])

    for direction in directions:
        dir_x, dir_y = direction
        x, y = pos_x, pos_y
        count = 0

        while 0 <= x < 6 and 0 <= y < 7 and board[x][y] == player:
            count += 1
            x, y = x + dir_x, y + dir_y

        # terminal board state
        if count >= 4:
            return board, stack, True

    # not terminal state
    return board, stack, False


def _get_stack(board: np.ndarray) -> np.ndarray:
    """
    Helper to count number available actions
    """

    return np.array(
        [col - np.count_nonzero(board[:, i]) for i in range(col)]
    )


def expand_board(node: Node, policy_network: PolicyModel, value_network: ValueModel):
    """
    expand node's children
    :return:
    """

    # generate all legal actions
    # generate board state from actions
    # check if states are terminal
    # set node's children to generated children nodes

    current_board = node.board
    stack = _get_stack(current_board)
    dist = policy_network.compute_policy(current_board)

    # remove illegal actions from action distribution
    for i in range(row):
        if stack[i] == col:
            dist[i] = 0

    # normalize distribution
    dist = dist / np.sum(dist)

    for i in range(row):
        if stack[i] != col:
            # update accordingly
            node.children.append(
                Node()
            )
