"""
Connect 4 board logic
"""
from node import Node
from constants import row, col, directions, WIN, PLAY, DRAW
import numpy as np
from model import PolicyModel, ValueModel


def _get_stack(board: np.ndarray) -> np.ndarray:
    """
    Helper to count number available actions
    """

    return np.array(
        [col - np.count_nonzero(board[:, i]) for i in range(col)]
    )


def _new_state(board: np.ndarray, stack, action) -> int:
    """
    Helper to check if current board state is terminal
    a board state is terminal if it is w/l/draw
    :param board:
    :return:
    """

    pos_x, pos_y = action, stack[action]

    # terminal (check if win)
    for direction in directions:
        dir_x, dir_y = direction
        x, y = pos_x, pos_y
        count = 0

        while 0 <= x < row and 0 <= y < col and board[x][y] == 1:
            count += 1
            x, y = x + dir_x, y + dir_y

            # terminal board state connect 4!
            if count >= 4:
                return WIN

    # draw
    if np.sum(stack) == row * col - 1:
        return DRAW

    # non terminal
    return PLAY


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

        # reverse player and update move
        new_board = current_board.copy() * -1
        new_board[stack[i]] = 1

        # check if node is terminal

        if stack[i] != col:
            # update accordingly

            new_state = _new_state(current_board, stack, i)

            # terminal leaf node
            if new_state == WIN or new_state == DRAW:
                terminal = True
                actual_reward = 1 if new_state == WIN else 0
                expected_reward = actual_reward + value_network.compute_value(current_board)

                node.children.append(
                    Node(
                        expected_reward=expected_reward,
                        probability=dist[i],
                        board=new_board,
                        is_terminal=terminal,
                    )
                )

            else:
                node.children.append(
                    Node(
                        expected_reward=value_network.compute_value(current_board),
                        probability=dist[i],
                        board=new_board,
                    )
                )
