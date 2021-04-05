"""
Game play for
"""

from agent import Agent
import numpy as np
from constants import col, row
from util import get_stack, get_new_state, WIN, DRAW


def run(player_0: Agent, player_1: Agent):
    """
    simulate game with 2 agents
    :return:
    """

    board = np.zeros((col, row))
    stack = get_stack(board)
    num_turns = 0

    players = {
        0: player_0,
        1: player_1
    }

    is_terminal = False

    while is_terminal:

        turn = num_turns % 2
        player = (-1) ** turn

        # play
        action = players[turn].play()
        players[(num_turns + 1) % 2].update_board(action)

        # check board status
        new_state = get_new_state(board, stack, action)
        is_terminal = new_state == WIN or new_state == DRAW

        # update board
        board[action][col - stack[action]] = player
        num_turns += 1

