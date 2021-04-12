"""
Game play for
"""

from connect4.agent import Agent
from constants import PLAY
from connect4.util import Board
import time
from numba import jit

def run(player_0: Agent, player_1: Agent):
    """
    simulate game with 2 agents
    :return:
    """

    board = Board.empty()
    num_turns = 0

    players = {
        0: player_0,
        1: player_1
    }

    is_terminal = True

    start = time.time()
    while is_terminal:
        # print(f'Turn {num_turns}')
        turn = num_turns % 2

        # play
        action = players[turn].play()
        players[(num_turns + 1) % 2].update_board(action)

        board = board.play_action(action)
        # print(board)
        num_turns += 1
        is_terminal = board.state == PLAY

    print(f' {is_terminal} {time.time() - start} s')
