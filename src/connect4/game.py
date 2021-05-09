"""
Game play for
"""

from src.connect4.agent import Agent
from src.constants import PLAY
from src.connect4.board import Board
import time


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

    start = time.time()

    while True:
        turn = num_turns % 2

        # play
        action = players[turn].play()
        players[(num_turns + 1) % 2].update_state(action)

        board = board.play_action(action)
        # print(f'{players[turn].get_agent_type()} plays {action}')
        # print(board)

        num_turns += 1

        if board.state != PLAY:
            break

    print(f' {time.time() - start} s')
    if num_turns % 2 == 1:
        print(f'{player_0.get_agent_type()} wins!')
    else:
        print(f'{player_1.get_agent_type()} wins!')
    print(board)
