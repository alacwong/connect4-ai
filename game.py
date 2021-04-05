"""
Game play for
"""

from agent import Agent
import numpy as np
from constants import col, row


def run(agent_a: Agent, agent_b: Agent):
    """
    simulate game with 2 agents
    :return:
    """

    board = np.zeros((col, row))
    turn = 1
    players = {
        1:  agent_a,
        -1: agent_b
    }

    is_terminal = False

    while is_terminal:
        action = players[turn]

