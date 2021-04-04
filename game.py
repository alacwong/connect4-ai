"""
Game play for
"""

from agent import Agent
import numpy as np
from constants import col, row


def run(agent_a: Agent, agent_b: Agent):
    """
    simulate game with 2 agents
    :param agent_a:
    :param agent_b:
    :return:
    """

    board = np.zeros((row, col))


