"""
Game play for
"""

from agent import Agent
from typing import List
from abc import ABC


class InitAgents(ABC):

    def init_agents(self) -> List[Agent]:
        """
        Initialize agents
        :return:
        """
        pass


def run(load_agents: InitAgents):
    """
    simulate game with 2 agents
    :return:
    """

    agent_a, agent_b = load_agents.init_agents()
