from model import AlphaPolicyModel, AlphaValueModel
from agent import RandomAgent, MCTSAgent
import numpy as np
from constants import row, col
from game import run

if __name__ == '__main__':
    print('Hello world!')

    alpha_value = AlphaValueModel()
    alpha_policy = AlphaPolicyModel()

    agent_1 = MCTSAgent(
        board=np.zeros((col, row)),
        player=1,
        value_network=alpha_value,
        policy_network=alpha_policy
    )

    agent_2 = RandomAgent(
        board=np.zeros((col, row)),
        player=-1
    )

    run(agent_1, agent_2)
