from ml.model import AlphaPolicyModel, AlphaValueModel
from connect4.agent import RandomAgent, MCTSAgent
from connect4.game import run
from connect4.util import Board
import time

if __name__ == '__main__':
    print('Hello world!')
    start = time.time()

    alpha_value = AlphaValueModel()
    alpha_policy = AlphaPolicyModel()

    agent_1 = MCTSAgent(
        board=Board.empty(),
        player=1,
        value_network=alpha_value,
        policy_network=alpha_policy
    )

    agent_2 = RandomAgent(
        board=Board.empty(),
        player=-1
    )

    run(agent_1, agent_2)
    print(f'{time.time() - start} s')
