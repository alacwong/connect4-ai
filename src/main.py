from ml.model import AlphaPolicyModel, AlphaValueModel
from connect4.agent import RandomAgent, MCTSAgent
from connect4.game import run
import time
from ml.train_util import record_tree

if __name__ == '__main__':
    print('Connect 4 Ai')

    alpha_value = AlphaValueModel()
    alpha_policy = AlphaPolicyModel()

    agent_1 = MCTSAgent(
        value_network=alpha_value,
        policy_network=alpha_policy
    )

    agent_2 = RandomAgent()

    start = time.time()
    run(agent_1, agent_2)
    print(f' {time.time() - start} s')
