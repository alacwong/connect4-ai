from ml.wrapper import AlphaPolicyModel, AlphaValueModel
from connect4.agent import RandomAgent, MCTSAgent
from connect4.game import run
import time

# https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98

if __name__ == '__main__':
    print('Connect 4 Ai')

    alpha_value = AlphaValueModel()
    alpha_policy = AlphaPolicyModel()

    agent_1 = MCTSAgent(
        value_network=alpha_value,
        policy_network=alpha_policy,
        agent_type='mcts'
    )

    agent_2 = RandomAgent()

    start = time.time()
    run(agent_1, agent_2)
