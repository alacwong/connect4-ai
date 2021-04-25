"""
Train neural network

Plan for training networks (from mcts self play)

1. Randomly select previous agent to train
2. Play n games with m simulations per move
3. For each game, record mcts distributions and value from tree.
4. Split n games into k batches.
5. Average state values for each batch.
6. Train networks on batches.
7. Iterate with updated networks.

(batches may not be necessary due to simulations averaging out values, will try without or with small batches)
"""

from abc import ABC, abstractmethod
import numpy as np
import uuid

from ml.model import Model
from src.ml.train_util import record_tree
from src.constants import PLAY, DRAW, MCTS
from src.connect4.board import Board
from src.connect4.agent import Agent, AgentFactory
from src.ml.game_log import GameLog
from src.ml.nn import get_value_network, get_policy_network
from src.ml.wrapper import AlphaPolicyModel, AlphaValueModel
from datetime import datetime


class Trainer(ABC):

    @abstractmethod
    def train(self):
        """
        Train mcts
        """
        pass

    @abstractmethod
    def _sample_opponent(self) -> Agent:
        """
        Create adversarial agent to train mcts agent against
        """
        pass

    @abstractmethod
    def get_trained_model(self):
        """
        get training data generated from self play
        """
        pass


class RandomSingle(Trainer):

    def get_trained_model(self):
        pass

    def train(self):
        """
        main training loop
        1. generate raw agent
        2. self play raw agent and previous iterations **
        3. grab training data from self play
        4. train on new neural network
        5. iterate/repeat
        """

        num_generations = 0
        agent = self.agents[self.current_agent].copy()

        # may increase during later iterations
        max_games = 1

        while num_generations < self.max_generations:
            print(f'Generation: {num_generations}')
            self._train(agent, max_games)

            # Train new model with
            value_network, policy_network = get_value_network(), get_policy_network()
            states, priors, values = np.array(self.states, dtype=int), np.array(self.priors), np.array(self.values)
            value_network.fit(states, values)
            policy_network.fit(states, priors)

            # Create new agent
            value_network = Model.from_keras(value_network)
            policy_network = Model.from_keras(policy_network)
            agent_type = f'mcts <Generation: {num_generations}>'
            kwargs = {
                'value_network': AlphaValueModel(model=value_network),
                'policy_network': AlphaPolicyModel(model=policy_network),
                'agent_type': agent_type
            }
            agent = AgentFactory.get_agent('mcts', **kwargs)

            # Save every kth generation (excluding the first)
            if num_generations % 5 == 0 and num_generations > 0:
                value_network.save(f'generated/{self.save_dir}/generation_{num_generations}/value')
                policy_network.save(f'generated/{self.save_dir}/generation_{num_generations}/policy')

            self.priors, self.values, self.states = [], [], []
            num_generations += 1

    def _train(self, agent, max_games):
        """
        Train ai with
        """

        num_games = 0

        while num_games < max_games:

            print(f'Game: {num_games + 1}')

            # Setup agents
            board = Board.empty()
            num_turns = 0
            opposition = self._sample_opponent()

            flip = np.random.randint(0, 2)
            if flip == 0:
                player_0, player_1 = agent, opposition
            else:
                player_0, player_1 = opposition, agent

            players = {
                0: player_0,
                1: player_1
            }

            # Connect 4 game loop
            while True:
                turn = num_turns % 2

                # play
                action = players[turn].play()
                players[(num_turns + 1) % 2].update_state(action)

                board = board.play_action(action)

                if board.state != PLAY:
                    break
                else:
                    num_turns += 1

            # Updates result of the game
            if board.state == DRAW:
                self.game_log.update(agent, opposition, 1)
                self.game_log.update(agent, opposition, 0)
            elif turn == 0 and flip == 0:
                self.game_log.update(agent, opposition, 1)
            elif turn == 0 and flip == 1:
                self.game_log.update(agent, opposition, 0)
            elif turn == 1 and flip == 0:
                self.game_log.update(agent, opposition, 0)
            else:
                self.game_log.update(agent, opposition, 1)

            # Reset player memories with founding titan
            agent.reset()
            opposition.reset()

            num_games += 1

    def _update_data(self, agent_1: Agent, agent_2: Agent):
        """ Record training data from """

        # Add to training data based on game play
        def update(agent):
            priors, values, states = record_tree(agent.tree)
            self.priors.extend(priors)
            self.values.extend(values)
            self.states.extend(states)

        if agent_1.get_agent_type() == self.current_agent:
            update(agent_1)

        if agent_2.get_agent_type() == self.current_agent:
            update(agent_2)

    def _sample_opponent(self) -> Agent:
        """
        Draws opponent randomly based on win rate vs other agents
        """
        log = self.game_log.log

        agents = np.array([agent for agent in log])
        dist = np.array(
            [log[agent]['l'] / (log[agent]['l'] + log[agent]['w']) for agent in log]
        )
        selected_agent = np.random.choice(np.array(agents), p=dist)
        return self.agents[selected_agent]

    def __init__(self, max_generations, initial_value, initial_policy):
        self.max_generations = max_generations
        self.current_agent = 'mcts: <Generation: Initial>'
        self.agents = {
            self.current_agent: AgentFactory.get_agent(
                MCTS, **{
                    'value_network': AlphaValueModel(initial_value),
                    'policy_network': AlphaPolicyModel(initial_policy),
                    'agent_type': 'mcts: <Generation: Initial>'
                }
            )
        }
        self.game_log = GameLog(self.agents)
        self.priors = []
        self.values = []
        self.states = []
        self.save_dir = f'generated/{datetime.today().isoformat()}'


class RandomMulti(Trainer):
    """
    Train using multi-processing
    """

    def get_trained_model(self):
        pass

    def train(self):
        pass

    def _sample_opponent(self) -> Agent:
        pass


if __name__ == '__main__':
    print('Test training :)')
    trainer = RandomSingle(
        max_generations=5,
        initial_value=Model.from_file('generated/initial/value/model.tflite'),
        initial_policy=Model.from_file('generated/initial/policy/model.tflite')
    )
    trainer.train()
