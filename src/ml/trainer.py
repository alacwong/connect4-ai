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
import csv
import numpy as np
import uuid

from src.ml.train_util import record_tree
from src.constants import PLAY, DRAW
from src.connect4.board import Board
from src.connect4.agent import Agent, AgentFactory
from src.ml.game_log import GameLog
from src.ml.nn import get_value_network, get_policy_network
from src.ml.wrapper import AlphaPolicyModel, AlphaValueModel


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

        num_iterations = 0

        value_network = get_value_network()
        policy_network = get_policy_network()
        kwargs = {
            'value_network': AlphaValueModel(network=value_network),
            'policy_network': AlphaPolicyModel(network=policy_network)
        }
        agent = AgentFactory.get_agent('mcts', kwargs=kwargs)
        self.agents = {agent.get_agent_type(): kwargs}
        self.current_agent = agent

        # may increase during later iterations
        max_games = 100

        while num_iterations < self.max_iterations:
            self._train(agent, max_games)

            value_network = get_value_network()
            policy_network = get_policy_network()
            states = np.array(self.states, dtype=int)
            priors = np.array(self.priors)
            values = np.array(self.values)
            value_network.fit(states, values)
            policy_network.fit(states, priors)
            kwargs = {
                'value_network': AlphaValueModel(network=value_network),
                'policy_network': AlphaPolicyModel(network=policy_network)
            }
            agent_type = 'todo'
            value_network.save(f'generated/{agent_type}')
            policy_network.save(f'generated/{agent_type}')
            agent = AgentFactory.get_agent('mcts', **kwargs)
            # TODO need to clear training data after fitting
            num_iterations += 1

    def _train(self, agent, max_games):
        """
        Train ai with
        """

        num_games = 0

        while num_games < max_games:

            board = Board.empty()
            num_turns = 0
            opposition = self._sample_opponent()

            flip = np.random.randint(0, 2)
            if flip == 0:
                player_1, player_2 = agent, opposition
            else:
                player_1, player_2 = opposition, agent

            players = {
                1: player_1,
                2: player_2
            }

            while True:
                turn = num_turns % 2

                # play
                action = players[turn].play()
                players[(num_turns + 1) % 2].update_board(action)

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

            num_games += 1

    def _update_data(self, agent_1: Agent, agent_2: Agent):
        """ Record training data from """

        def update(agent):
            priors, values, states = record_tree(agent.tree)
            self.priors.append(priors)
            self.values.append(values)
            self.states.append(states)

        if agent_1.get_agent_type() == self.current_agent:
            update(agent_1)

        if agent_2.get_agent_type() == self.current_agent:
            update(agent_2)

    @staticmethod
    def _serialize_data(data, path):
        with open(path, 'a') as f:
            writer = csv.writer(f)
            for cluster in data:
                for element in cluster:
                    writer.writerow(element)

    def _sample_opponent(self) -> Agent:
        """
        Draws opponent randomly based on win rate vs other agents
        """
        log = self.game_log.log

        agents = np.array([game for game in log])
        dist = np.array(
            [game['l'] / (game['l'] + game['w']) for game in log]
        )
        selected_agent = agents[np.random.choice(np.arange(len(agents)), p=dist)]
        return AgentFactory.get_agent(
            self.agents[selected_agent]['agent_type'],
            **self.agents[selected_agent]['kwargs']
        )

    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.stats = {'games': []}
        self.current_agent = uuid.uuid4()
        self.agents = {
            uuid.uuid4(): {},
        }
        self.game_log = GameLog([])
        self.priors = []
        self.values = []
        self.states = []


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
