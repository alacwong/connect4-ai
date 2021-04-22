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
from src.constants import PLAY
from src.connect4.board import Board
from src.connect4.agent import Agent, MCTSAgent, AgentFactory
from src.ml.game_log import GameLog
from src.ml.model_wrapper import AlphaPolicyModel, AlphaValueModel
import numpy as np
import uuid
from src.ml.train_util import record_tree


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
        """

    def _train(self, agent):
        """
        Train ai with
        """

        num_games = 0

        while num_games < self.max_games:

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

            if (num_turns % 2 == 0 and player_1.get_agent_name() == agent.get_agent_name()) \
                    or (num_turns % 2 == 1 and player_2.get_agent_name() == agent.get_agent_name()):
                self._record_game(1, agent, opposition)
            else:
                self._record_game(0, agent, opposition)

            num_games += 1

    def _record_game(self, result: int, agent: MCTSAgent, opposition: Agent):
        """
        record game data
        """

        prior, value = record_tree(agent.root)
        self.stats['games'].append(
            {
                'result': result,
                'prior': prior,
                'value': value,
            }
        )

        if opposition.get_agent_id() == agent.get_agent_id():
            # record data from both agents if they have the same id
            opposition_prior, opposition_value = record_tree(opposition.node)
            opposition_result = 1 if result == 0 else 0
            self.stats['games'].append(
                {
                    'result': opposition_result,
                    'prior': opposition_prior,
                    'value': opposition_value
                }
            )

    def _sample_opponent(self) -> Agent:
        """
        Draws opponent randomly based on win rate vs other agents
        """
        log = self.game_log.get_log()

        agents = np.array([game for game in log])
        dist = np.array(
            [game['l'] / (game['l'] + game['w']) for game in log]
        )
        selected_agent = agents[np.random.choice(np.arange(len(agents)), p=dist)]
        return AgentFactory.get_agent(
            self.agents[selected_agent]['agent_type'],
            **self.agents[selected_agent]['kwargs']
        )

    def _get_agent(self, agent_id) -> Agent:
        """
        Generate agent models
        """
        kwargs = self.agents[agent_id]
        return MCTSAgent(**kwargs)

    def __init__(self, max_games):
        self.max_games = max_games
        self.stats = {'games': []}
        self.current_agent = uuid.uuid4()
        self.agents = {
            uuid.uuid4(): {},
        }
        self.game_log = GameLog([])


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
