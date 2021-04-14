from abc import ABC, abstractmethod
from constants import PLAY
from connect4.board import Board
from connect4.agent import Agent, MCTSAgent
from ml.model import AlphaPolicyModel, AlphaValueModel
import numpy as np
import uuid

class Trainer(ABC):

    @abstractmethod
    def train(self):
        """
        Train mcts
        """
        pass

    @abstractmethod
    def _draw_opponent(self) -> Agent:
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
            opposition = self._draw_opponent()

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

    def _record_game(self, result: int, agent: Agent, opposition: Agent):
        """
        record game data
        """

        if agent.get_agent_id() == agent.get_agent_id():
            # record data from both agents if they have the same id
            pass
        else:
            pass

    def _draw_opponent(self) -> Agent:
        """
        Draws opponent randomly based on win rate vs other agents
        """
        pass

    def _get_agent(self, agent_id) -> Agent:
        """
        Generate agent models
        """
        kwargs = self.agents[agent_id]
        return MCTSAgent(**kwargs)

    def __init__(self, max_games):
        self.max_games = max_games
        self.stats = {}
        self.current_agent = uuid.uuid4()
        self.agents = {
            uuid.uuid4(): {},

        }


class RandomMulti(Trainer):
    """
    Train using multi-processing
    """

    def get_trained_model(self):
        pass

    def train(self):
        pass

    def _draw_opponent(self) -> Agent:
        pass
