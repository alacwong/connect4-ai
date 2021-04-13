from abc import ABC, abstractmethod
from constants import PLAY
from connect4.board import Board
from connect4.agent import Agent, MCTSAgent


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


class RandomSingle(Trainer):

    def train(self):
        """
        Train ai with
        """

        num_games = 0

        while num_games < self.max_games:

            board = Board.empty()
            num_turns = 0
            agent = MCTSAgent()
            player_1 = agent
            player_2 = self._draw_opponent()

            players = {
                1: player_1,
                2: player_2
            }

            is_terminal = True

            while is_terminal:
                turn = num_turns % 2

                # play
                action = players[turn].play()
                players[(num_turns + 1) % 2].update_board(action)

                board = board.play_action(action)
                num_turns += 1
                is_terminal = board.state == PLAY

    def _draw_opponent(self) -> Agent:
        """
        Draws opponent randomly based on win rate vs other agents
        """
        pass

    def __init__(self, max_games):
        self.max_games = max_games
