"""
Connect 4 board logic
"""
from constants import row, col, directions, WIN, PLAY, DRAW
import numpy as np
from typing import Set


class Board:
    """
    Wrapper class abstraction for connect 4 board data structure
    """

    @staticmethod
    def get_empty(player):
        return Board(
            board=np.zeros(
                (col, row)
            ),
            state=PLAY,
            player=player
        )

    def __init__(self, board: np.ndarray, player, state):
        self.board = board
        self.state = state
        self.stack = np.array([np.count_nonzero(board[:, i]) for i in range(row)])
        self.player = player

    def play_action(self, action):
        """
        computes new board generated from perform action a
        :param action:
        :return:
        """

        new_board = -1 * self.board.copy()
        count = col - self.stack[action]
        new_board[count][action] = -1 * self.player
        state = None

        pos_x, pos_y = action, count

        # terminal (check if win)
        for direction in directions:
            dir_x, dir_y = direction
            x, y = pos_x, pos_y
            count = 0

            while 0 <= x < row and 0 <= y < col and new_board[y][x] == 1:
                count += 1
                x, y = x + dir_x, y + dir_y

                # terminal board state connect 4!
                if count >= 4:
                    state = WIN

        if np.sum(self.stack) == (col * row - 1) and not state:
            state = DRAW
        else:
            state = PLAY

        return Board(
            board=new_board,
            player=-1 * self.player,
            state=state
        )

    def get_valid_actions(self) -> Set:

        actions = set()
        for elem in self.stack:
            if elem < col:
                actions.add(elem)

        return actions

    def __copy__(self):
        return Board(
            board=self.board.copy(),
            state=self.state,
            player=self.player
        )

    def copy(self):
        return self.__copy__()

