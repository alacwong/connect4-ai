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
    def empty():
        return Board(
            board=np.zeros(
                (col, row),
                dtype=int
            ),
            state=PLAY,
        )

    def __init__(self, board: np.ndarray, state):
        self.board = board
        self.state = state
        self.stack = np.array([np.count_nonzero(board[:, i]) for i in range(row)])

    def play_action(self, action):
        """
        computes new board generated from perform action a
        :param action:
        :return:
        """

        new_board = self.board.copy() * -1
        count = col - self.stack[action] - 1
        new_board[count][action] = 1
        state = None

        pos_x, pos_y = action, count

        # terminal (check if win)
        for direction in directions:
            dir_x, dir_y = direction
            x, y = pos_x, pos_y
            count = 0

            while state != WIN and 0 <= x < row and 0 <= y < col and new_board[y][x] == 1:
                count += 1
                x, y = x + dir_x, y + dir_y

                # terminal board state connect 4!
                if count >= 4:
                    return Board(
                        board=new_board,
                        state=WIN
                    )

        if np.sum(self.stack) == (col * row - 1) and not state:
            state = DRAW
        else:
            state = PLAY

        return Board(
            board=new_board,
            state=state
        )

    def get_valid_actions(self) -> Set:

        actions = set()
        for action, count in enumerate(self.stack):
            if count < col:
                actions.add(action)

        return actions

    def __copy__(self):
        return Board(
            board=self.board.copy(),
            state=self.state,
        )

    def copy(self):
        return self.__copy__()

    def __str__(self):
        return str(self.board)

