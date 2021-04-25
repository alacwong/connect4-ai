"""
Connect 4 board logic
"""
from __future__ import annotations

from typing import Set

import numpy as np

from src.constants import row, col, WIN, PLAY, DRAW

# Compute diagonal mapping and then cache
diagonal_ref = np.ones((col, row), dtype='int')
for index in range(-row + 1, row, 1):
    diagonal = np.diag(diagonal_ref, index)
    diagonal.setflags(write=True)
    diagonal.fill(index)

reverse_diagonal_ref = np.flipud(diagonal_ref)

# Compute and cache indices of diagonal elements
diagonal_indices = np.ones((col, row), dtype='int')
for index in range(-row + 1, row, 1):
    diagonal = np.diag(diagonal_indices, index)
    diagonal.setflags(write=True)
    np.put(diagonal,
           np.array([ii for ii in range(abs(len(diagonal)))], dtype='int'),
           np.array([ii for ii in range(abs(len(diagonal)))], dtype='int')
           )
reverse_diagonal_indices = np.flipud(diagonal_indices)


class Board:
    """
    Wrapper class abstraction for connect 4 board data structure
    """

    @staticmethod
    def empty():
        return Board(
            board=np.zeros(
                (col, row),
                dtype='int'
            ),
            state=PLAY,
        )

    def __init__(self, board: np.ndarray, state, stack=None):
        self.board = board
        self.state = state
        if stack is None:
            self.stack = np.array([np.count_nonzero(board[:, i]) for i in range(row)])
        else:
            self.stack = stack

    @staticmethod
    def _is_win(array, start_index) -> bool:
        """
        Determines if there is a 4 in a row
        """

        my_index = start_index - 1
        count = 1
        while my_index > 0:
            if array[my_index] == 1:
                count += 1
                my_index -= 1
            else:
                break

        my_index = start_index + 1
        while my_index < len(array):
            if array[my_index] == 1:
                count += 1
                my_index += 1
            else:
                break

        return count >= 4

    def play_action(self, action):
        """
        computes new board generated from perform action a
        :param action:
        :return:
        """

        # Copy data and then play action with copied data
        new_board = self.board.copy() * -1
        stack = self.stack.copy()
        count = col - stack[action] - 1
        new_board[count][action] = 1
        stack[action] += 1

        # Compute current state
        state = PLAY

        # Draw
        if np.sum(stack) == (col * row):
            state = DRAW

        # Horizontal
        current_row = new_board[count, :]
        if self._is_win(current_row, count):
            state = WIN

        # Vertical case
        current_col = new_board[:, action]
        if self._is_win(current_col, action):
            state = WIN

        # Diagonal cases (first diagonal)
        diagonal_index = diagonal_ref[count][action]
        current_diagonal = new_board.diagonal(diagonal_index)
        if self._is_win(current_diagonal, diagonal_indices[count, action]):
            state = WIN

        # Second diagonal
        diagonal_index = reverse_diagonal_ref[count][action]
        current_diagonal = np.flipud(new_board).diagonal(diagonal_index)
        if self._is_win(current_diagonal, reverse_diagonal_indices[count, action]):
            state = WIN

        # Game does not terminate
        return Board(
            board=new_board,
            state=state,
            stack=stack
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


if __name__ == '__main__':
    # Show matrices
    print(diagonal_ref)
    print(reverse_diagonal_ref)
    print(diagonal_indices)
    print(np.flipud(diagonal_indices))

    # Test indices are correct
    print(diagonal_indices.diagonal())
    print(np.flipud(reverse_diagonal_indices).diagonal())

    # Show diagonals work
    test_board = Board.empty()
    test_board.board[5, 5] = 1
    print(test_board)
    print(test_board.board.diagonal(diagonal_ref[5, 5]))
    print(np.flipud(test_board.board).diagonal(reverse_diagonal_ref[5, 5]))
