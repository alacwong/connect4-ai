"""
Connect 4 board logic
"""

width = 5
col = 7


def update_board(board, stack, action, player):
    """Update board position"""

    directions = [
        (0, 1),
        (1, 0),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1)
    ]

    board[action][stack[action]] = player
    stack[action] += 1
    pos_x, pos_y = (action, stack[action])

    for direction in directions:
        dir_x, dir_y = direction
        x, y = pos_x, pos_y
        count = 0

        while 0 <= x < 6 and 0 <= y < 7 and board[x][y] == player:
            count += 1
            x, y = x + dir_x, y + dir_y

        # terminal board state
        if count >= 4:
            return board, stack, True

    # not terminal state
    return board, stack, False


def expand_board(node):
    """
    expand node's children
    :return:
    """

    # generate all legal actions
    # generate board state from actions
    # check if states are terminal
    # set node's children to generated children nodes
    pass
