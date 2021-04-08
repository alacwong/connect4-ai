
# Connect 4 constants
row = 6
col = 7
directions = (
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1)
)

# MCTS constants
max_iterations = 100
simulation_constant = 0.5
exploration_constant = 5
exploration_temperature = 0.2

# RL constants
DRAW = 3
WIN = 1
LOSS = 2
PLAY = 0
