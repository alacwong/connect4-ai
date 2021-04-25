import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Connect 4 constants
row = 7
col = 6

# MCTS constants
max_iterations = 200
simulation_constant = 0.5
exploration_constant = 10
exploration_temperature = 0.2

# RL constants
DRAW = 3
WIN = 1
LOSS = 2
PLAY = 0

# Agent types
MINIMAX = 'minimax'
MCTS = 'monte_carlo_tree_search'
HUMAN = 'human'
RANDOM = 'random'
QLEARN = 'deep_q_learning'

# Hardware
if tf.test.gpu_device_name() == '/device:GPU:0':
    device = '/device:GPU:0'
else:
    device = '/device:CPU:0'
