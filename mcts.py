""" Monte carlo tree search"""
from constants import max_iterations, row, col, PLAY, WIN, DRAW
from node import Node
from util import expand_board, get_stack, get_new_state
from model import MockValueModel, MockPolicyModel
import numpy as np


# ucb = Q(s,a) + u(s,a)
# u(s,a) = p(s,a)/ 1 + visit_count
# probability of playing action from state s decaying over visit count

# Compute Q(s,a) from from value network as well as simulated
# Use hyper parameter lambda
# Sources
# http://joshvarty.github.io/AlphaZero/
# https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319


# do until n simulations:
# select node (traverse tree by ucb)
# expand node's children
# simulate each of the node's children using policy distrbution

def monte_carlo_tree_search(root: Node) -> int:
    """
    Run monte carlo tree search
    1. Repeat for n simulations
    2. Select Node
    :return:
    """

    num_iterations = 0

    current_node = root
    value_model = MockValueModel()
    policy_model = MockPolicyModel()

    while num_iterations < max_iterations:
        node = current_node.select_node()

        if not node.is_terminal:
            expand_board(node, policy_model, value_model)
            reward = simulate(node, policy_model)
            node.update_reward(reward)

    max_visit, optimal_action = 0, 0
    for child in root.children:
        if child.visit_count > max_visit:
            max_visit = child.visit_count
            optimal_action = child.action_id

    return optimal_action


def simulate(node, policy):
    """
    Simulate game end of game by sampling actions from policy
    :param node:
    :param policy:
    :return: expected value of node
    """

    if node.is_terminal:

        # immediately propagate
        return node.expected_reward
    else:
        simulated_reward = 0

        # compute expected reward through simulation
        for child in node.children:
            board = child.board
            stack = get_stack(board)

            end_simulation = False
            num_moves = 0

            while end_simulation:

                dist = policy.compute_policy(board)

                # remove illegal actions from action distribution
                for i in range(row):
                    if stack[i] == col:
                        dist[i] = 0

                # normalize distribution
                dist /= np.sum(dist)

                # randomly sample from distribution
                action = np.random.choice(np.arange(row), p=dist)
                new_state = get_new_state(board, stack, action)

                # update board and continue simulating
                if new_state == PLAY:
                    stack[action] += 1
                    board[col - stack[action]] = 1
                    num_moves += 1
                elif new_state == DRAW:
                    simulated_reward += 0.5
                    end_simulation = True
                else:
                    # update reward (1 for win, 0 for loss)
                    if num_moves % 2 == 0:
                        simulated_reward += 1
                    end_simulation = True

        return simulated_reward / len(node.children)

