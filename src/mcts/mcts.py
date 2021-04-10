""" Monte carlo tree search"""
from constants import max_iterations, row, PLAY, WIN, DRAW, exploration_constant, exploration_temperature
from mcts.node import Node
from ml.model import ValueModel, PolicyModel
import numpy as np
from numba import jit


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
# simulate each of the node's children using policy distribution

@jit(nopython=True)
def monte_carlo_tree_search(root: Node, value_model: ValueModel, policy_model: PolicyModel) -> Node:
    """
    Run monte carlo tree search
    1. Repeat for n simulations
    2. Select Node
    3. Expand Node
    4. Simulate descendants and iterate
    5. Return optimal child after simulations completion
    :return:
    """

    # Need to implement decreasing exploration factor
    # Need to stochastically select best moves during training
    # Need to update value to use mean with visit count and count total reward instead

    num_iterations = 0

    current_node = root

    while num_iterations < max_iterations:
        print(num_iterations)
        exploration_factor = (0.1 + num_iterations / max_iterations) * exploration_constant
        node = current_node.select_node(exploration_factor)

        if not node.is_terminal:
            expand_board(node, policy_model, value_model)
            reward = simulate(node, policy_model)
            node.update_reward(reward)

        num_iterations += 1

    chosen_child = None
    # sample first 7 moves stochastically
    if num_iterations < 7:
        tao = 1 / exploration_temperature

        dist = np.array([child.visit_count ** tao for child in root.children])
        dist /= np.sum(dist)
        action = np.random.choice(np.arange(len(root.children)), p=dist)
        chosen_child = root.children[action]
    else:  # choose optimal
        max_visit = 0
        for child in root.children:
            if child.visit_count > max_visit:
                max_visit = child.visit_count
                chosen_child = child

    return chosen_child


@jit(nopython=True)
def expand_board(node: Node, policy_network: PolicyModel, value_network: ValueModel):
    """
    Expand node's children
    1. generate all legal actions
    2. generate board states from actions
    3. deal with terminal edge case
    4. add new children to parent
    :return:
    """

    current_board = node.board
    valid_actions = current_board.get_valid_actions()
    dist = policy_network.compute_policy(current_board.board, valid_actions)

    for action in valid_actions:
        new_board = current_board.play_action(action)

        node.children.append(
            Node(
                action_id=action,
                expected_reward=value_network.compute_value(new_board.board),
                probability=dist[action],
                board=new_board,
                parent=node,
                is_terminal=new_board.state != PLAY
            )
        )


@jit(nopython=True)
def simulate(node, policy):
    """
    Simulate game end of game by sampling actions from policy
    :param node:
    :param policy:
    :return: expected value of node
    """

    if node.is_terminal:

        # compute actual reward
        if node.board.state == WIN:
            return 1
        else:  # must be draw
            return 0
    else:
        simulated_reward = 0

        # compute expected reward through simulation
        board = node.board.copy()

        end_simulation = True
        num_moves = 0

        while end_simulation:
            actions = board.get_valid_actions()
            dist = policy.compute_policy(board.board, actions)

            # randomly sample from distribution
            action = np.random.choice(np.arange(row), p=dist)
            board = board.play_action(action)

            if board.state == PLAY:
                num_moves += 1
            elif board.state == DRAW:
                end_simulation = False
            else:
                if num_moves % 2 == 0:  # Win
                    simulated_reward += 1
                else:  # Loss
                    simulated_reward -= 1
                end_simulation = False

        return simulated_reward
