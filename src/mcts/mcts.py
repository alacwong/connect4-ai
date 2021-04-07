""" Monte carlo tree search"""
from constants import max_iterations, row, PLAY, WIN, DRAW
from mcts.node import Node
from ml.model import ValueModel, PolicyModel
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
# simulate each of the node's children using policy distribution

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

    num_iterations = 0

    current_node = root

    while num_iterations < max_iterations:
        node = current_node.select_node()

        if not node.is_terminal:
            expand_board(node, policy_model, value_model)
            reward = simulate(node, policy_model)
            node.update_reward(reward)

        num_iterations += 1

    max_visit, optimal_child = 0, None
    for child in root.children:
        if child.visit_count > max_visit:
            max_visit = child.visit_count
            optimal_child = child

    return optimal_child


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

    # remove illegal actions from action distribution
    for i in range(row):
        if i not in valid_actions:
            dist[i] = 0

    # normalize distribution
    dist = dist / np.sum(dist)

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
        else:   # must be draw
            return 0.5
    else:
        simulated_reward = 0

        # compute expected reward through simulation
        for child in node.children:
            board = child.board.copy()

            end_simulation = True
            num_moves = 0

            while end_simulation:
                print('simulating')
                actions = board.get_valid_actions()
                dist = policy.compute_policy(board.board, actions)

                # randomly sample from distribution
                action = np.random.choice(np.arange(row), p=dist)
                board = board.play_action(action)

                if board.state == PLAY:
                    num_moves += 1
                elif board.state == DRAW:
                    simulated_reward += 0.5
                    end_simulation = False
                else:   # win/loss
                    if num_moves % 2 == 0:
                        simulated_reward += 1

        return simulated_reward / len(node.children)