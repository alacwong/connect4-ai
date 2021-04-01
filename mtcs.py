""" Monte carlo tree search"""
from constants import max_iterations
from node import Node

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

def monte_carlo_tree_search():
    """
    Run monte carlo tree search
    1. Repeat for n simulations
    2. Select Node
    :return:
    """

    num_iterations = 0

    root = Node()
    current_node = root

    while num_iterations < max_iterations:
        node = current_node.select_node()

        if node.is_terminal:
            # immediately back-propagate value
            pass
        else:
            # expand
            pass

            # simulate


def simulate(node, policy) -> float:
    """
    Simulate game end of game by sampling actions from policy
    :param node:
    :param policy:
    :return: expected value of node
    """
    pass


