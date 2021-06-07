import numpy as np


def get_state_fitness(state):
    """calculates the fitness score for a given state as the number of pairs 
    of queens that are attacking. 0 is perfect state

    Args:
        state (np.array[8]): single state of N queens problem 

    Returns:
        int: number of pairs of queens that are attacking
    """

    def count_pairs(state):
        _, count = np.unique(state, return_counts=True)
        pairs = 0.5 * (count * (count - 1)).sum()
        return pairs

    n = np.size(state)

    # index masks for diagonals
    diag_left = np.arange(n) - state
    diag_right = np.arange(n) + state

    return (count_pairs(state) + count_pairs(diag_left) + count_pairs(diag_right))
