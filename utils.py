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
    int_state = state.astype(int)     # because some of the functions turn queens into floats

    # index masks for diagonals
    diag_left = np.arange(n) - int_state
    # print(diag_left)
    diag_right = np.arange(n) + int_state
    # print(diag_right)

    return (count_pairs(int_state) + count_pairs(diag_left) + count_pairs(diag_right))


def generate_population(n, population_size):
  """generate random N_Queens states

  Args:
      n (int): number of Queens
      population_size (int): number of states to generate

  Returns:
      np.array(population_size, n): 2D array, rows = states, col = queens in cols
  """  
  
  rng = np.random.default_rng()
  return (rng.integers(n, size=(population_size, n)))