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
    if state.dtype != 'int':
      state = state.astype(int)     # because some of the functions turn queens into floats

    # index masks for diagonals
    diag_left = np.arange(n) - state
    # print(diag_left)
    diag_right = np.arange(n) + state
    # print(diag_right)

    return (count_pairs(state) + count_pairs(diag_left) + count_pairs(diag_right))

def population_fitness(population):
  """return array of fitness for whole population

  Args:
      population (np.array[population_size, n]): np array of N-queens states

  Returns:
      np.array[population_size]: fitness for each population
  """  

  return np.asarray([get_state_fitness(state) for state in population])


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


def generate_permuted_population(n, population_size):
  """generate random N_Queens states with no queens in the same row or column

  Args:
      n (int): number of Queens
      population_size (int): number of states to generate

  Returns:
      np.array(population_size, n): 2D array, rows = states, col = queens in cols
  """  
  
  rng = np.random.default_rng()
  return rng.permuted((np.full((population_size, n), np.arange(n))), axis=1)