import numpy as np
import pyswarms as ps
from pyswarms.utils.decorators import cost
from pyswarms.utils.search import GridSearch
from utils import generate_permuted_population, get_state_fitness

n = 8    # n must be at least 4

swarm_size = 20


if __name__ == '__main__' :

  # Set-up hyperparameters
  options = {'c1': [0.5,1, 2],
             'c2': [0.3, 1, 2], 
             'w': [0.2, 0.4, 0.9],
             'k': [n//2, n-3, n-1], 
             'p':[2]}

  min_bound = np.zeros(n)
  max_bound = (n) * np.ones(n)
  bounds = (min_bound, max_bound)
  center = np.full(n, n//2, dtype=int)


  # Call instance of PSO

  g = GridSearch(ps.single.LocalBestPSO, n_particles=swarm_size,dimensions=n, options=options, bounds=bounds, objective_func=cost(get_state_fitness), iters=100)
  cost, cost_options = g.search()
  print(f'{cost=}, {cost_options}')


  print(f'{n//2=}, {n-3=}, {n-1=}')
