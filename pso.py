import numpy as np
import pyswarms as ps
from pyswarms.utils.decorators import cost
from utils import get_state_fitness

n = 8

min_bound = np.zeros(n)
max_bound = (n) * np.ones(n)
bounds = (min_bound, max_bound)
center = np.full(n, n//2, dtype=int)

if __name__ == '__main__' :
  # Set-up hyperparameters
  options = {'c1': 2, 'c2': 2, 'w':0.9, 'k':n-1, 'p':1}

  # Call instance of PSO
  optimizer = ps.single.LocalBestPSO(n_particles=100, dimensions=n, options=options,bounds=bounds)

  # Perform optimization
  cost, pos = optimizer.optimize(cost(get_state_fitness), iters=1000)

  # print(f'{cost=}, {pos=}')
  print(f'{cost=}, {pos.astype(int)}')