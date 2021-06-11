import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.decorators import cost
from pyswarms.utils.search import GridSearch
from pyswarms.utils.plotters import(plot_cost_history)
from utils import generate_permuted_population, get_state_fitness

n = 8    # n must be at least 4

swarm_size = 20


if __name__ == '__main__' :

  # Set-up hyperparameters
  options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':n-1, 'p':1}

  min_bound = np.zeros(n)
  max_bound = (n) * np.ones(n)
  bounds = (min_bound, max_bound)
  center = np.full(n, n//2, dtype=int)


  # Call instance of PSO
  # optimizer = ps.single.LocalBestPSO(n_particles=swarm_size, dimensions=n, options=options,bounds=bounds, init_pos=start)

  # optimizer = ps.single.LocalBestPSO(n_particles=swarm_size, dimensions=n, options=options,bounds=bounds, center=center)
  optimizer = ps.single.LocalBestPSO(n_particles=swarm_size, dimensions=n, options=options,bounds=bounds)

  cost, pos = optimizer.optimize(cost(get_state_fitness), iters=1000)
  print(f'{cost=}, {pos.astype(int)}')

  plot_cost_history(cost_history=optimizer.cost_history)
  plt.show()
  

