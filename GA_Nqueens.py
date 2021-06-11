import numpy as np
import random
from matplotlib import pyplot as plt
from utils import get_state_fitness

rng = np.random.default_rng()


"""*************************************************************
            utility functions
************************************************************"""


def plot_setup():
    fig = plt.figure()
    plt.xlabel("Number of Generations")
    plt.ylabel("Avg Fitness")
    plt.suptitle("Fitness over Generations")
    return fig


def add_plot(fitness_over_time, population_size):

    plt.plot(np.arange(fitness_over_time.size),
             fitness_over_time, label=f'pop = {population_size}')


def plot_save(fig):
    plt.legend()
    fig.savefig(
        f"fit_NQueens.png"
    )


"""*************************************************************
            GA functions
************************************************************"""

def get_population_fitness(population):
    """returns the fitness for each state in the population

    Args:
        population (np.array[population_size, n]): population of N-queens states

    Returns:
        np.array[population_size]: fitness for each state in population
    """
    return np.asarray([get_state_fitness(state) for state in population])


def choose_parents(population, pop_fit, n):
    """return the index of 2 parents chosen using probability as a weight

    Args:
        population (np.array[population_size, n]): population of N queens states
        pop_fit (np.array[population_size]): fitness for each state in population

    Returns:
        [int, int]: index into population of two states
    """
    max_fit = np.sum(np.arange(n))

    # invert the fitness score from minimum is best to maximum is best so 
    # most fit has the highest probability
    inverted_pop_fitness = max_fit - pop_fit
    proportional_fit = inverted_pop_fitness / inverted_pop_fitness.sum()
    return (rng.choice(population.shape[0], 2, p=proportional_fit))


def mutate(child):
    """mutate the state by changing a randomly selected column to a randomly selected value 0 to 7

    Args:
        child (np.array[n]): N queens state to mutate

    Returns:
        np.array[n]: mutated N queens state
    """
    index = rng.integers(child.shape)
    value = rng.integers(child.shape)
    child[index] = value
    return child


def generate_children(parent1, parent2, MutationPct):
    """generate a pair of children from the given parents with a possible mutation

    Args:
        parent1 (np.array[n]): N queens state parent 1
        parent2 (np.array[n]): N queens state second parent, may be same as first

        MutationPct (float): probability that a child is mutated

    Returns:
        [np.array[n], np.array[n]]: list of two N queens states
    """
    n = parent1.shape[0]

    cross = random.randint(0, n-1)
    child1 = np.concatenate((parent1[:cross], parent2[cross:]))
    child2 = np.concatenate((parent2[:cross], parent1[cross:]))
    if rng.choice(2, 1, p=[1 - MutationPct, MutationPct]):
        child1 = mutate(child1)
    if rng.choice(2, 1, p=[1 - MutationPct, MutationPct]):
        child2 = mutate(child2)
    return [child1, child2]


def create_generation(population, p_fitness, mutation_pct):
    """generate the same number of children as are in the population
    each paring generates 2 children so we only need to do it pop_size/2 times.

    Args:
        population (np.array): array of N-queens states (each N element array of values 0-7)
        p_fitness (np.array): fittness for each state in the population
        mutation_pct (float): probability of mutation

    Returns:
        np.array: next generation of N-queens states
    """

    n = population.shape[1]
    next_gen = []
    for _ in range(0, population.shape[0], 2):
        p1, p2 = choose_parents(population, p_fitness, n)
        next_gen.extend(generate_children(
            population[p1], population[p2], mutation_pct))
    return np.asarray(next_gen)


def ga_Nqueens(population_size, n, num_iterations, mutation_pct):
    """use a genetic algorithm to create solutions to the N queens problem

    Args:
        population_size (int): size of population
        n (int:) number of queens
        num_iterations (int): number of generations 
        mutation_pct (float): probability of mutation in generated children`
    """
    # create an array to record average fitness for each generation
    accuracy_over_generations = np.zeros(num_iterations)

    # create population
    population = rng.integers(n, size=(population_size, n))
    population_fitness = get_population_fitness(population)

    start_state = population[0]

    for i in range(num_iterations):
        # create next generation from current + children
        next_generation = np.concatenate(
            (population, create_generation(population, population_fitness, mutation_pct)))

        # order the indexes by fitness, then select the top population_size
        next_gen_fitness = get_population_fitness(next_generation)
        most_fit = (next_gen_fitness).argsort()
        population = next_generation[most_fit[:population_size], :]
        population_fitness = next_gen_fitness[most_fit[:population_size]]

        accuracy_over_generations[i] = population_fitness.mean()
        # if one is a success, return it
        # success = np.where(population_fitness == 0)
        # if (success[0].size > 0):
        #     return(accuracy_over_generations, population[0], i+1)

    return(accuracy_over_generations, population[0], start_state)


if __name__ == "__main__":
    """*************************************************************
                main program
    ************************************************************"""

    n_sizes = [8, 12, 40, 88]
    population_sizes = [12, 18, 60, 132]
    fig = plot_setup()

    for p_s,n_s in zip(population_sizes,n_sizes):
        average_fitness, end_state, start_state = ga_Nqueens(p_s, n_s, 40, 0.001)
        print(f'Population size: {p_s}')
        print(f'Average fitness: {average_fitness}')
        print(f'An initial state: \n{start_state}')
        print(f'A final state: \n{end_state}')
        add_plot(average_fitness, p_s)

    plot_save(fig)
