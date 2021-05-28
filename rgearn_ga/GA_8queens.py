import numpy as np
import random
from matplotlib import pyplot as plt

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
        f"fit_8Queens.png"
    )


"""*************************************************************
            GA functions
************************************************************"""


def count_pairs(state):
    """[summary]return the number of pairs of queens that are not attacking each other. 
    The higher the number the more fit the state. A score of 28 means there are no queens 
    attacking another.

    Args:
        state (np.array): one possible state for 8-queens, element values 0-7 representing 
        rows on the board

    Returns:
        int: number of pairs of queens not attacking each other: max 28
    """
    _, count = np.unique(state, return_counts=True)
    pairs = 0.5 * (count * (count - 1)).sum()
    return pairs


def get_state_fitness(state):
    """calculates the fitness score for a given state as the number of pairs 
    of queens that are not attacking. The higher the number, the more fit the state.

    Args:
        state (np.array[8]): single state of 8 queens problem 

    Returns:
        int: number of pairs of queens that are not attacking 0 indexed 
             between 8 and 27 inclusive
    """
    diag_left = np.arange(8) - state

    diag_right = np.arange(8) + state
    return (
        28
        - count_pairs(state)
        - count_pairs(diag_left)
        - count_pairs(diag_right)
    )


def get_population_fitness(population):
    """returns the fitness for each state in the population

    Args:
        population (np.array[population_size, 8]): population of 8-queens states

    Returns:
        np.array[population_size]: fitness for each state in population
    """
    return np.asarray([get_state_fitness(state) for state in population])


def choose_parents(population, pop_fit):
    """return the index of 2 parents chosen using probability as a weight

    Args:
        population (np.array[population_size, 8]): population of 8 queens states
        pop_fit (np.array[population_size]): fitness for each state in population

    Returns:
        [int, int]: index into population of two states
    """
    prop_fit = pop_fit / pop_fit.sum()
    return (rng.choice(population.shape[0], 2, p=prop_fit))


def mutate(child):
    """mutate the state by changing a randomly selected column to a randomly selected value 0 to 7

    Args:
        child (np.array[8]): 8 queens state to mutate

    Returns:
        np.array[8]: mutated 8 queens state
    """
    index = rng.integers(child.shape)
    value = rng.integers(8)
    child[index] = value
    return child


def generate_children(parent1, parent2, MutationPct):
    """generate a pair of children from the given parents with a possible mutation

    Args:
        parent1 (np.array[8]): 8 queens state parent 1
        parent2 (np.array[8]): 8 queens state second parent, may be same as first

        MutationPct (float): probability that a child is mutated

    Returns:
        [np.array[8], np.array[8]]: list of two 8 queens states
    """
    cross = random.randint(0, 7)
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
        population (np.array): array of 8-queens states (each 8 element array of values 0-7)
        p_fitness (np.array): fittness for each state in the population
        mutation_pct (float): probability of mutation

    Returns:
        np.array: next generation of 8-queens states
    """
    next_gen = []
    for _ in range(0, population.shape[0], 2):
        p1, p2 = choose_parents(population, p_fitness)
        next_gen.extend(generate_children(
            population[p1], population[p2], mutation_pct))
    return np.asarray(next_gen)


def ga_8queens(population_size, num_iterations, mutation_pct):
    """use a genetic algorithm to create solutions to the 8 queens problem

    Args:
        population_size (int): size of population
        num_iterations (int): number of generations 
        mutation_pct (float): probability of mutation in generated children`
    """
    # create an array to record average fitness for each generation
    accuracy_over_generations = np.zeros(num_iterations)

    # create population
    population = rng.integers(8, size=(population_size, 8))
    population_fitness = get_population_fitness(population)

    start_state = population[0]

    for i in range(num_iterations):
        # create next generation from current + children
        next_generation = np.concatenate(
            (population, create_generation(population, population_fitness, mutation_pct)))

        # order the indexes by fitness, then select the top population_size
        next_gen_fitness = get_population_fitness(next_generation)
        most_fit = (-next_gen_fitness).argsort()
        population = next_generation[most_fit[:population_size], :]
        population_fitness = next_gen_fitness[most_fit[:population_size]]

        accuracy_over_generations[i] = population_fitness.mean()
        # if one is a success, return it
        # success = np.where(population_fitness == 28)
        # if (success[0].size > 0):
        #     return(accuracy_over_generations, population[0], i+1)

    return(accuracy_over_generations, population[0], start_state)


if __name__ == "__main__":
    """*************************************************************
                main program
    ************************************************************"""

    population_sizes = [10, 100, 500, 1000]
    fig = plot_setup()

    for p_s in population_sizes:
        average_fitness, end_state, start_state = ga_8queens(p_s, 40, 0.001)
        print(f'Population size: {p_s}')
        print(f'Average fitness: {average_fitness}')
        print(f'An initial state: \n{start_state}')
        print(f'A final state: \n{end_state}')
        add_plot(average_fitness, p_s)

    plot_save(fig)
