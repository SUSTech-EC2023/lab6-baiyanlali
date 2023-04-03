import random
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sin(10*np.pi*x)*x + np.cos(2*np.pi*x)*x

def create_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)

def dis(i, j):
    return abs(i-j)

def sh(d, sigma, alpha):
    if d < sigma:
        return 1 - (d/sigma) ^ alpha
    return 0

# TODO: Implement fitness sharing
def fitness_sharing(population, fitness, sigma_share, alpha):
    '''
    Parameters:
    population (np.array): Array of population values
    fitness (np.array): Array of fitness values
    sigma_share (float): Sharing distance
    alpha (float): Sharing exponent

    Return:
    shared_fitness
    '''
    # your code here
    size = len(population)
    shared_fitness = np.zeros((size))
    sh : np.array = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            sh[i, j] = sh(dis(i,j))
    sum = sh.sum()

    for i in range(size):
        shared_fitness[i] = fitness[i]/sum

    return shared_fitness


def tournament_selection(population, fitness, k = 2):
    '''
    @describe: the tournament selection method, select k individuals from polulation, the best one is selected as one parent
    @param: Determine by yourself
    @return parent
    '''
    selected = random.sample(population, k)
    return min(selected, key=lambda x: fitness[x])

# TODO: Implement your own parent selection operator
def selection(population, fitness):
    # your code here
    return tournament_selection(population, fitness)

# TODO: Implement your own crossover operator
def crossover(parent1, parent2):
    # your code here
    cut = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]
    child2 = parent2[:cut] + [c for c in parent1 if c not in parent2[:cut]]
    return child1, child2


# TODO: Implement your own mutation operator
def mutation(individual):
    # your code here
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]


# TODO: Implement main genetic algorithm process
def run_genetic_algorithm(population_size, generations, lower_bound, upper_bound,
                          mutation_rate, sigma_share, alpha):
    '''
    Parameters:
    population_size (int): Size of population
    generations (int): Number of generations
    lower_bound (float): Lower bound of search space
    upper_bound (float): Upper bound of search space
    mutation_rate (float): Mutation rate
    sigma_share (float): Sharing distance
    alpha (float): Sharing exponent

    Return:
    final_population (np.array): Array of final population values
    '''
    # Initialize population
    population = create_population(population_size, lower_bound, upper_bound)

    # Run GA
    for generation in range(generations):
        fitness = objective_function(population)
        shared_fitness = fitness_sharing(population, fitness, sigma_share, alpha)

        new_population = []
        for _ in range(population_size):
            # TODO: Implement evolutionary process
            
            pass

    return

def plot_population(population, generation):
    x = np.linspace(lower_bound, upper_bound, 1000)
    y = objective_function(x)
    plt.plot(x, y, label="Objective function")

    plt.scatter(population, objective_function(population), color="red", label=f"Population (Gen {generation})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



# GA parameters
population_size = 100
generations = 100
lower_bound = 0
upper_bound = 1
mutation_rate = 0.1
sigma_share = 0.1
alpha = 1

population = run_genetic_algorithm(population_size, generations, lower_bound, upper_bound,
                                   mutation_rate, sigma_share, alpha)

# Plot results
x = np.linspace(lower_bound, upper_bound, 1000)
y = objective_function(x)
plt.plot(x, y, label="Objective function")

plt.scatter(population, objective_function(population), color="red", label="Final population")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
