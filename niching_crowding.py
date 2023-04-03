import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sin(10 * np.pi * x) * x + np.cos(2 * np.pi * x) * x

def create_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)

# TODO: Implement your selection operator
def selection():
    # your code here
    return

# TODO: Implement your crossover operator
def crossover(parent1, parent2):
    # your code here
    return

# TODO: Implement your mutation operator
def mutation():
    # your code here
    return

# TODO: Implement main genetic algorithm process with crowding
def run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate):
    population = create_population(population_size, lower_bound, upper_bound)

    for generation in range(generations):
        fitness = objective_function(population)
        # your code here

    return population

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

population = run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate)

# Plot results
x = np.linspace(lower_bound, upper_bound, 1000)
y = objective_function(x)
plt.plot(x, y, label="Objective function")

plt.scatter(population, objective_function(population), color="red", label="Final population")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


