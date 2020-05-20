from Genetic_Algorithm import *
# from Snake_Game import *
import numpy as np
import sys
# n_x -> no. of input units
# n_h -> no. of units in hidden layer 1
# n_h2 -> no. of units in hidden layer 2
# n_y -> no. of output units
sol_per_pop = 50
# num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y
num_weights = 9*12+12*16+16*3

# Defining the population size.
pop_size = (sol_per_pop,num_weights)
#Creating the initial population.

# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

num_generations = 100

num_parents_mating = 12

def calNextGene(prevGene):
    new_population = np.random.choice(np.arange(-1,1,step=0.01),size=pop_size,replace=True)
    # fitness = cal_pop_fitness(new_population,generation)# score return 
    # parents = select_mating_pool(new_population, fitness, num_parents_mating)
    parent = np.empty((num_parents_mating, new_population.shape[1]))
    parent[0:num_parents_mating] = prevGene[0:num_parents_mating]
    offspring_crossover = crossover(parent, offspring_size=(pop_size[0] - parent.shape[0], num_weights))
    offspring_mutation = mutation(offspring_crossover)
    new_population[0:parent.shape[0], :] = parent
    new_population[parent.shape[0]:, :] = offspring_mutation
    return new_population

def reshapeMat(mat):
    weights = []
    # print(mat, file=sys.stderr)
    w1 = np.array(mat[0:9*12]).reshape(9,12).tolist()
    w2 = np.array(mat[9*12:9*12+12*16]).reshape(12,16).tolist()
    w3 = np.array(mat[9*12+12*16:]).reshape(16,3).tolist()
    weights.append(w1)
    weights.append(w2)
    weights.append(w3)
    return weights

# for generation in range(num_generations):
#     print('##############        GENERATION ' + str(generation)+ '  ###############' )
#     # Measuring the fitness of each chromosome in the population.
#     fitness = cal_pop_fitness(new_population,generation)

#     print('#######  fittest chromosome in gneneration ' + str(generation) +' is having fitness value:  ', np.max(fitness))
#     # Selecting the best parents in the population for mating.
#     parents = select_mating_pool(new_population, fitness, num_parents_mating)

#     # Generating next generation using crossover.
#     offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

#     # Adding some variations to the offsrping using mutation.
#     offspring_mutation = mutation(offspring_crossover)

#     # Creating the new population based on the parents and offspring.
#     new_population[0:parents.shape[0], :] = parents
#     new_population[parents.shape[0]:, :] = offspring_mutation
