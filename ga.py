import numpy as np 



def calc_pop_fitness(equation_inputs, pop): 
    return np.sum(np.e ** (pop*equation_inputs), axis=1)


def select_mapping_pool(pop, fitness, num_parents): 
    parents = np.empty((num_parents, pop.shape[1]))

    for parent_num in range(num_parents): 
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]

        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999

    return parents


def crossover(parents, offspring_size): 
    offspring = np.empty(offspring_size)

    crossover_point = np.uint8(offspring_size[1] / 2)


    for event in range(offspring_size[0]): 
        parent1_idx = event % parents.shape[0]
        parent2_idx = (event + 1) % parents.shape[0]

        offspring[event, 0:crossover_point] = parents[parent1_idx, 
                                                    0:crossover_point
                                                    ]

        offspring[event, crossover_point:] = parents[parent2_idx, 
                                                    crossover_point:
                                                    ]
    return offspring



def mutation(offspring_crossover, num_mutations=1): 
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)

    for idx in range(offspring_crossover.shape[0]): 
        gene_idx = mutations_counter - 1

        for mutation_num in range(num_mutations): 
            random_value = np.random.uniform(-1.0, 1.0, 1)

            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx += mutations_counter

    return offspring_crossover



