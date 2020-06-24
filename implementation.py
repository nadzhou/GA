import numpy as np 
import ga

import seaborn as sns
import matplotlib.pyplot as plt


def main(): 
    equation_inputs = [2, -5, 4.6, 4, -15, 8]
    num_weights = len(equation_inputs)


    solution_per_pop = 8
    num_parents_mating = 4


    pop_size = (solution_per_pop, num_weights)
    new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
    print(new_population)


    best_outputs = []
    num_generations = 1000

    for generation in range(num_generations): 
        print(f"Generation: {generation}")

        fitness = ga.calc_pop_fitness(equation_inputs, new_population)
        print(f"Fitness: ]n{fitness}")

        best_score = np.max(np.sum(new_population - equation_inputs, axis=1))
        best_outputs.append(best_score)
        print(f"Best scores: \n{best_score}")

        parents = ga.select_mapping_pool(new_population, fitness, 
                                        num_parents_mating)
        print(f"Parents: \n{parents}")

        offspring_crossover = ga.crossover(parents, 
                                            offspring_size=(pop_size[0] - parents.shape[0],
                                            num_weights)
                                            )                                       
        print(f"Crossover: {offspring_crossover}")                                    


        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
        print(f"Mutations: \n{offspring_mutation}")

        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation


    fitness = ga.calc_pop_fitness(equation_inputs, new_population)
    best_match_idx = np.where(fitness == np.max(fitness))

    plotter(best_outputs)
    plt.show()


def plotter(best_outputs): 
    x_axis = np.arange(0, len(best_outputs))
    fig = sns.lineplot(x_axis, best_outputs)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness score")

    return fig


if __name__ == '__main__': 
    main()






