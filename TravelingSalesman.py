import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from tqdm import trange

def readTSP(filename):
    coord_list = []
    # x_list = []
    # y_list = []
    with open(filename) as fp:
        Lines = fp.readlines()
        for line in Lines:
            # print(line.strip())
            # print(line.strip().split("   "))
            x, y = line.strip().split("   ")
            coord_list.append((float(x), float(y)))
            # x_list.append(float(x))
            # y_list.append(float(y))
    return coord_list

def localSwapSearch(parent, coords, iterations=10):
    parent_copy = parent
    for _ in range(iterations):
        # try a swap
        indexes = random.sample(range(0, len(parent)), 2)
        parent[indexes[0]], parent[indexes[1]] = parent[indexes[1]], parent[indexes[0]]

        # if it is better, keep it
        if not (fitness(parent, coords) > fitness(parent_copy, coords)):
            parent = parent_copy
    return parent

def geneticAlgorithmTA(coords, mutate_prob, iterations=1000, population_size=10, memetic=False):
    # initialize fixed-size population
    population = [generateNumSeq(len(coords)) for _ in range(population_size)]
    fitness_list = []

    # evaluate quality of each candidate
    best_fitness, best_gene = bestFitness(population, coords)
    for _ in trange(iterations):

        # 1. crossover current population
        children = []
        for j in range(0, population_size-1):
            children.append(orderCrossover(population[j], population[j+1]))
        children.append(orderCrossover(population[population_size-1], population[0]))

        # 2. mutate current population
        for child in children:
            orderMutation(child, mutate_prob)

        # IF MEMETIC APPLY LOCAL SEARCH
        if memetic:
            for child in children:
                localSwapSearch(child, coords)

        # 3. if fitness is higher than previously found fitness, store it, also store best gene
        current_fitness, best_gene = bestFitness(children, coords)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
        fitness_list.append(best_fitness)

        # 4. remove one random gene
        random_gene = random.choice(population)
        population.remove(random_gene)

        # 5. add best gene
        population.append(best_gene)
    return fitness_list, best_gene

def generateNumSeq(length=50):
    return random.sample(range(0, length), length)

def bestFitness(num_sequences, coords):
    best_fitness = float('-inf')
    best_gene = []
    for num_seq in num_sequences:
        current_fitness = fitness(num_seq, coords)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_gene = num_seq
    return best_fitness, best_gene

def fitness(num_seq, coords):
    nr_cities = len(num_seq)
    total_distance = 0
    for i in range(0, nr_cities-1):
        total_distance += distance.euclidean(coords[num_seq[i]], coords[num_seq[i+1]]) # (0,2), (1,1), (2,0)
    return -(total_distance + distance.euclidean(coords[num_seq[nr_cities-1]], coords[num_seq[0]])) # loop last

def orderMutation(parent, mutate_prob):
    if random.random() > mutate_prob:
        return parent
    indexes = random.sample(range(0, len(parent)), 2)
    parent[indexes[0]], parent[indexes[1]] = parent[indexes[1]], parent[indexes[0]]
    return parent

def orderCrossover(parent_1, parent_2):
    if len(parent_1) <= 0 or len(parent_2) <= 0:
        raise SyntaxError('length of parent <=0')
    gene_length = len(parent_1)

    # generate 2 cut points
    geneA = int(random.random() * gene_length)
    geneB = int(random.random() * gene_length)
    while geneA == geneB:
        geneB = int(random.random() * gene_length)
    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    # take a splice of parent 1 from start_gene to end_gene
    child_parent_1 = [parent_1[i] for i in range(start_gene, end_gene)]

    # leftover of parent 2 = tail + head
    leftover_gene_parent_2 = parent_2[end_gene:gene_length] + parent_2[0:end_gene]

    # add whatever is left of parent 2 in order if it isn't in the gene already
    for c in leftover_gene_parent_2:
        if c not in child_parent_1:
            child_parent_1.append(c)

    return child_parent_1

def plotGA(fitness_list):
    sns.set_theme()
    sns.lineplot(data=fitness_list)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.show()

def plotTSP(coords):
    x, y = zip(*coords)
    sns.set_theme()
    plt.scatter(x, y, alpha=0.5)
    plt.title('Traveling Salesman Problem')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plotPath(gene):
    sns.set_theme()
    solution_tsp = []
    for index in gene:
        solution_tsp.append(list(coords[index]))
    plt.figure(figsize=(8, 8))
    plt.title("Final TSP solution")
    for j in range(0, len(gene) - 1):
        plt.plot((solution_tsp[j][0], solution_tsp[j+1][0]),
                 (solution_tsp[j][1], solution_tsp[j+1][1]), alpha=0.6, color='blue')
    plt.plot((solution_tsp[len(gene)-1][0], solution_tsp[0][0]),
             (solution_tsp[len(gene)-1][1], solution_tsp[0][1]), alpha=0.6, color='blue')
    x, y = zip(*coords)
    sns.set_theme()
    plt.scatter(x, y, alpha=0.5)
    plt.show()
    plt.xlabel("Final distance: " + str(-fitness_list[-1]))
    print(f"Final distance: {fitness_list[-1]}")

coords = readTSP('file-tsp.txt')
fitness_list, best_gene = geneticAlgorithmTA(coords, mutate_prob=0.1, iterations=5000, population_size=10, memetic=False)
# plotTSP(coords)
plotGA(fitness_list)
plotPath(best_gene)
# nr_cities = len(coords)
# take_nr_cities = 10
# coords = coords[0:take_nr_cities]
# print(coords)
# # showSalesMan(coords)
# num_seq = generateNumSeq(take_nr_cities-1)
# print(num_seq)
# test_coords = [(0.2554, 18.2366), (0.4339, 15.2476), (0.7377, 8.3137)]
# test_seq = [0, 1, 2]
# test_seq_2 = [2, 1, 0]
# test_seq_3 = [1, 2, 0]
# print(fitness(num_seq, coords))
# print(fitness(test_seq, test_coords))
# print(fitness(test_seq_2, test_coords))
# print(fitness(test_seq_3, test_coords))
# p1 = [3, 5, 7, 2, 1, 6, 4, 8]
# p2 = [2, 5, 7, 6, 8, 1, 3, 4]
# print(p1)
# print(p2)
# np1 = orderCrossover(p1, p2)
# print(np1)
# mp1 = orderMutation(np1)
# print(mp1)