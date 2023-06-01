import random
import numpy as np
import os
import math
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

dt = np.dtype(int)

class Individual:
    def __init__(self, chromosome_length):
        self.chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        self.fitness = 0

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.chromosome)



class EvolutionaryAlgorithm:
    def __init__(self, population_size,fitness_function, mutation_rate, crossover_rate,data_filename):
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.data_filename = data_filename
        self.readfile()

    def readfile(self):
        # 从文件中读取数据
        with open(self.data_filename, 'r') as file:
            n = int(file.readline().strip())  # 读取数字n
            self.chromosome_length = n
            print(n)


if __name__ == '__main__':
    datafile = "./qapdata/chr12a.dat"
    EvolutionaryAlgorithm(1,2,3,4,datafile)
