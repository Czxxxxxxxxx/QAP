import random
import numpy as np
import os
import math
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

class Individual:
    def __init__(self, chromosome_length):
        self.chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        self.chromosome_length = chromosome_length
        self.fitness =

    def random_permutation(self):
        nums = list(range(1, self.chromosome_length + 1))
        for i in range(self.chromosome_length - 1):
            j = random.randint(i, self.chromosome_length - 1)
            nums[i], nums[j] = nums[j], nums[i]

        self.chromosome = nums

    def calculate_fitness(self,fitness_function):
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
            # 读取矩阵A
            rows_A = []
            i = 0
            while i < n :
                row = list(map(int, file.readline().strip().split()))
                if len(row) == 0:
                    continue
                row = np.array(row)
                rows_A.append(row)
                i += 1
            A = np.array(rows_A,dtype=object)

            # 读取矩阵B
            rows_B = []
            i = 0
            while i < n:
                row = list(map(int, file.readline().strip().split()))
                if len(row) == 0:
                    continue
                row = np.array(row)
                rows_B.append(row)
                i += 1
            B = np.array(rows_B,dtype=object)
        self.Matrix_D = A
        self.Matrix_F = B
        # # 打印矩阵A和B
        # print("Matrix A:")
        # print(A)
        # print("\nMatrix B:")
        # print(B)

def getcost(A,B,perm):



if __name__ == '__main__':
    datafile = "./qapdata/chr12a.dat"

    EvolutionaryAlgorithm(population_size=100,fitness_function=2,mutation_rate=0.1,crossover_rate=0.1,data_filename=datafile)
