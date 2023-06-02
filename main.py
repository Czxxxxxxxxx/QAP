import random
import numpy as np
import os
import math
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

class Individual:
    def __init__(self, chromosome_length):
        self.chromosome_length = chromosome_length
        self.chromosome = self.get_random_permutation()
        self.fitness = -1

    #直接使用列表进行初始化
    def __init__(self, chromosome_length, list):
        self.chromosome_length = chromosome_length
        self.chromosome = list
        self.fitness = -1

    #获取随机的一个解
    def get_random_permutation(self):
        nums = list(range(1, self.chromosome_length + 1))
        for i in range(self.chromosome_length - 1):
            j = random.randint(i, self.chromosome_length - 1)
            nums[i], nums[j] = nums[j], nums[i]
        return nums

    def calculate_fitness(self,fitness_function,Matrix_D,Matrix_F):
        self.fitness = fitness_function(Matrix_D,Matrix_F,self.chromosome)

class EvolutionaryAlgorithm:
    def __init__(self, population_size,fitness_function, mutation_rate, crossover_rate,data_filename):
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.data_filename = data_filename
        self.ReadFile()

        self.population.append(Individual(5,[1,2,3,4,5]))
        self.insert_mutation()
        print(self.population[0].chromosome)

    def ReadFile(self):
        # 从文件中读取数据
        with open(self.data_filename, 'r') as file:
            n = int(file.readline().strip())  # 读取数字n
            self.chromosome_length = n
            # print(n)
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

        self.Matrix_F = A
        self.Matrix_D = B
        # # 打印矩阵A和B
        # print("Matrix A:")
        # print(A)
        # print("\nMatrix B:")
        # print(B)

    def evaluate_population(self):
        for individual in self.population:
            individual.calculate_fitness(self.fitness_function,self.Matrix_D,self.Matrix_F)

    def scramble_mutation(self): #争夺变异，中间随机排序
        for individual in self.population:
            if random.random() < self.mutation_rate: #处理变异率
                # 随机选择一段子串
                a = random.randint(0, individual.chromosome_length-1)
                b = random.randint(0, individual.chromosome_length) #结束点可以往后挪，因为[start:end]是左闭右开的
                start = min(a,b)
                end = max(a,b)
                segment = individual.chromosome[start:end]
                # 随机重新排序子串
                random.shuffle(segment)
                # 更新个体的染色体
                individual.chromosome[start:end] = segment


    def insert_mutation(self): #插入变异，保留大部分的邻接关系，但破坏序关系
        for individual in self.population:
            if random.random() < self.mutation_rate: #处理变异率
                # 随机选取两个基因的索引
                index1 = random.randint(0, individual.chromosome_length - 1)
                index2 = random.randint(0, individual.chromosome_length - 1)
                #插入
                print(index1,index2)
                gene = individual.chromosome.pop(index2)
                individual.chromosome.insert(index1 + 1, gene)


#计算总运输成本
def getcost(D,F,perm):
    n = len(perm)
    total_cost = 0
    for i in range(0,n):
        for j in range(n):
            factory_i = perm[i] -1
            factory_j = perm[j] -1
            distance = D[factory_i][factory_j]
            cost = F[i][j] * distance
            total_cost += cost
    # print('totalcost=',total_cost)
    return total_cost

if __name__ == '__main__':
    datafile = "./qapdata/nug12.dat"
    EvolutionaryAlgorithm(population_size=100,fitness_function=getcost,mutation_rate=1,crossover_rate=0.1,data_filename=datafile)
