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

    def calculate_fitness(self, fitness_function, Matrix_D, Matrix_F):
        self.fitness = fitness_function(Matrix_D, Matrix_F, self.chromosome)


class EvolutionaryAlgorithm:

    def __init__(self, population_size, fitness_function, mutation_rate,
                 crossover_rate, data_filename):
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.data_filename = data_filename
        self.ReadFile()

        self.population.append(Individual(9, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.population.append(Individual(9, [9, 3, 7, 8, 2, 6, 5, 1, 4]))
        # self.insert_mutation()
        # self.order_crossover()
        self.cycle_crossover()
        print(self.population[0].chromosome)
        print(self.population[1].chromosome)

    def ReadFile(self):
        # 从文件中读取数据
        with open(self.data_filename, 'r') as file:
            n = int(file.readline().strip())  # 读取数字n
            self.chromosome_length = n
            # print(n)
            # 读取矩阵A
            rows_A = []
            i = 0
            while i < n:
                row = list(map(int, file.readline().strip().split()))
                if len(row) == 0:
                    continue
                row = np.array(row)
                rows_A.append(row)
                i += 1
            A = np.array(rows_A, dtype=object)

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
            B = np.array(rows_B, dtype=object)

        self.Matrix_F = A
        self.Matrix_D = B
        # # 打印矩阵A和B
        # print("Matrix A:")
        # print(A)
        # print("\nMatrix B:")
        # print(B)

    def evaluate_population(self):
        for individual in self.population:
            individual.calculate_fitness(self.fitness_function, self.Matrix_D,
                                         self.Matrix_F)

    def scramble_mutation(self):  #争夺变异，中间随机排序
        for individual in self.population:
            if random.random() < self.mutation_rate:  #处理变异率
                # 随机选择一段子串
                a = random.randint(0, individual.chromosome_length - 1)
                b = random.randint(0, individual.chromosome_length
                                   )  #结束点可以往后挪，因为[start:end]是左闭右开的
                start = min(a, b)
                end = max(a, b)
                segment = individual.chromosome[start:end]
                # 随机重新排序子串
                random.shuffle(segment)
                # 更新个体的染色体
                individual.chromosome[start:end] = segment

    def insert_mutation(self):  #插入变异，保留大部分的邻接关系，但破坏序关系
        for individual in self.population:
            if random.random() < self.mutation_rate:  #处理变异率
                # 随机选取两个基因的索引
                index1 = random.randint(0, individual.chromosome_length - 1)
                index2 = random.randint(0, individual.chromosome_length - 1)
                #插入
                # print(index1,index2)
                gene = individual.chromosome.pop(index2)
                individual.chromosome.insert(index1 + 1, gene)

    def order_crossover(self):  #序交叉
        for i in range(0, self.population_size - 1, 2):
            if random.random() < self.crossover_rate:
                # 选择相邻个体交叉，随机选择两个交叉点
                parent1 = self.population[i]
                parent2 = self.population[i + 1]

                point1 = random.randint(0, parent1.chromosome_length - 1)
                point2 = random.randint(0, parent1.chromosome_length - 1)
                # 确定交叉点的较小和较大值
                start = min(point1, point2)
                end = max(point1, point2)
                # 创建子代
                offspring1 = [None] * parent1.chromosome_length
                offspring2 = [None] * parent1.chromosome_length
                # 将父代的交叉片段复制到子代的相同位置
                offspring1[start:end + 1] = parent1.chromosome[start:end + 1]
                offspring2[start:end + 1] = parent2.chromosome[start:end + 1]
                # 从另一个父代的第二个交叉点开始填充子代
                pointer_parent = end + 1
                pointer_offspring = end + 1
                while None in offspring1:
                    if parent2.chromosome[pointer_parent] not in offspring1:
                        offspring1[pointer_offspring] = parent2.chromosome[
                            pointer_parent]
                        pointer_offspring = (pointer_offspring +
                                             1) % parent1.chromosome_length
                    pointer_parent = (pointer_parent +
                                      1) % parent1.chromosome_length

                # 填充第二个子代
                pointer_parent = end + 1
                pointer_offspring = end + 1
                while None in offspring2:
                    if parent1.chromosome[pointer_parent] not in offspring2:
                        offspring2[pointer_offspring] = parent1.chromosome[
                            pointer_parent]
                        pointer_offspring = (pointer_offspring +
                                             1) % parent1.chromosome_length
                    pointer_parent = (pointer_parent +
                                      1) % parent1.chromosome_length

                # 创建子代个体并替换父代个体
                self.population[i] = Individual(parent1.chromosome_length,
                                                offspring1)
                self.population[i + 1] = Individual(parent1.chromosome_length,
                                                    offspring2)

    def cycle_crossover(self):  #圈交叉
        for i in range(0, self.population_size - 1, 2):
            if random.random() < self.crossover_rate:
                # 选择相邻个体交叉
                parent1 = self.population[i]
                parent2 = self.population[i + 1]
                n = parent1.chromosome_length
                # 创建子代
                offspring1 = [None] * n
                offspring2 = [None] * n
                circle_flag1 = [None] * n  #代表该元素属于第几个圈
                circle_flag2 = [None] * n
                circle_idx = 1
                start = 0  #起始位置
                circle_flag1[start] = circle_idx
                cur = start
                # CX算法核心部分
                while None in circle_flag1:
                    circle_flag2[cur] = circle_idx
                    cur = parent1.chromosome.index(parent2.chromosome[cur]) #在p1 找 p2的value
                    circle_flag1[cur] = circle_idx
                    if(cur == start):
                        circle_idx += 1
                        start = circle_flag1.index(None)
                        cur = start
                        circle_flag1[start] = circle_idx
                circle_flag2[cur] = circle_idx
                # print(f'circle_flag1 = {circle_flag1}, circle_flag2 = {circle_flag2}')
                #交替选择圈，构造子代
                flag = 1 #选择第一个父代的元素

                for z in range(1,circle_idx+1): #i是圈号，表示当前要复制第i个圈的解
                    for j in range(0,n):        #j表示当前元素下标
                        if circle_flag1[j] == z: 
                            if flag == 1:
                                offspring1[j] = parent1.chromosome[j]
                                offspring2[j] = parent2.chromosome[j]
                            else:
                                offspring1[j] = parent2.chromosome[j]
                                offspring2[j] = parent1.chromosome[j]    
                    flag = not flag

                # 创建子代个体并替换父代个体
                self.population[i] = Individual(parent1.chromosome_length,offspring1)
                self.population[i + 1] = Individual(parent1.chromosome_length,offspring2)

                # print(f'child1 = {offspring1}, child2 = {offspring2}')


#计算总运输成本
def getcost(D, F, perm):
    n = len(perm)
    total_cost = 0
    for i in range(0, n):
        for j in range(n):
            factory_i = perm[i] - 1
            factory_j = perm[j] - 1
            distance = D[factory_i][factory_j]
            cost = F[i][j] * distance
            total_cost += cost
    # print('totalcost=',total_cost)
    return total_cost


if __name__ == '__main__':
    datafile = "./qapdata/nug12.dat"
    EvolutionaryAlgorithm(population_size=2,
                          fitness_function=getcost,
                          mutation_rate=1,
                          crossover_rate=1,
                          data_filename=datafile)
