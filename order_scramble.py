import random
import numpy as np
import os
import copy
import csv
import math

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


class Individual:

    def __init__(self, chromosome_length, list=None):
        if list == None:
            self.chromosome_length = chromosome_length
            self.chromosome = self.get_random_permutation()
            self.fitness = -1
        else:
            self.chromosome_length = chromosome_length
            self.chromosome = list
            self.fitness = -1

    # 获取随机的一个解
    def get_random_permutation(self):
        nums = list(range(1, self.chromosome_length + 1))
        for i in range(self.chromosome_length - 1):
            j = random.randint(i, self.chromosome_length - 1)
            nums[i], nums[j] = nums[j], nums[i]
        return nums

    def calculate_fitness(self, fitness_function, Matrix_D, Matrix_F):
        self.fitness = fitness_function(Matrix_D, Matrix_F, self.chromosome)
        return self.fitness


class EvolutionaryAlgorithm:

    def __init__(self, population_size, fitness_function, mutation_rate,
                 crossover_rate, data_filename):
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.newPop = []
        self.best_individual = None
        self.num_evaluations = 0
        self.data_filename = data_filename
        self.history = []  # 存储历史数据
        self.ReadFile()
        print(f'Solving problem in {data_filename}. self.chromosome_length={self.chromosome_length}')

    def run(self, MAX_EVALUATIONS):
        self.initialize_population()
        self.MAX_EVALUATIONS = MAX_EVALUATIONS
        self.evaluate_population()

        # # 找到初始化后，适应度值最低的个体，对其做禁忌搜索，然后在原种群中将其替换掉
        # min_fitness = float('inf')
        # min_individual = None
        # min_index = -1
        # for index, individual in enumerate(self.population):
        #     fitness = individual.fitness
        #     if fitness < min_fitness:
        #         min_fitness = fitness
        #         min_individual = individual
        #         min_index = index
        # print(f"Before tabu, best fitness = {min_fitness}")
        # self.tabu_search(min_individual, 10)
        # self.population[min_index] = self.best_individual
        print(f"self.best_individual.fitness={self.best_individual.fitness}")
        if self.num_evaluations >= self.MAX_EVALUATIONS:
            print(
                f'bestchrome={self.best_individual.chromosome},fitness={self.best_individual.fitness},FE={self.num_evaluations}')
            return
        while True:
            # self.getBest()
            self.newPop = [self.best_individual]  # 精英保留
            self.tournament_selection(2)  # 父代选择 2-锦标赛
            self.order_crossover()  # 交叉之后，newpop的数量为原数量+1（保留了1个精英）
            # self.cycle_crossover()
            # self.insert_mutation()
            self.scramble_mutation()
            self.evaluate_newPop()  # 每一轮只做一次函数评估，在这里
            if self.num_evaluations >= self.MAX_EVALUATIONS:
                break
            self.Substitude()
            # print(
            #     f'generation={i},bestchrome={self.population[0].chromosome},fitness={self.population[0].fitness}'
            # )
        # bestin = self.getBest()
        print(
            f'bestchrome={self.best_individual.chromosome},fitness={self.best_individual.fitness},FE={self.num_evaluations}')

    def tabu_search(self, individual, num_iterations):  # 禁忌搜索
        tabu_list = []
        current_individual = individual
        best_individual = individual

        for i in range(num_iterations):
            neighbors = self.generate_neighbors(current_individual)
            feasible_neighbors = [
                neighbor for neighbor in neighbors
                if neighbor.chromosome not in tabu_list
            ]

            # 无可行邻域，提前结束搜索
            if not feasible_neighbors:
                break
            # 计算适应度
            for individual in feasible_neighbors:
                self.evaluate_fitness(individual)
                if self.num_evaluations >= self.MAX_EVALUATIONS:
                    print(f"tabu:Reaching maximum FE {self.num_evaluations}")
                    break

            if self.num_evaluations >= self.MAX_EVALUATIONS:
                break

            tabu_list.append(current_individual.chromosome)
            # 更新禁忌列表，保持长度不超过邻域的大小
            if len(tabu_list) > len(neighbors):
                tabu_list.pop(0)
            # print(
            #     f'tabu:iteration={i},best_ind={self.best_individual.chromosome},best_fitness={self.best_individual.fitness}'
            # )
        print(
            f'tabu:best_ind={self.best_individual.chromosome},best_fitness={self.best_individual.fitness}'
        )

    def generate_neighbors(self, individual):
        # 遍历所有可能的交换一次的情况，作为当前个体的邻居
        neighbors = []
        for i in range(self.chromosome_length - 1):
            for j in range(i + 1, self.chromosome_length):
                neighbor = self.swap_positions(individual, i, j)
                neighbors.append(neighbor)
        return neighbors

    def swap_positions(self, individual, i, j):
        permutation = copy.deepcopy(individual.chromosome[:])
        permutation[i], permutation[j] = permutation[j], permutation[i]
        return Individual(chromosome_length=self.chromosome_length,
                          list=permutation)

    def tournament_selection(self, tournament_size):
        selected_population = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            tournament_fitness = [individual.fitness for individual in tournament]
            winner_index = tournament_fitness.index(min(tournament_fitness))
            selected_population.append(tournament[winner_index])

        self.population = selected_population

    def Substitude(self):  # 去掉newPop中适应度最差的解
        self.newPop.sort(key=lambda individual: individual.fitness)
        self.population = self.newPop[:self.population_size]

    def printPop(self):
        for i in range(0, self.population_size):
            print(
                f'i={i},chromosome={self.population[i].chromosome},fitness={self.population[i].fitness}'
            )

    def printNewPop(self):
        for i in range(0, len(self.newPop)):
            print(
                f'newpop: i={i},chromosome={self.newPop[i].chromosome},fitness={self.newPop[i].fitness}'
            )

    def initialize_population(self):
        self.population = [
            Individual(chromosome_length=self.chromosome_length)
            for _ in range(self.population_size)
        ]

    # def getBest(self):
    #     #获取种群中适应度值最低的个体
    #     self.best_individual = min(self.population,
    #                                key=lambda individual: individual.fitness)
    #     return self.best_individual

    def AddIndidual(self, perm):
        self.population.append(Individual(len(perm), perm))
        self.population_size += 1

    def ReadFile(self):
        # 从文件中读取数据
        with open(self.data_filename, 'r') as file:
            n = int(file.readline().strip())  # 读取数字n
            self.chromosome_length = n
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
        # 打印矩阵A和B
        # print("Matrix A:")
        # print(A)
        # print("\nMatrix B:")
        # print(B)

    # 函数评估，同时更新最优解
    def evaluate_fitness(self, individual):
        fitness = individual.calculate_fitness(self.fitness_function, self.Matrix_D, self.Matrix_F)
        self.num_evaluations += 1
        # 更新最优解
        if (self.best_individual is None) or (fitness < self.best_individual.fitness):
            self.best_individual = individual
        # 每隔1000次函数评估打印最优解
        if self.num_evaluations % 100 == 0:
            self.history.append(self.best_individual.fitness)
        if self.num_evaluations % 1000 == 0:
            print(f"FE:{self.num_evaluations}, Current best fitness:{self.best_individual.fitness}")
        return fitness

    def evaluate_population(self):
        for individual in self.population:
            self.evaluate_fitness(individual)
            if self.num_evaluations >= self.MAX_EVALUATIONS:
                print(f"Reaching maximum FE {self.num_evaluations}")
                break

    def evaluate_newPop(self):
        for individual in self.newPop:
            self.evaluate_fitness(individual)
            if self.num_evaluations >= self.MAX_EVALUATIONS:
                print(f"Reaching maximum FE {self.num_evaluations}")
                break

    def scramble_mutation(self):  # 争夺变异，中间随机排序
        for individual in self.newPop:
            if individual == self.best_individual:  # 精英保留
                continue
            if random.random() < self.mutation_rate:  # 处理变异率
                # 随机选择一段子串
                a = random.randint(0, individual.chromosome_length - 1)
                b = random.randint(0, individual.chromosome_length
                                   )  # 结束点可以往后挪，因为[start:end]是左闭右开的
                start = min(a, b)
                end = max(a, b)
                segment = individual.chromosome[start:end]
                # 随机重新排序子串
                random.shuffle(segment)
                # 更新个体的染色体
                individual.chromosome[start:end] = segment

    def insert_mutation(self):  # 插入变异，保留大部分的邻接关系，但破坏序关系
        for individual in self.newPop:
            if individual == self.best_individual:  # 精英保留
                continue
            if random.random() < self.mutation_rate:  # 处理变异率
                # 随机选取两个基因的索引
                index1 = random.randint(0, individual.chromosome_length - 1)
                index2 = random.randint(0, individual.chromosome_length - 1)
                # 插入
                # print(index1,index2)
                gene = individual.chromosome.pop(index2)
                individual.chromosome.insert(index1 + 1, gene)

    def order_crossover(self):  # 序交叉
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
                pointer_parent = (end + 1) % parent1.chromosome_length
                pointer_offspring = (end + 1) % parent1.chromosome_length
                while None in offspring1:
                    if parent2.chromosome[pointer_parent] not in offspring1:
                        offspring1[pointer_offspring] = parent2.chromosome[
                            pointer_parent]
                        pointer_offspring = (pointer_offspring +
                                             1) % parent1.chromosome_length
                    pointer_parent = (pointer_parent +
                                      1) % parent1.chromosome_length

                # 填充第二个子代
                pointer_parent = (end + 1) % parent1.chromosome_length
                pointer_offspring = (end + 1) % parent1.chromosome_length
                while None in offspring2:
                    if parent1.chromosome[pointer_parent] not in offspring2:
                        offspring2[pointer_offspring] = parent1.chromosome[
                            pointer_parent]
                        pointer_offspring = (pointer_offspring +
                                             1) % parent1.chromosome_length
                    pointer_parent = (pointer_parent +
                                      1) % parent1.chromosome_length

                # 更新子代
                self.newPop.append(
                    Individual(parent1.chromosome_length, offspring1))
                self.newPop.append(
                    Individual(parent1.chromosome_length, offspring2))

    def cycle_crossover(self):  # 圈交叉
        for i in range(0, self.population_size - 1, 2):
            if random.random() < self.crossover_rate:
                # 选择相邻个体交叉
                parent1 = self.population[i]
                parent2 = self.population[i + 1]
                n = parent1.chromosome_length
                # 创建子代
                offspring1 = [None] * n
                offspring2 = [None] * n
                circle_flag1 = [None] * n  # 代表该元素属于第几个圈
                circle_flag2 = [None] * n
                circle_idx = 1
                start = 0  # 起始位置
                circle_flag1[start] = circle_idx
                cur = start
                # CX算法核心部分
                while None in circle_flag1:
                    circle_flag2[cur] = circle_idx
                    cur = parent1.chromosome.index(
                        parent2.chromosome[cur])  # 在p1 找 p2的value
                    circle_flag1[cur] = circle_idx
                    if (cur == start):
                        circle_idx += 1
                        start = circle_flag1.index(None)
                        cur = start
                        circle_flag1[start] = circle_idx
                circle_flag2[cur] = circle_idx
                # print(f'circle_flag1 = {circle_flag1}, circle_flag2 = {circle_flag2}')
                # 交替选择圈，构造子代
                flag = 1  # 选择第一个父代的元素

                for z in range(1, circle_idx + 1):  # i是圈号，表示当前要复制第i个圈的解
                    for j in range(0, n):  # j表示当前元素下标
                        if circle_flag1[j] == z:
                            if flag == 1:
                                offspring1[j] = parent1.chromosome[j]
                                offspring2[j] = parent2.chromosome[j]
                            else:
                                offspring1[j] = parent2.chromosome[j]
                                offspring2[j] = parent1.chromosome[j]
                    flag = not flag

                # 更新子代
                self.newPop.append(
                    Individual(parent1.chromosome_length, offspring1))
                self.newPop.append(
                    Individual(parent1.chromosome_length, offspring2))

                # print(f'child1 = {offspring1}, child2 = {offspring2}')


# 计算总运输成本
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
    return total_cost


def write_solutions_to_csv(problem, solutions, filename):
    # 确定列头
    header = ['Problem'
              ] + ['Run {}'.format(i + 1) for i in range(len(solutions))]

    # 检查文件是否存在
    file_exists = os.path.isfile(filename)

    # 打开CSV文件进行写入
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        # 如果文件不存在，写入列头
        if not file_exists:
            writer.writerow(header)

        # 写入解
        writer.writerow([problem] + solutions)
    print(f'Results have been saved to {filename}')


def write_history_to_csv(history, filename):
    # 确定列头
    header = ['FE'] + ['Run {}'.format(i + 1) for i in range(len(history))]
    # 打开CSV文件进行写入
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # 写入列头
        writer.writerow(header)

        # 写入历史数据
        max_length = max(len(run) for run in history)
        for i in range(max_length):
            row = [(i + 1) * 100] + [run[i] if i < len(run) else '' for run in history]
            writer.writerow(row)

    print(f'History has been saved to {filename}')


if __name__ == '__main__':
    datafiles = [
        "./qapdata/tai15a.dat",
        "./qapdata/tai30a.dat",
        "./qapdata/tai60a.dat",
        "./qapdata/tai80a.dat",
    ]
    for datafile in datafiles:
        problem = datafile.split("/")[2].split(".")[0]
        results = []

        filename = f'./experimental results/order_scramble_results.csv'

        history_filename = f'./experimental results/order_scamble_{problem}_history.csv'
        # os.makedirs(folder_name, exist_ok=True)  # 创建结果文件夹，如果不存在则创建

        history = []  # 存储每个run的历史数据
        num_runs = 25
        for i in range(0, num_runs):
            algorithm = EvolutionaryAlgorithm(population_size=50,
                                              fitness_function=getcost,
                                              mutation_rate=0.05,
                                              crossover_rate=0.8,
                                              data_filename=datafile)
            algorithm.run(MAX_EVALUATIONS=100000)
            results.append(algorithm.best_individual.fitness)
            history.append(algorithm.history)

        write_history_to_csv(history=history, filename=history_filename)

        # print(results)

        write_solutions_to_csv(problem=problem,
                               solutions=results,
                               filename=filename)
    print("All experiments has been done!")
