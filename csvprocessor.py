import csv
import statistics
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
def calculate_statistics(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 获取CSV文件的标题行
        data = []  # 存储数据行的列表
        problems = []
        for row in reader:

            problems.append(row[0])
            data.append([int(value) for value in row[1:]])  # 将每行的数据转换为整数列表
            # print('row=', data[-1])
            # for a in data[-1]:
            #     print(type(a))

        print('{:<10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
            'Problem ', 'Best', 'Worst', 'Mean', 'Median', 'Std'))
        result = []  # 存储统计结果的列表
        result.append(['Problem', 'Min', 'Max', 'Mean', 'Median', 'Std'])
        for i, row in enumerate(data):
            problem = problems[i]
            values = row[:]
            # print('valuesi=',values)
            minimum = min(values)
            maximum = max(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            std_dev = statistics.stdev(values)

            result.append([problem, minimum, maximum, mean, median, std_dev])
            print('{:<10s} {:10d} {:10d} {:10.2f} {:10.2f} {:10.2f}'.format(
                problem, minimum, maximum, mean, median, std_dev))

    output_filename = filename.replace('.csv', '_analysis.csv')
    output_filename = output_filename.replace('experimental results','result analysis')
    with open(output_filename, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(result)


# 使用示例
if __name__ == '__main__':
    filenames = ['./experimental results/order_scramble_results.csv','./experimental results/insert_cycle_results.csv','./experimental results/tabu_order_scramble_results.csv']
    for filename in filenames:
        calculate_statistics(filename)
