import random
import math

# 定義函數來計算兩個點之間的距離
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 定義函數來計算給定路徑的總長度
def path_length(path, points):
    total_length = 0
    for i in range(len(path) - 1):
        total_length += distance(points[path[i]], points[path[i+1]])
    return total_length

# 定義函數來解決 TSP 問題
def tsp(points, max_iterations, initial_temperature, cooling_rate):
    # 隨機初始化路徑
    path = list(range(len(points)))
    random.shuffle(path)

    # 計算初始溫度
    current_length = path_length(path, points)
    temperature = initial_temperature
    best_path = path
    best_length = current_length

    # 迭代
    for i in range(max_iterations):
        # 遍遍所有交換相鄰點的可能性，並選擇導致路徑長度減小的那個
        j = random.randint(0, len(points) - 2)
        new_path = path.copy()
        new_path[j], new_path[j+1] = new_path[j+1], new_path[j]
        new_length = path_length(new_path, points)

        # 根據 Metropolis 準則決定是否接受更差的解
        delta = new_length - current_length
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            path = new_path
            current_length = new_length

        # 如果當前解是最好的，則更新最好的解
        if current_length < best_length:
            best_path = path
            best_length = current_length

        # 降温
        temperature *= cooling_rate

    return best_path, best_length

# 測試代碼
points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
path, length = tsp(points, 10000, 100, 0.99)
print("Path:", path)
print("Length:", length)