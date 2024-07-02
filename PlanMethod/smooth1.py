import csv
import numpy as np
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def smooth_path(path, num_points=100):
    """
    Smooths a path using cubic spline interpolation.
    Arguments:
        path: List containing coordinates of a path
        num_points: Number of points to interpolate between each pair of original points.
    Output:
        smoothed_path: List containing smoothed coordinates.
    """
    # Extract x and y coordinates from the path
    x = [point[0] for point in path]
    y = [point[1] for point in path]

    # Perform cubic spline interpolation
    spline = CubicSpline(range(len(path)), np.array([x, y]).T, axis=0)
    smoothed_path = spline(np.linspace(0, len(path) - 1, num_points))
    
    return smoothed_path.tolist()

class Map():
    def __init__(self, width, height, map):
        self.width = width
        self.height = height
        self.map = map
    
    def createBlock(self, block_num):
        for i in range(block_num):
            x, y = (randint(0, self.width-1), randint(0, self.height-1))
            self.map[y][x] = 1
    
    def generatePos(self, rangeX, rangeY):
        x, y = (randint(rangeX[0], rangeX[1]), randint(rangeY[0], rangeY[1]))
        while self.map[y][x] == 1:
            x, y = (randint(rangeX[0], rangeX[1]), randint(rangeY[0], rangeY[1]))
        return (x , y)

    def showMap(self):
        # print("+" * (3 * self.width + 2))
        for row in self.map:
            s = '+'
            for entry in row:
                s += ' ' + str(entry) + ' '
            s += '+'
            # print(s)
        # print("+" * (3 * self.width + 2))

def visualize_path(map, path):
    map_array = np.array(map.map)
    plt.imshow(map_array, cmap='gray', origin='upper')  # 设置origin参数为'upper'

    if path is not None:
        path_array = np.array(path)
        plt.plot(path_array[:, 0], path_array[:, 1], marker='.', color='red')

    plt.title('Path Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def read_path_from_csv(filename):
    path = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            x, y = map(float, row)  # 将字符串转换为浮点数
            path.append((x, y))
    return path

def read_csv_to_maze(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        maze = []
        for row in reader:
            maze.append([int(cell) for cell in row])
    return np.array(maze)

if __name__ == "__main__":
    # Example usage
    WIDTH = 512
    HEIGHT = 512
    BLOCK_NUM = 15

    maps = read_csv_to_maze("origin.csv")
    map123 = Map(WIDTH, HEIGHT, maps)
    path = read_path_from_csv('path0.csv')
    aim = [path[i] for i in range(0, len(path) - 90, 60)]
    path1 = smooth_path(np.array(aim))
    visualize_path(map123, path1)

    # path1 = smooth_path(np.array(path))
    # visualize_path(map123, path1)
