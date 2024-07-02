from random import randint
import numpy as np
import csv
import matplotlib.pyplot as plt

def visualize_path(map, path):
    map_array = np.array(map.map)
    plt.imshow(map_array, cmap='gray', origin='upper')  # 设置origin参数为'upper'

    if path is not None:
        path_array = np.array(path)
        plt.plot(path_array[:, 0], path_array[:, 1], marker='o', color='red')

    plt.title('Path Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    # plt.savefig("123.jpg")  
    plt.show()

class SearchEntry():
    def __init__(self, x, y, g_cost, f_cost=0, pre_entry=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.pre_entry = pre_entry
    
    def getPos(self):
        return (self.x, self.y)

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

def AStarSearch(map, source, dest):
    def getNewPosition(map, location, offset):
        x, y = (location.x + offset[0], location.y + offset[1])
        if x < 0 or x >= map.width or y < 0 or y >= map.height or map.map[y][x] == 1:
            return None
        return (x, y)

    def getPositions(map, location):
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        poslist = []
        for offset in offsets:
            pos = getNewPosition(map, location, offset)
            if pos is not None:
                poslist.append(pos)
        return poslist

    def calHeuristic(pos, dest):
        return abs(dest.x - pos[0]) + abs(dest.y - pos[1])

    def getMoveCost(location, pos):
        if location.x != pos[0] and location.y != pos[1]:
            return 1.4
        else:
            return 1

    def isInList(list, pos):
        return pos in list

    def addAdjacentPositions(map, location, dest, openlist, closedlist):
        poslist = getPositions(map, location)
        for pos in poslist:
            if not isInList(closedlist, pos):
                findEntry = openlist.get(pos)
                h_cost = calHeuristic(pos, dest)
                g_cost = location.g_cost + getMoveCost(location, pos)
                if findEntry is None:
                    openlist[pos] = SearchEntry(pos[0], pos[1], g_cost, g_cost+h_cost, location)
                elif findEntry.g_cost > g_cost:
                    findEntry.g_cost = g_cost
                    findEntry.f_cost = g_cost + h_cost
                    findEntry.pre_entry = location

    def getFastPosition(openlist):
        fast = None
        for entry in openlist.values():
            if fast is None or fast.f_cost > entry.f_cost:
                fast = entry
        return fast

    openlist = {}
    closedlist = {}
    location = SearchEntry(source[0], source[1], 0.0)
    dest = SearchEntry(dest[0], dest[1], 0.0)
    openlist[source] = location
    while True:
        location = getFastPosition(openlist)
        if location is None:
            print("Can't find a valid path")
            break

        if location.x == dest.x and location.y == dest.y:
            break

        closedlist[location.getPos()] = location
        openlist.pop(location.getPos())
        addAdjacentPositions(map, location, dest, openlist, closedlist)

    if location is not None:
        path = []
        while location is not None:
            path.append([location.x, location.y])
            location = location.pre_entry
        path.reverse()
        return path
    else:
        return None

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
    maps = read_csv_to_maze("temp1.csv")
    map = Map(WIDTH, HEIGHT, maps)
    # map.createBlock(BLOCK_NUM)
    source = (159, 135)
    dest = (400, 400)
    path = AStarSearch(map, source, dest)

    map.showMap()
    if path is not None:
        # print("Found path:")
        # for point in path:
        #     print(point)
        visualize_path(map, path)
    else:
        print("No valid path found.")
