import csv
import numpy as np

def calculate_orientation(front_point, rear_point):
    """
    计算两点之间的方向角度。

    Args:
    - front_point: 前置点的坐标，格式为 (x, y)
    - rear_point: 后置点的坐标，格式为 (x, y)

    Returns:
    - orientation: 两点连线相对于水平轴的角度，范围为 [-π, π]
    """
    delta_x = rear_point[0] - front_point[0]
    delta_y = rear_point[1] - front_point[1]
    orientation = np.arctan2(delta_y, delta_x)
    return orientation

def write_path_to_csv(path, filename):
    """
    将路径点写入CSV文件。

    Args:
    - path: 包含点坐标的列表，每个点的格式为 (x, y)
    - filename: 要写入的CSV文件名

    Returns:
    - None
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point in path:
            writer.writerow(point)

def near(car_center, target_center):
    (car_Xcenter, car_Ycenter) = car_center
    (target_Xcenter, target_Ycenter) = target_center
    distance = np.sqrt((car_Xcenter-target_Xcenter)**2+(car_Ycenter-target_Ycenter)**2)
    
    if distance > 2.8:
        return False
    else:
        return True

def save_map_to_file(binary_map, filename):
    with open(filename, 'w') as file:
        for row in binary_map:
            file.write(','.join(map(str, row)) + '\n')