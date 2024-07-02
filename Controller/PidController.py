import math
import numpy as np

class PidControl:
    def __init__(self, Kp_linear, Ki_linear, Kp_angular, Ki_angular):
        # PI控制参数
        self.Kp_linear = Kp_linear
        self.Ki_linear = Ki_linear
        self.Kp_angular = Kp_angular
        self.Ki_angular = Ki_angular
        
        # 初始化其他参数
        self.prev_error_linear = 0 # 记录线性误差
        self.prev_error_angular = 0 # 记录角度误差
        self.integral_linear = []
        self.integral_angular = []
        
        self.L = 0.2  # 轮子间距
        self.R = 0.04  # 轮子半径
        self.vel_L = 0.0 # 记录左轮速度
        self.vel_R = 0.0 # 记录右轮速度
        self.tracker = None # 初始化跟踪器
        
    def control_bubble(self, current_position, target_position, red_position, blue_position):
        error_linear = self.distance_error(target_position, current_position)
        error_angular = self.angle_error(red_position, blue_position, target_position)

        # 限制误差防止抖动 0.1
        if error_linear < 0.1:
            error_linear = 0
        if abs(error_angular) < 0.1:
            error_angular = 0

        # 计算线速度控制量
        if len(self.integral_linear) < 30:
            self.integral_linear.append(error_linear)
        else:
            self.integral_linear.pop(0)
            self.integral_linear.append(error_linear)
        integral_linear_sum = sum(self.integral_linear)
        linear_velocity = self.Kp_linear * error_linear + self.Ki_linear * integral_linear_sum

        # 计算角速度控制量
        if len(self.integral_angular) < 30:
            self.integral_angular.append(error_angular)
        else:
            self.integral_angular.pop(0)
            self.integral_angular.append(error_angular)
        integral_angular_sum = sum(self.integral_angular)
        angular_velocity = self.Kp_angular * error_angular + self.Ki_angular * integral_angular_sum

        # 计算左右轮速度
        vl = linear_velocity + self.L * 0.5 * angular_velocity + 0.8266
        vr = linear_velocity - self.L * 0.5 * angular_velocity + 0.8266  
        
        return vl, vr

    def distance_error(self, position1, position2):
        """
        计算两个向量之间的夹角，单位为度。

        Args:
        - position1: 位置点1
        - position2: 位置点2

        Returns:
        - distance: 两点之间的距离
        """
        return math.hypot(position2[0] - position1[0], position2[1] - position1[1])  

    def angle_error(self, red_position, blue_position, green_position):
        """
        计算小车的夹角，单位为度。

        Args:
        - red_position: 红色点的坐标，格式为 (x, y)
        - blue_position: 蓝色点的坐标，格式为 (x, y)
        - green_position: 绿色点的坐标，格式为 (x, y)

        Returns:
        - angle_degrees: 小车运动方向的向量与车头到终点之间的夹角，单位为度
        """
        # 计算蓝色到红色的向量
        BR_vector = np.array([red_position[0] - blue_position[0], red_position[1] - blue_position[1]])
        # 计算红色到绿色的向量
        RG_vector = np.array([green_position[0] - red_position[0], green_position[1] - red_position[1]])

        # 计算两个向量的点积
        dot_product = np.dot(BR_vector, RG_vector)
        # 计算两个向量的叉积
        cross_product = np.cross(BR_vector, RG_vector)

        # 计算夹角的弧度
        angle_radians = np.arctan2(cross_product, dot_product)
        # 将弧度转换为角度
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees