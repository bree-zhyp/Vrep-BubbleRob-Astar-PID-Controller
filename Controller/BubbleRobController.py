import time
import sys
import numpy as np
import cv2
import PlanMethod.FoundNextPoint as fd

import Controller.PidController as pid
import Controller.FunctionTool as FT
import PlanMethod.Astar as As
import PlanMethod.smooth as sm
import PlanMethod.smooth1 as sm1

try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')
            
class BubbleRobController:
    def __init__(self):
        # PI控制参数
        self.Kp_linear = 0.000
        self.Ki_linear = 0.000
        self.Kp_angular = 0.03
        self.Ki_angular = 0.00033

        self.L = 0.2  # 轮子间距
        self.R = 0.04  # 轮子半径
        self.vel_L = 0.0 # 记录左轮速度
        self.vel_R = 0.0 # 记录右轮速度
        self.tracker = None # 初始化跟踪器

    def control_bubble(self, current_position, target_position, red_position, blue_position):
        """
        控制机器人移动到目标位置的方法。

        Args:
        - current_position: 当前机器人的位置，格式为 (x, y)
        - target_position: 目标位置，机器人要移动到的位置，格式为 (x, y)
        - red_position: 红色点的位置，用于计算控制器的目标方向，格式为 (x, y)
        - blue_position: 蓝色点的位置，用于计算控制器的目标方向，格式为 (x, y)

        Returns:
        - vl: 左轮的目标速度
        - vr: 右轮的目标速度
        """
        # 创建PID控制器实例，对速度进行规划
        PIDController = pid.PidControl(self.Kp_linear, self.Ki_linear, self.Kp_angular, self.Ki_angular)
        vl, vr = PIDController.control_bubble(current_position, target_position, red_position, blue_position)

        # 记录左右轮的速度
        self.vel_L = vl
        self.vel_R = vr

        # 设置左右轮速度
        sim.simxSetJointTargetVelocity(self.clientID, self.left_jointHandle, vl / self.R, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID, self.right_jointHandle, vr / self.R, sim.simx_opmode_streaming)

    def stop_bubble(self):
        """
        停止机器人的运动，将左右轮速度设为0。
        """
        # 设置左右轮速度为0
        sim.simxSetJointTargetVelocity(self.clientID, self.left_jointHandle, 0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID, self.left_jointHandle, 0, sim.simx_opmode_streaming)      

    def connect_to_simulator(self):
        """
        连接到仿真器的方法，并设置同步模式。
        """
        # 尝试关闭所有的已开启的连接，开始运行program
        print ('Program started')
        sim.simxFinish(-1) # just in case, close all opened connections

        # 连接到CoppeliaSim的远程API服务器
        self.clientID = sim.simxStart('127.0.0.1', -3, True, True, 5000, 5)
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            print('Failed connecting to remote API server')
            sys.exit(1)

        # 设置同步模式
        synchronous = True
        sim.simxSynchronous(self.clientID, synchronous)
        time.sleep(2)

    def start_simulation(self):
        # 开始仿真
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_streaming)

    def stop_simulation(self):
        # 停止仿真
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_streaming)

    def disconnect_from_simulator(self):
        # 关闭连接
        sim.simxFinish(self.clientID)

    def __del__(self):
        self.disconnect_from_simulator()

    def Astar(self, image_data, lower_white, upper_white, center, destination):
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        
        # 创建白色区域的掩码
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # 将图像中的白色区域转换为1，其他区域转换为0
        binary_map = mask_white.astype(int)

        # 直接将掩码图像作为二值图像使用
        binary_map = mask_white

        # 直接将掩码图像作为二值图像使用
        binary_map = np.array(mask_white)
        
        FT.save_map_to_file(binary_map, "MapData/origin.csv")
        
        # 膨胀操作，将白色区域扩展一定距离
        kernel = np.ones((65, 65), np.uint8)  # 定义膨胀核的大小
        binary_map = (binary_map / 200).astype(int)
        map1 = As.Map(512, 512, binary_map)

        binary_map = cv2.dilate(binary_map.astype(np.uint8), kernel, iterations=1)      

        # 保存地图信息到文件
        FT.save_map_to_file(binary_map, "MapData/temp1.csv")

        map = As.Map(512, 512, binary_map)
        
        center = (int(center[0]), int(center[1]))
        destination = (int(destination[0]), int(destination[1]))
        start_time = time.time()
        path = As.AStarSearch(map, center, destination)
        end_time = time.time()

        elapsed_time = end_time - start_time
        if path:
            print("路径已找到：", path)
            As.visualize_path(map, path)
            print("规划花费了", elapsed_time, "s")
        else:
            print("没有找到路径。")

        return path, map1

    def run(self, reporter):
        # 连接到远程 API
        self.connect_to_simulator()
        # 开始启动仿真
        self.start_simulation()

        true_path = []
        # 获取视觉传感器和关节的句柄
        Vision_sensor = 'top_view_camera'
        error, Vision_sensor_Handle = sim.simxGetObjectHandle(self.clientID, Vision_sensor, sim.simx_opmode_blocking)
        if error != sim.simx_return_ok:
            print(f'Failed to get handle for sensor {Vision_sensor}')
            sys.exit(1)

        jointName1 = 'bubbleRob_leftMotor'
        error1, self.left_jointHandle = sim.simxGetObjectHandle(self.clientID, jointName1, sim.simx_opmode_blocking)
        if error1 != sim.simx_return_ok:
            print(f'Failed to get handle for sensor {jointName1}')
            sys.exit(1)

        jointName2 = 'bubbleRob_rightMotor'
        error2, self.right_jointHandle = sim.simxGetObjectHandle(self.clientID, jointName2, sim.simx_opmode_blocking)
        if error2 != sim.simx_return_ok:
            print(f'Failed to get handle for sensor {jointName2}')
            sys.exit(1)

        path = []
        count = 0
        flag = 3
        # 主循环
        for i in range(1, 2000):
            print('--------------------------epoch', i, '------------------------------')
            error_code, resolution, image_data = sim.simxGetVisionSensorImage(self.clientID, Vision_sensor_Handle, options=0, operationMode=sim.simx_opmode_streaming) 

            if error_code == sim.simx_return_ok:  
                image_data = np.array(image_data, dtype=np.int16)+256
                image_data = np.array(image_data, dtype=np.uint8)
                image_data.resize([resolution[1], resolution[0], 3])
                image_data = np.flipud(image_data)

                # 转换到HSV颜色空间  
                hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)  
                
                # 定义红色在HSV空间中的范围
                lower_red = np.array([0, 70, 50])  
                upper_red = np.array([10, 255, 255])

                # 定义蓝色在HSV空间中的范围
                lower_blue = np.array([100, 70, 50])  
                upper_blue = np.array([130, 255, 255]) 

                # 定义绿色在HSV空间中的范围
                lower_green = np.array([40, 70, 50])   
                upper_green = np.array([80, 255, 255])

                # 定义白色在HSV空间中的范围
                lower_white = np.array([0, 0, 230])
                upper_white = np.array([180, 10, 255])

                # 创建红色、蓝色、绿色和白色区域的掩码  
                mask_red = cv2.inRange(hsv, lower_red, upper_red)  
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
                mask_green = cv2.inRange(hsv, lower_green, upper_green) 
                mask_white = cv2.inRange(hsv, lower_white, upper_white)

                # 找到红色、蓝色和绿色的区域的轮廓  
                contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
                contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
                contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 计算红色和蓝色区域的中心点  
                if contours_red:  
                    c_red = max(contours_red, key=cv2.contourArea)  
                    M_red = cv2.moments(c_red)  
                    if M_red["m00"] != 0:  
                        cX_red = int(M_red["m10"] / M_red["m00"])  
                        cY_red = int(M_red["m01"] / M_red["m00"])  
                        print(f"Red region center at step {i}: ({cX_red}, {cY_red})")  
                else:  
                    # 没有找到红色轮廓  
                    cX_red, cY_red = None, None  
                
                if contours_blue:  
                    c_blue = max(contours_blue, key=cv2.contourArea)  
                    M_blue = cv2.moments(c_blue)  
                    if M_blue["m00"] != 0:  
                        cX_blue = int(M_blue["m10"] / M_blue["m00"])  
                        cY_blue = int(M_blue["m01"] / M_blue["m00"])  
                        print(f"Blue region center at step {i}: ({cX_blue}, {cY_blue})") 
                else:  
                    # 没有找到蓝色轮廓  
                    cX_blue, cY_blue = None, None  
                
                # 计算绿色区域的中心点（假设障碍物是最大的绿色轮廓）  
                if contours_green:  
                    c_green = max(contours_green, key=cv2.contourArea) 
                    M_green = cv2.moments(c_green)   
                    if M_green["m00"] != 0:  
                        cX_green = int(M_green["m10"] / M_green["m00"])  
                        cY_green = int(M_green["m01"] / M_green["m00"])  
                        print(f"Green region center at step {i}: ({cX_green}, {cY_green})") 
                else:
                    # 没有找到绿色轮廓
                    cX_green, cY_green = None, None  

                # 处理每个白色区域的轮廓
                for contour in contours_white:
                    # 计算白色区域的中心点
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # 在这里进行进一步的处理，例如，将中心点添加到一个列表中，以便后续使用
                        print(f"White region center at step {i}: ({cX}, {cY})")

                # 在图像中绘制白色区域的轮廓
                image_data_cv = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)  # 将图像转换为 BGR 格式
                cv2.drawContours(image_data_cv, contours_white, -1, (0, 255, 0), 2)  # 在转换后的图像上绘制轮廓
                image_data_cv_rgb = cv2.cvtColor(image_data_cv, cv2.COLOR_BGR2RGB)  # 将图像重新转换为 RGB 格式

                # 计算物体中心（红色与蓝色中心的平均值）  
                car_Xcenter = (cX_red + cX_blue) / 2 if cX_red is not None and cX_blue is not None else None  
                car_Ycenter = (cY_red + cY_blue) / 2 if cY_red is not None and cY_blue is not None else None  

                if car_Xcenter is not None and car_Ycenter is not None:
                    if count % 5==0:
                        true_path.append([car_Xcenter, car_Ycenter])

                if reporter._init_image is None and car_Ycenter is not None:
                    reporter.log_init_image(image_data)

                if (reporter._start_position is None or reporter._goal_position is None) and car_Xcenter is not None and car_Ycenter is not None:
                    car_position = np.array([car_Ycenter, car_Xcenter])
                    reporter.log_start_position(car_position)
                    goal_position = np.array([cY_green, cX_green])
                    reporter.log_goal_position(goal_position)
                    mask_white_inverted = cv2.bitwise_not(mask_white)
                    reporter.log_obstacle_mask(mask_white_inverted)

                if path == [] and cX_green is not None and car_Xcenter is not None:
                    path, map = self.Astar(image_data, lower_white, lower_white, [car_Xcenter, car_Ycenter], [cX_green, cY_green])
                    path = [path[i] for i in range(0, len(path), 55)]
                    path.append([cX_green, cY_green])
                    path = sm1.smooth_path(path, 200)
                    path = sm.smooth(path)
                    column1 = [row[0] for row in path]  
                    column2 = [row[1] for row in path]  
                    temp = 0
                    while temp < 3:
                        column1.pop(-2)
                        column2.pop(-2)
                        temp = temp + 1

                    print(column1)
                    print(column2)
                    self.tracker = fd.Point(column1, column2)

                    np_path = np.array(path)
                    np_path_corrected = np_path[:, [1, 0]]
                    reporter.log_plan_path(np_path_corrected)
                    
                    As.visualize_path(map, path)
                    (target_Xcenter, target_Ycenter) = self.tracker.search_target(car_Xcenter, car_Ycenter, self.vel_L, self.vel_R)

                if cX_green is not None and car_Xcenter is not None:

                    if FT.near((car_Xcenter, car_Ycenter), (cX_green, cY_green)):
                        print("到达目的地")  
                        self.stop_bubble()
                        robot_orientation = FT.calculate_orientation((cX_red, cY_red), (cX_blue, cY_blue))
                        sim_time = sim.simxGetLastCmdTime(self.clientID)/1000 + 0.0000000001
                        reporter.log_robot_sim_state(np.array([car_Ycenter, car_Xcenter]), robot_orientation, sim_time) 
                        reporter.report_all()
                        self.stop_simulation()
                        print('Program ended') 
                        break
                    (target_Xcenter, target_Ycenter) = self.tracker.search_target(car_Xcenter, car_Ycenter, self.vel_L, self.vel_R)

                    flag = flag - 1   
                    print((target_Xcenter, target_Ycenter))     
                    self.control_bubble((car_Xcenter, car_Ycenter), (target_Xcenter, target_Ycenter), (cX_red, cY_red), (cX_blue, cY_blue)) 
                    if flag < 0:
                        robot_orientation = FT.calculate_orientation((cX_red, cY_red), (cX_blue, cY_blue))
                        sim_time = sim.simxGetLastCmdTime(self.clientID)/1000
                        reporter.log_robot_sim_state(np.array([car_Ycenter, car_Xcenter]), robot_orientation, sim_time)
                        flag = 2
                sim.simxSynchronousTrigger(self.clientID)