import numpy as np
import math

k = 0.2  # 预瞄距离系数
Lfc = 20  # 初始预瞄距离

class Point:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target(self, x, y, vl, vr):
        if self.old_nearest_point_index is None:
            # 初始搜索最近点
            dx = [x - icx for icx in self.cx]
            dy = [y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = math.hypot(self.cx[ind] - x, self.cy[ind] - y)
            while True:
                if ind + 1 >= len(self.cx):
                    return (self.cx[ind], self.cy[ind])
                distance_next_index = math.hypot(self.cx[ind+1] - x, self.cy[ind+1] - y)
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * (vl + vr)/2 + Lfc  # 根据速度更新预瞄距离

        # 搜索预瞄点索引
        while Lf > math.hypot(self.cx[ind] - x, self.cy[ind] - y):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        if ind < 5:
            ind = 5
        return (self.cx[ind], self.cy[ind])