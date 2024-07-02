import reporter as rp
import Controller.BubbleRobController as BRC

if __name__ == "__main__":
    # 实例化控制器并运行
    controller = BRC.BubbleRobController()
    # 实例化report类
    reporter = rp.Reporter()
    # 调用启动控制函数
    controller.run(reporter)
