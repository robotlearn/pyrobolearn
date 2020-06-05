#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Boyang Ti"
__copyright__ = "Copyright 2020, PyRoboLearn"
__credits__ = ["Boyang Ti"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Boyang Ti"
__email__ = "tiboyang@outlook.com"
__status__ = "Development"

from pyrobolearn.utils.plotting.plot import RealTimePlot
import pyrobolearn as prl
import numpy as np
from pyrobolearn.utils.transformation import *
from pyrobolearn.robots import Body

class EeFtRealTimePlot(RealTimePlot):

    def __init__(self, robot, sensor, forcex=False, forcey=False, forcez=False, torquex=False,
                 torquey=False, torquez=False, num_point=100, xlims=None, ylims=None,
                 suptitle='End effector Force and Torque', ticks=1, blit=True, interval=0.0001):
        """
        初始化实时绘制的机械臂的相关配置

        参数：
            robot: 所创造的机械臂实体
            force: 如果为真则绘制力信息
            torque: 如果为真则绘制力矩信息
            num_point: 保持多少个点在实时的绘制坐标系下
            xlims: x轴的限制
            ylims: y轴的限制
            suptitle: 图的标题
            ticks: 采样实时点的时间步间隔
            blit: 如果为真只更新数据内容不会改变标注等内容
            interval: 在不同frame之间的延迟单位mm
        """
        # 设置机器人实例
        if not isinstance(robot, prl.robots.Robot):
            raise TypeError("Expecting the given 'robot' to be an instance of `Robot`, but got instead: "
                            "{}".format(robot))
        if not isinstance(sensor, prl.robots.sensors.JointForceTorqueSensor):
            raise TypeError("Expecting the given 'sensor' to be an instance of `sensor`, but got instead: "
                            "{}".format(sensor))
        self._robot = robot
        self._sensor = sensor

        self.axis_ids = ['Force', 'Torque']
        # 设置图像布局
        nrows, ncols = 1, 1

        # 设置我们所需要绘制的参数
        self._plot_Fx = bool(forcex)
        self._plot_Fy = bool(forcey)
        self._plot_Fz = bool(forcez)
        self._plot_Tx = bool(torquex)
        self._plot_Ty = bool(torquey)
        self._plot_Tz = bool(torquez)

        states = np.array([self._plot_Fx, self._plot_Fy, self._plot_Fz, self._plot_Tx, self._plot_Ty, self._plot_Tz])
        self._num_states = len(states[states])

        if len(self.axis_ids) == 0:
            raise ValueError("Expecting to plot at least something (force or torque)")
        if len(self.axis_ids) == 1:
            ncols = 1
        else:
            ncols = 2

        # 设置点
        self._num_points = num_point if num_point > 10 else 10

        # 检查x和y的极限
        if xlims is None:
            xlims = (0, self._num_points)
        if ylims is None:
            ylims = (-2000, 2000)

        super(EeFtRealTimePlot, self).__init__(nrows=nrows, ncols=ncols, xlims=xlims, ylims=ylims,
                                                    titles=['Force', 'Torque'],
                                                    suptitle=suptitle, ticks=ticks, blit=blit, interval=interval)

    def _init(self, axes):
        """
        初始化图像在每个轴下创造线
        :param axes:
        :return:
        """
        self._lines = []
        for i, axis_ids in enumerate(['Force', 'Torque']):
            axes[0].legend(loc='upper left')
            axes[1].legend(loc='upper left')
            if self._plot_Fx:
                line, = axes[0].plot([], [], lw=self._linewidths[i], color='r', label='Fx')
                self._lines.append(line)
            if self._plot_Fy:
                line, = axes[0].plot([], [], lw=self._linewidths[i], color='y', label='Fy')
                self._lines.append(line)
            if self._plot_Fz:
                line, = axes[0].plot([], [], lw=self._linewidths[i], color='g', label='Fz')
                self._lines.append(line)
            if self._plot_Tx:
                line, = axes[1].plot([], [], lw=self._linewidths[i], color='m', label='Tx')
                self._lines.append(line)
            if self._plot_Ty:
                line, = axes[1].plot([], [], lw=self._linewidths[i], color='k', label='Ty')
                self._lines.append(line)
            if self._plot_Tz:
                line, = axes[1].plot([], [], lw=self._linewidths[i], color='b', label='Tz')
                self._lines.append(line)
        self._x = []
        length = len(self.axis_ids) * self._num_states
        self._ys = [[] for _ in range(length)]

    def _init_anim(self):
        """
        Init function (plot the background of each frame) that is passed to FuncAnimation. This has to be
        implemented in the child class.
        :return:
        """
        for line in self._lines:
            line.set_data([], [])
        return self._lines

    def _set_line(self, line_idx, data, state_name):
        """
        设置新的数据的来划线
        :param axis_idx: joint index
        :param line_idx: line index
        :param data: data sent through the pipe
        :param state_name: name of the state; select from Fx, Fy, Fz, Tx, Ty, Tz
        :return:
        """
        self._ys[line_idx].append(data[state_name])
        self._ys[line_idx] = self._ys[line_idx][-self._num_points:]
        self._lines[line_idx].set_data(self._x, self._ys[line_idx])
        line_idx += 1
        return line_idx

    def _animate_data(self, i, data):
        """
        Animate function that is passed to FuncAnimation. This has to be implemented in the child class.
        :param i: frame counter
        :param data: data that has been received from the pipe
        :return: list of object to update
        """
        if len(self._x) < self._num_points:
            self._x = range(len(self._x) + 1)

        k = 0
        for j in range(len(self.axis_ids)):
            if self._plot_Fx:
                k = self._set_line(line_idx=k, data=data, state_name='Fx')
            if self._plot_Fy:
                k = self._set_line(line_idx=k, data=data, state_name='Fy')
            if self._plot_Fz:
                k = self._set_line(line_idx=k, data=data, state_name='Fz')
            if self._plot_Tx:
                k = self._set_line(line_idx=k, data=data, state_name='Tx')
            if self._plot_Ty:
                k = self._set_line(line_idx=k, data=data, state_name='Ty')
            if self._plot_Tz:
                k = self._set_line(line_idx=k, data=data, state_name='Tz')
        return self._lines

    def _update(self):
        """
        This return the next data to be plotted; this has to be implemented in the child class.
        :return:data to be sent through the pipe and that have to be plotted. This will be given to `_animate_data`.
        """
        data = {}
        if self._sensor.sense() is None:
            data['Fx'] = 0
            data['Fy'] = 0
            data['Fz'] = 0
            data['Tx'] = 0
            data['Ty'] = 0
            data['Tz'] = 0
            return data
        if self._plot_Fx:
            data['Fx'] = self._sensor.sense()[0]
        if self._plot_Fy:
            data['Fy'] = self._sensor.sense()[1]
        if self._plot_Fz:
            data['Fz'] = self._sensor.sense()[2]
        if self._plot_Tx:
            data['Tx'] = self._sensor.sense()[3]
        if self._plot_Ty:
            data['Ty'] = self._sensor.sense()[4]
        if self._plot_Tz:
            data['Tz'] = self._sensor.sense()[5]
        return data

if __name__ == '__main__':
    # Try to move the robot in the simulator
    # WARNING: DON'T FORGET TO CLOSE FIRST THE FIGURE THEN THE SIMULATOR OTHERWISE YOU WILL HAVE THE PLOTTING PROCESS
    # STILL RUNNING
    from itertools import count



    sim = prl.simulators.Bullet()
    world = prl.worlds.BasicWorld(sim)
    robot = world.load_robot('kuka_iiwa')

    box = world.load_visual_box(position=[0.7, 0., 0.2], orientation=get_quaternion_from_rpy([0, 1.57, 0]),
                                dimensions=(0.2, 0.2, 0.2))
    box = Body(sim, body_id=box)

    sensor = prl.robots.sensors.JointForceTorqueSensor(sim, body_id=robot.id, joint_ids=5)
    plot = EeFtRealTimePlot(robot, sensor=sensor, forcex=True, forcey=True, forcez=True,
                            torquex=True, torquey=True, torquez=True, ticks=24)

    for t in count():
        plot.update()
        world.step(sim.dt)