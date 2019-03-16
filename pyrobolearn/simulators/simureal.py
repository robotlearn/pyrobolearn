# This file defines an interface which is used by the robot classes.
# This falls under the "Adapter" design pattern, where we add an abstraction
# layer, by providing a common interface to different simulators and real robots.
#
# The UML diagram is depicted below:
#
#   simuRealInterface  -----------<>  robot / gym-env
#     -----^-----
#     |         |
#  ros_rbdl   pybullet
#     |
#  ros_gazebo
#
# where the robot and gym-env classes only interact with children from env_interface.
#
# --- Example ---
# env_gazebo = ros_gazebo()
# robot = Robot(env_gazebo, 'path_to_urdf')
# print(robot.getJointStates())         # will check the joint state in gazebo.
# robot.drawCoM()                       # will draw a small sphere at the CoM in the gazebo simulator.
#
# env_bullet = pybullet()
# robot.change_env(env_bullet)          # change env and reload the urdf in the given env.
# print(robot.getJointStates())         # will check the joint state in pybullet.
# robot.drawCoM()                       # will draw a small sphere at the CoM in the pybullet simulator.
#
# env_ros = ros_rbdl()                  # assuming the real robot can send and recv msgs via
# robot.change_env(env_ros)             # rostopics/rosservices, you can interact with it.
# print(robot.getJointStates())         # will check the joint state via ros.
# robot.drawCoM()                       # return error as we can't draw in the real world.
# ---------------
#
# You can thus interact with different simulators or the real robots.
# Simulators: pybullet, pygazebo, ros-gazebo
#
# Warning: the name might change in the future.


from abc import ABCMeta, abstractmethod


class SimuRealInterface(object):
    """Simulation-Reality Interface.
    This abstract class must be inherited by any simulators, or real interfaces.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def stepSimulation(self):
        raise NotImplementedError("Step simulation is not implemented.")

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def loadURDF(self, filename, position, orientation):
        raise NotImplementedError()

