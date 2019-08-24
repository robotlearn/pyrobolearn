
# General imports
import os
import importlib
import inspect
import re

# General robot class
from .base import Body, MovableBody, ControllableBody
from . import actuators
from . import sensors
# from .actuators import *
# from .sensors import *
from .robot import Robot

# Categories/types of robots
from .legged_robot import LeggedRobot, BipedRobot, QuadrupedRobot, HexapodRobot
from .manipulator import Manipulator, BiManipulator
from .wheeled_robot import WheeledRobot, DifferentialWheeledRobot, AckermannWheeledRobot
from .uav import UAVRobot, FixedWingUAV, RotaryWingUAV, FlappingWingUAV
from .usv import USVRobot
from .uuv import UUVRobot
from .hand import Hand, TwoHand
from .gripper import Gripper, ParallelGripper, AngularGripper, VacuumGripper

# Mujoco models
from .ant import Ant
from .hopper import Hopper
from .walker2d import Walker2D
from .half_cheetah import HalfCheetah
from .swimmer import Swimmer
from .humanoid import Humanoid

# Quadruped
from .aibo import Aibo
from .minitaur import Minitaur
from .littledog import LittleDog
from .anymal import ANYmal
from .hyq import HyQ
from .hyq2max import HyQ2Max
from .opendog import OpenDog
from .laikago import Laikago

# Biped
from .cassie import Cassie

# Biped + Bi-manipulators
from .atlas import Atlas
from .nao import Nao
from .icub import ICub
from .coman import Coman
from .walkman import Walkman
from .cogimon import Cogimon
from .darwin import Darwin
# from .kondo import KHR3HV
from .hubo import Hubo

# Hexapod
from .crab import Crab
from .sea_hexapod import SEAHexapod
from .phantomx import PhantomX
from .morphex import Morphex
from .rhex import Rhex

# Control
from .acrobot import Acrobot
from .pendulum import Pendulum

# Manipulators
from .rrbot import RRBot
from .wam import WAM, BarrettHand
from .kuka_lwr import KukaLWR
from .kuka_iiwa import KukaIIWA
from .jaco import Jaco, JacoGripper
from .franka import Franka, FrankaGripper
from .sawyer import Sawyer
from .edo import Edo
from .kr5 import KR5
from .manipulator2d import Manipulator2D
from .ur import UR3, UR5, UR10

# Bi-Manipulators
from .baxter import Baxter, BaxterGripper

# Hands
from .allegrohand import AllegroHand
from .softhand import SoftHand
from .shadowhand import ShadowHand
from .schunk_hand import SchunkHand

# Wheeled
from .epuck import Epuck
from .f10_racecar import F10Racecar
from .mkz import MKZ
from .husky import Husky

# Wheeled + (single) manipulator
from .fetch import Fetch, FetchGripper

# Wheeled + Bi-manipulators
from .pepper import Pepper
from .pr2 import PR2, PR2Gripper

# Wheeled + Quadruped + Bi-manipulators
from .centauro import Centauro

# UAV
from .quadcopter import Quadcopter
# from .techpod import Techpod
from .flappy import Flappy

# UUV
# from .ecaa9 import ECAA9

# USV

# Quadruped + UUV + USV
from .pleurobot import Pleurobot

# Others
from .cartpole import CartPole
from .cubli import Cubli
from .sea_snake import SEASnake
from .bb8 import BB8
from .youbot import Youbot, YoubotBase, KukaYoubotArm, YoubotDualArm, YoubotGripper
from .ballbot import Ballbot

# Robots #

# get a list of implemented robots
path = os.path.dirname(__file__)
implemented_robots = set([f[:-3] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                          and f.endswith('.py')])
# remove few items from the set
for s in ['__init__', 'actuators', 'sensors', 'legged_robot', 'manipulator', 'wheeled_robot', 'uav', 'usv',
          'uuv', 'hand', 'gripper']:
    if s in implemented_robots:
        implemented_robots.remove(s)

implemented_robots = list(implemented_robots)

# TODO: fix problem with icub
implemented_robots.remove('icub')

# create dictionary that maps robot names to robot classes
robot_names_to_classes = {}
implemented_grippers = []
for robot_name in implemented_robots:
    module = importlib.import_module('pyrobolearn.robots.' + robot_name)  # 'robots.'+robot)
    # robot_class = getattr(module, robot.capitalize())
    for name, cls in inspect.getmembers(module):
        # check if it is a class, and the names match
        if inspect.isclass(cls) and issubclass(cls, Robot):
            if name.lower() == ''.join(robot_name.split('_')):
                robot_names_to_classes[robot_name] = cls
                name = robot_name
            else:
                name_list = re.findall('[0-9]*[A-Z]+[0-9]*[a-z]*', name)
                name = '_'.join([n.lower() for n in name_list])
                # TODO: improve regex
                if name == 'wamgripper':
                    name = 'wam_gripper'
                elif name == 'uuvrobot':
                    name = 'uuv_robot'
                elif name == 'usvrobot':
                    name = 'usv_robot'
                robot_names_to_classes[name] = cls

            # add grippers and hands
            if issubclass(cls, Gripper) or issubclass(cls, Hand):
                implemented_grippers.append(name)

implemented_robots = set(list(robot_names_to_classes.keys()))
implemented_grippers = set(implemented_grippers)


# function to disable the motors
# this can be useful when resetting the joint state
def disable_motors(robot, joint_ids=None):
    """Return a function that disables the motors."""
    def reset():
        robot.disable_motor(joint_ids=joint_ids)
    return reset
