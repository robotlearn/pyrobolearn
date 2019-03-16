
# General imports
import os
import importlib
import inspect

# General robot class
from base import Object, MovableObject, ControllableObject
from actuators import *
from sensors import *
from robot import Robot

# Categories/types of robots
from legged_robot import *
from manipulator import *
from wheeled_robot import *
from uav import *
from usv import *
from uuv import *
from hand import *

# Mujoco models
from ant import Ant
from hopper import Hopper
from walker2d import Walker2D
from half_cheetah import HalfCheetah
from swimmer import Swimmer
from humanoid import Humanoid

# Quadruped
from aibo import Aibo
from minitaur import Minitaur
from littledog import LittleDog
# from anymal import ANYmal
from hyq import HyQ
from hyq2max import HyQ2Max
from opendog import OpenDog
from laikago import Laikago

# Biped
from cassie import Cassie

# Biped + Bi-manipulators
from atlas import Atlas
from nao import Nao
from icub import ICub
from coman import Coman
from walkman import Walkman
from cogimon import Cogimon
from darwin import Darwin
# from kondo import KHR3HV

# Hexapod
from crab import Crab
from sea_hexapod import SEAHexapod
from phantomx import PhantomX
from morphex import Morphex

# Manipulators
from rrbot import RRBot
from wam import WAM
from kuka_lwr import KukaLWR
from kuka_iiwa import KukaIIWA
from jaco import Jaco
from franka import Franka
from sawyer import Sawyer

# Bi-Manipulators
from baxter import Baxter

# Hands
from allegrohand import AllegroHand
from softhand import SoftHand

# Wheeled
from epuck import Epuck
from f10_racecar import F10Racecar

# Wheeled + (single) manipulator
from fetch import Fetch

# Wheeled + Bi-manipulators
from pepper import Pepper
from pr2 import PR2

# Wheeled + Quadruped + Bi-manipulators
from centauro import Centauro

# UAV
from quadcopter import Quadcopter
# from techpod import Techpod

# UUV
# from ecaa9 import ECAA9

# USV

# Quadruped + UUV + USV
from pleurobot import Pleurobot

# Others
from cartpole import CartPole
from cubli import Cubli
from sea_snake import SEASnake
from bb8 import BB8
from youbot import Youbot, YoubotBase, KukaYoubotArm, YoubotDualArm


# Robots #

# get a list of implemented robots
path = os.path.dirname(__file__)
implemented_robots = set([f[:-3] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                          and f.endswith('.py')])
# remove few items from the set
for s in ['__init__', 'actuators', 'sensors', 'legged_robot', 'manipulator', 'wheeled_robot', 'uav', 'usv',
          'uuv', 'hand']:
    if s in implemented_robots:
        implemented_robots.remove(s)

implemented_robots = list(implemented_robots)


# create dictionary that maps robot names to robot classes
robot_names_to_classes = {}
for robot_name in implemented_robots:
    module = importlib.import_module('pyrobolearn.robots.' + robot_name)  # 'robots.'+robot)
    # robot_class = getattr(module, robot.capitalize())
    for name, obj in inspect.getmembers(module):
        # check if it is a class, and the names match
        if inspect.isclass(obj) and name.lower() == ''.join(robot_name.split('_')):
            robot_names_to_classes[robot_name] = obj
            break
