# -*- coding: utf-8 -*-

# import the abstract model interface
from .model import ModelInterface

# import the rbdl model interface
try:
    import rbdl
    from .rbdl_model import RBDLModelInterface
except ImportError as e:
    print("RBDL could not be found on this system... Skipping priorities.RBDLModelInterface...")

# import the robot model interface
from .robot_model import RobotModelInterface
