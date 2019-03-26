#!/usr/bin/env python
"""Define the Isaac SDK simulator API.

This is the main interface that communicates with the Isaac SDK simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator.

The signature of each method defined here are inspired by [2] but in accordance with the PEP8 style guide [3].
Parts of the documentation for the methods have been copied-pasted from [2] for completeness purposes.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] Isaac SDK: https://developer.nvidia.com/isaac-sdk
    [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    [3] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

# TODO: waiting for its release at the end of March

import time
import numpy as np

from pyrobolearn.simulators.simulator import Simulator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Isaac(Simulator):
    r"""Isaac simulator.

    "Isaac Sim is a virtual robotics laboratory, a high-fidelity 3D world simulator, that accelerates the research,
    design and development of robots by reducing both cost and risk. Developers can quickly and easily train and test
    their robots created with the Isaac SDK, in detailed, highly realistic scenarios resulting robots that can safely
    operate and cooperate with humans." [1]

    References:
        [1] https://developer.nvidia.com/isaac-sdk
        [2] https://www.nvidia.com/en-au/deep-learning-ai/industries/robotics/
        [3] "GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning", Liang et al., 2018
    """

    def __init__(self, render=True, **kwargs):
        super(Isaac, self).__init__()
