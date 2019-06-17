#!/usr/bin/env python
"""Define the Nvidia FleX Simulator API.

This is the main interface that communicates with the FleX simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
FleX.

Warnings: We are waiting for [3] to publish their code.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] Nvidia FleX: https://developer.nvidia.com/flex
    [2] Python bindings for the Nvidia FleX simulator: https://github.com/henryclever/FleX_PyBind11
    [3] "GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning":
        https://sites.google.com/view/accelerated-gpu-simulation/home
"""

# TODO: see Isaac instead...

from pyrobolearn.simulators.simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Flex(Simulator):
    r"""FleX simulator
    """

    def __init__(self):
        super(Flex, self).__init__()
        raise NotImplementedError
