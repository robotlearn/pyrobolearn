#!/usr/bin/env python
"""Define the OpenSim Simulator API.

This is the main interface that communicates with the OpenSim simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
OpenSim.

Warnings: This simulator only works for musculoskeletal models.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] OpenSim: https://opensim.stanford.edu/
    [2] OpenSim Core: https://github.com/opensim-org/opensim-core
    [3] OpenSim Reinforcement Learning: https://github.com/stanfordnmbl/osim-rl
"""

from simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class OpenSim(Simulator):
    r"""OpenSim simulator

    """

    def __init__(self):
        super(OpenSim, self).__init__()
        raise NotImplementedError
