#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Chrono Simulator API.

Dependencies in PRL: None

References:
    - [1] Chrono - An Open Source Multi-physics Simulation Engine: https://projectchrono.org/
    - [1] PyChrono: http://projectchrono.org/pychrono/
"""

# import standard libraries
import os
import time
import numpy as np
from collections.abc import OrderedDict

# import pychrono
try:
    import pychrono as chrono
except ImportError as e:
    raise ImportError(e.__str__() + "\n: HINT: you can install `pychrono` by following the instructions given at: "
                                    "http://api.projectchrono.org/development/pychrono_installation.html")

# import PRL simulator
from pyrobolearn.simulators.simulator import Simulator

# check Python version
import sys
if sys.version_info[0] < 3:
    raise RuntimeError("You must use Python 3 with the Dart simulator.")


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Chrono(Simulator):
    r"""Chrono simulator class.

    """
    pass
