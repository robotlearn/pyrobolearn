"""PyRoboLearn: a Python framework in Robot Learning for Education and Research.
"""

name = "pyrobolearn"

import sys
import signal
from itertools import count

# logging
import logging

# create logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# import simulators
from . import simulators

# import robots
from . import robots

# import worlds
from . import worlds

# import physics randomizer
from . import physics

# import states
from . import states

# import actions
from . import actions

# import terminal conditions
from . import terminal_conditions

# import rewards
from . import rewards

# import environments
from . import envs

# import models
from . import models

# import approximators
from . import approximators

# import policies
from . import policies

# import values
from . import values

# import actor-critics
from . import actorcritics

# import dynamical models
from . import dynamics

# import tools (interfaces and bridges)
from . import tools

# import recorders
from . import recorders

# import tasks
from . import tasks

# import metrics
from . import metrics

# import losses
from . import losses

# import optimizers
from . import optimizers

# import algos
from . import algos

# import experiments

# import priority tasks
# from . import priorities


# Meta-information about the package
__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# Capture a SIGINT in Python: make sure we quit PRL when Ctrl+C is pressed
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# https://stackoverflow.com/questions/30483246/how-to-check-if-a-python-module-has-been-imported
# https://stackoverflow.com/questions/14050281/how-to-check-if-a-python-module-exists-without-importing-it/25045228
def module_imported(module_name):  # TODO: improve this method
    """Check if the given module has been already imported."""
    if not isinstance(module_name, str):
        module_name = str(module_name)
    if module_name in sys.modules:
        return True
    return False


# Define what submodules/classes/functions should be loaded when writing 'from pyrobolearn import *'
# __all__ = [
#     # Submodules
#
#     # Classes
#
#     # Functions
#
#     # Context managers
#
#     # package information
#     "__version__",
#     # Deprecated
#
# ]
