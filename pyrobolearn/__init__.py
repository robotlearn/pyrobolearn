
import sys

# import simulators
# from simulators import *
import simulators

# import robots
# from robots import *
import robots

# import worlds
# from worlds import *
import worlds

# import states
# from states import *
import states

# import actions
# from actions import *
import actions

# import rewards
# from rewards import *
import rewards

# import environments
# from envs import *
import envs

# import models
import models

# import approximators
import approximators

# import policies
import policies

# import values

# import actor-critics

# import dynamical models

# import tools (interfaces and bridges)
# import tools

# import tasks
import tasks

# import metrics

# import algos
import algos

# import experiments


# Meta-information about the package
__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


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
