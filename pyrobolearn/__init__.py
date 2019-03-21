
import sys

# import simulators
from . import simulators

# import robots
from . import robots

# import worlds
from . import worlds

# import states
from . import states

# import actions
from . import actions

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

# import actor-critics

# import dynamical models

# import tools (interfaces and bridges)
# from . import tools  # uncommenting this will oblige the user to install a bunch of libraries which are not straightforward to install...

# import tasks
from . import tasks

# import metrics

# import optimizers
# from . import optimizers

# import algos
from . import algos

# import experiments


# Meta-information about the package
__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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
