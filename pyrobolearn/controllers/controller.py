#!/usr/bin/env python
"""Provide the abstract controller's class.
"""

__author__ = ["Brian Delhaisse"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Controller(object):
    r"""Controller (abstract) class.

    Controllers accept as inputs the robot's state (or observations), and outputs the joint actions. Compared to
    policies (as defined in `pyrobolearn/policies`), they do not possess any parameters to optimize. They are manually
    coded by the user. They can however use optimization processes inside the controller, such as quadratic
    programming.
    """

    def __init__(self):
        pass

    ###########
    # Methods #
    ###########

    def act(self, *args, **kwargs):
        pass

    #############
    # Operators #
    #############

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)
