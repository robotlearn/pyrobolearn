# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the abstract controller's class.
"""

__author__ = ["Brian Delhaisse"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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

    def __init__(self, rate=1):
        """
        Initialize the controller.

        Args:
            rate (int, float): rate (float) at which the controller operates if we are operating in real-time. If we
                are stepping deterministically in the simulator, it represents the number of ticks (int) to sleep
                before executing the model.
        """
        self.rate = rate

    ##############
    # Properties #
    ##############

    @property
    def rate(self):
        """Return the rate."""
        return self._rate

    @rate.setter
    def rate(self, rate):
        """Set the rate."""
        if not isinstance(rate, int):
            raise TypeError("Expecting the given 'rate' to be an int, instead got: {}".format(type(rate)))
        if rate <= 0:
            raise ValueError("Expecting the rate to be positive, instead got: {}".format(rate))
        self._rate = rate

    ###########
    # Methods #
    ###########

    def act(self, *args, **kwargs):
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)
