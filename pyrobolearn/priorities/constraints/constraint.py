#!/usr/bin/env python
"""Provide the various constraints used in QP.
"""

import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Constraint(object):
    r"""Constraint (abstract) class.

    Python implementation of Constraints based on the slides of the OpenSoT framework [1].

    References:
        [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN"
            ([code](https://opensot.wixsite.com/opensot),
            [slides](https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA),
            [tutorial video](https://www.youtube.com/watch?v=yFon-ZDdSyg),
            [old code](https://github.com/songcheng/OpenSoT)), Rocchi et al., 2015
    """

    def __init__(self, model):
        """
        Initialize the Constraint.

        Args:
            model (robot, str): robot model.
        """
        self._model = model

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the robot model."""
        return self._model

    ###########
    # Methods #
    ###########

    def compute(self):
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    def __call__(self):
        return self.compute()
