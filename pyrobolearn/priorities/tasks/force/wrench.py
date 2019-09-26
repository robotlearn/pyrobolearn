# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the wrench task.

The wrench task tries to generate a wrench near the desired one by minimizing:

.. math:: || w - k (w_{des} - w_t) ||^2

where :math:`w = [f \tau] \in \mathbb{R}^6` is the wrench vector being optimized, :math:`k` is a proportional gain,
:math:`w_{des}` is the desired wrench vector, and :math:`w_t` is the current wrench vector.

The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
:math:`A = I`, :math:`x = w`, and :math:`b = k (w_{des} - w_t)`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import ForceTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WrenchTask(ForceTask):  # TODO: improve this class by considering only forces or torques + using links
    r"""Wrench Task

    The wrench task tries to generate a wrench near the desired one by minimizing:

    .. math:: || w - k (w_{des} - w_t) ||^2

    where :math:`w = [f \tau] \in \mathbb{R}^6` is the wrench vector being optimized, :math:`k` is a proportional gain,
    :math:`w_{des}` is the desired wrench vector, and :math:`w_t` is the current wrench vector.

    The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = I`, :math:`x = w`, and :math:`b = k (w_{des} - w_t)`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, desired_wrenches, wrenches, kp=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            desired_wrenches (list[np.array[float[6]]]): list of desired wrenches.
            wrenches (list[np.array[float[6]]]): list of current wrenches that are usually read from F/T sensors. This
              has to be of the same size as the desired wrenches.
            weight (float, np.array[float[M*6,M*6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(WrenchTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set variables
        self.desired_wrenches = desired_wrenches
        self.wrenches = wrenches
        self.kp = kp

        # first update
        self.update()

    ##############
    # Properties #
    ##############
    
    @property
    def desired_wrenches(self):
        """Get the desired wrenches."""
        return self._desired_wrenches
    
    @desired_wrenches.setter
    def desired_wrenches(self, wrenches):
        """Set the desired wrenches."""
        if not isinstance(wrenches, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'desired_wrenches' to be a tuple/list of np.array, or a np.array, "
                            "but got instead: {}".format(type(wrenches)))
        self._desired_wrenches = np.asarray(wrenches).reshape(-1)  # (N*6,) or (N*3,)

        # enable / disable the tasks based on the number of contact links
        if len(self._desired_wrenches) == 0:
            self.disable()
        else:
            self.enable()
            # set A matrix
            self._A = np.identity(len(self._desired_wrenches))

    @property
    def wrenches(self):
        """Get the current wrenches."""
        return self._wrenches

    @wrenches.setter
    def wrenches(self, wrenches):
        """Set the current wrenches."""
        if not isinstance(wrenches, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'desired_wrenches' to be a tuple/list of np.array, or a np.array, "
                            "but got instead: {}".format(type(wrenches)))
        self._wrenches = np.asarray(wrenches).reshape(-1)  # (N*6,) or (N*3,)

    @property
    def kp(self):
        """Return the proportional gain."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the proportional gain."""
        if kp is None:
            kp = 1.
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given proportional gain kp to be an int, float, np.array, instead "
                            "got: {}".format(type(kp)))
        x_size = len(self.desired_wrenches)
        if isinstance(kp, np.ndarray) and kp.shape != (x_size, x_size):
            raise ValueError("Expecting the given proportional gain matrix kp to be of shape {}, but instead "
                             "got shape: {}".format((x_size, x_size), kp.shape))
        self._kp = kp

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        self._b = self._kp * (self._desired_wrenches - self._wrenches)
