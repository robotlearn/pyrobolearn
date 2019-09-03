#!/usr/bin/env python
r"""Provide the interaction task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask
from pyrobolearn.priorities.tasks.velocity import CartesianTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman, Alessio Rocchi) (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class InteractionTask(JointVelocityTask):
    r"""Interaction Task

    From the documentation of the framework of [1]: "the `InteractionTask` class implements an admittance based force
    control using the admittance law:

    .. math::

        dx = K_p (w_d - w) \\
        x_d = x + dx

    where :math:`w_d \in \mathbb{R}^6` is the desired wrench in some base_link frame, :math:`w` is the measured wrench
    transformed from the Force/Torque sensor frame to the base_link frame. The displacement :math:`dx` is integrated
    using the previous position :math:`x`, and a new desired position :math:`x_d` is computed. The references
    :math:`x_d` and :math:`dx` are then used inside a Cartesian task (see ``CartesianTask``).

    Warnings: the :math:`w_d` is the desired wrench that the robot has to exert on the environment, so the measured
    wrench :math:`w` is the wrench produced by the robot on the environment (and not the opposite)!"


    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, base_link=None, desired_wrench=0., kp=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the base root link.
            desired_wrench (float, np.array[float[6]]): desired wrench (force and torque) in the base link of reference
              frame.
            kp (float, np.array[float[6,6]]): proportional gain = compliance matrix.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(InteractionTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        
        self.desired_wrench = desired_wrench
        self.wrench = None  # measured wrench
        self.kp = kp

        # create sub-task
        self._task = CartesianTask(model, distal_link=distal_link, base_link=base_link, weight=weight)

    ##############
    # Properties #
    ##############

    @property
    def desired_wrench(self):
        """Get the desired wrench."""
        return self._desired_wrench

    @desired_wrench.setter
    def desired_wrench(self, wrench):
        """Set the desired wrench."""
        if wrench is None:
            wrench = np.zeros(6)
        elif isinstance(wrench, (float, int)):
            wrench = wrench * np.ones(6)
        if not isinstance(wrench, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'desired_wrench' to be a tuple/list of np.array, or a np.array, "
                            "but got instead: {}".format(type(wrench)))
        self._desired_wrench = np.asarray(wrench).reshape(-1)  # (N*6,) or (N*3,)

    @property
    def wrench(self):
        """Get the current wrench."""
        return self._wrench

    @wrench.setter
    def wrench(self, wrench):
        """Set the current wrench expressed in the base link of reference frame."""
        if wrench is not None:
            if not isinstance(wrench, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given 'desired_wrench' to be a tuple/list of np.array, or a np.array, "
                                "but got instead: {}".format(type(wrench)))
            wrench = np.asarray(wrench).reshape(-1)  # (6,) or (3,)
        self._wrench = wrench

        # enable / disable the tasks based on if the wrench was provided or not
        if self._wrench is None:
            self.disable()
        else:
            self.enable()

    @property
    def kp(self):
        """Return the proportional gain / compliance matrix."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the proportional gain / compliance matrix."""
        if kp is None:
            kp = 1.
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given compliance matrix gain kp to be an int, float, np.array, instead "
                            "got: {}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (6, 6):
            raise ValueError("Expecting the given compliance matrix gain kp to be of shape {}, but instead "
                             "got shape: {}".format((6, 6), kp.shape))
        self._kp = kp

    ###########
    # Methods #
    ###########
    
    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        x = self.model.get_pose(link=self.distal_link, wrt_link=self.base_link)
        dx = np.dot(self.kp, (self.desired_wrench - self.wrench))

        # update cartesian task
        self._task.set_desired_references(x_des=x, dx_des=dx)
        self._task.update()
        self._A = self._task.A
        self._b = self._task.b

        # set wrench to None
        self._wrench = None
