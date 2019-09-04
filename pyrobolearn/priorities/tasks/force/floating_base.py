#!/usr/bin/env python
r"""Provide the floating base force task.

From [1]: "this implements a task which maps forces acting on the floating base virtual chain to contacts".

.. math:: || J(q)[:,:6]^\top w - \tau ||^2,

where :math:`w \in \mathbb{R}^{6N_c}` are the wrench vector being optimized (with :math:`N_c` being the number of
contacts), :math:`J(q) = [J(q)_1^\top \cdot J(q)_{N_c}^\top]^\top \in \mathbb{R}^{6N_c \times 6 + N}` are the
concatenated jacobians, and :math:`\tau` are the torques applied on the floating base.

Note that this task assumes the robot has a floating base.

The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
:math:`A = J(q)[:,:6]^\top`, :math:`x = w`, and :math:`b = \tau`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import ForceTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Arturo Laurenzi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FloatingBaseForceTask(ForceTask):
    r"""Floating base Force Task

    From [1]: "this implements a task which maps forces acting on the floating base virtual chain to contacts".

    .. math:: || J(q)[:,:6]^\top w - \tau ||^2,

    where :math:`w \in \mathbb{R}^{6N_c}` are the wrench vector being optimized (with :math:`N_c` being the number of
    contacts), :math:`J(q) = [J(q)_1^\top \cdot J(q)_{N_c}^\top]^\top \in \mathbb{R}^{6N_c \times 6 + N}` are the
    concatenated jacobians, and :math:`\tau` are the torques applied on the floating base.

    Note that this task assumes the robot has a floating base.

    The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = J(q)[:,:6]^\top`, :math:`x = w`, and :math:`b = \tau`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, contact_links, floating_base_torque=0., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            contact_links (list[str], list[int]): list of unique contact link names or ids.
            floating_base_torque (float, np.array[float[6]]): external torque applied on the floating base.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(FloatingBaseForceTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # check if model has floating base
        if not self.model.has_floating_base():
            raise ValueError("Expecting the given robotic 'model' to have a floating base, but it seems this is not "
                             "the case...")

        # set variables
        self.contact_links = contact_links
        self.floating_base_torque = floating_base_torque

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def contact_links(self):
        """Get the contact links."""
        return self._contact_links

    @contact_links.setter
    def contact_links(self, contacts):
        """Set the contact links."""
        if contacts is None:
            contacts = []
        elif not isinstance(contacts, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'contact_links' to be a list of names/ids, but got instead: "
                            "{}".format(type(contacts)))
        self._contact_links = contacts

        # enable / disable the tasks based on the number of contact links
        if len(contacts) == 0:
            self.disable()
        else:
            self.enable()

    @property
    def floating_base_torque(self):
        """Get the floating base torques."""
        return self._floating_base_torque

    @floating_base_torque.setter
    def floating_base_torque(self, torque):
        """Set the floating base torque."""
        if not isinstance(torque, (int, float)):
            if not isinstance(torque, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given 'floating_base_torque' to be an int, float, list/tuple/np.array "
                                "of float, but instead got: {}".format(type(torque)))
            torque = np.asarray(torque).reshape(-1)
            if len(torque) != 6:
                raise ValueError("Expecting the given 'floating_base_torque' to be of size 6, but got a size of: "
                                 "{}".format(len(torque)))
        self._floating_base_torque = torque

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        jacobians = [self.model.get_jacobian(self.model.get_link_id(link))[:6, :6] for link in self.contact_links]
        jacobians = np.concatenate(jacobians)  # shape (6*N_c,6)
        self._A = jacobians.T  # shape (6,6*N_c)
        self._b = self.floating_base_torque  # shape (6,)
