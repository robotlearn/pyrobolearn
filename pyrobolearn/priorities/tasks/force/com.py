#!/usr/bin/env python
r"""Provide the center of mass force task.

From the documentation of the framework of [1]: "The CoM task computes the wrenches at the contact, in world frame,
in order to realize a certain acceleration and variation of angular momentum at the CoM considering the Centroidal
Dynamics":

.. math::

    m * \ddot{r} = \sum_i f_i + mg \\
    \dot{L} = \sum_i p_i \times f_i + \tau,

where :math:`w = [f \tau] \in \mathbb{R}^6` is the wrench vector composed of a force vector
:math:`f \in \mathbb{R}^3` and a torque vector :math:`\tau \in \mathbb{R}^3`, :math:`m` is the mass, :math:`r` is
the CoM position, :math:`g` is the gravity vector, :math:`L` is the angular momentum around the CoM, :math:`p` is
the position vector of where the wrench is applied (with respect to the CoM), and the subscript :math:`i` is to
denote each link where a wrench is applied to it (by contact).

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this

import numpy as np

from pyrobolearn.priorities.tasks import ForceTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoMForceTask(ForceTask):
    r"""CoM Force Task

    From the documentation of the framework of [1]: "The CoM task computes the wrenches at the contact, in world frame,
    in order to realize a certain acceleration and variation of angular momentum at the CoM considering the Centroidal
    Dynamics":

    .. math::

        m * \ddot{r} = \sum_i f_i + mg \\
        \dot{L} = \sum_i p_i \times f_i + \tau,

    where :math:`w = [f \tau] \in \mathbb{R}^6` is the wrench vector composed of a force vector
    :math:`f \in \mathbb{R}^3` and a torque vector :math:`\tau \in \mathbb{R}^3`, :math:`m` is the mass, :math:`r` is
    the CoM position, :math:`g` is the gravity vector, :math:`L` is the angular momentum around the CoM, :math:`p` is
    the position vector of where the wrench is applied (with respect to the CoM), and the subscript :math:`i` is to
    denote each link where a wrench is applied to it (by contact).

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, contact_links=[], wrenches=[], weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            contact_links (list[str], list[int]): list of unique contact link names or ids.
            wrenches (list[np.array[float[6]]]): list of associated wrenches applied to the contact links. It
              must have the same size as the number of contact links.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(CoMForceTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set variables
        self.contact_links = contact_links
        self.wrenches = wrenches

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
    def wrenches(self):
        """Get the wrenches."""
        return self._wrenches

    @wrenches.setter
    def wrenches(self, wrenches):
        """Set the wrenches."""
        if wrenches is None:
            wrenches = []
        elif not isinstance(wrenches, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'wrenches' to be a list of wrench vectors, but got instead: "
                            "{}".format(type(wrenches)))
        if isinstance(wrenches, np.ndarray) and wrenches.ndim == 1:
            wrenches = wrenches.reshape(-1, 6)
        self._wrenches = wrenches

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        x = self.model.get_com_position()
        dx = self.model.get_com_velocity()
        A_G = self.model.get_centroidal_momentum_matrix()

        angular_momentum = A_G[:3, :3]

        raise NotImplementedError



