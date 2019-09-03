#!/usr/bin/env python
r"""Provide the contact task.

The contact task tries to minimize the movement of a contact link:

.. math:: || C J_c(q) \dot{q} ||^2,

where :math:`C \in \mathbb{R}^{6 \times 6}` is the contact matrix (=a diagonal selector matrix where the entries
are 1 for cartesian velocities that we wish to minimize such that the link doesn't move, and 0 for cartesian
velocities that are free to change), :math:`J_c(q)` is the contact Jacobian (the Jacobian from the world frame to
the contact point), and :math:`\dot{q}` are the joint velocities being optimized.

This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A = C J_c(q)`,
:math:`x = \dot{q}`, and :math:`b = 0`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ContactTask(JointVelocityTask):
    r"""Contact Task

    The contact task tries to minimize the movement of a contact link:

    .. math:: || C J_c(q) \dot{q} ||^2,

    where :math:`C \in \mathbb{R}^{6 \times 6}` is the contact matrix (=a diagonal selector matrix where the entries
    are 1 for cartesian velocities that we wish to minimize such that the link doesn't move, and 0 for cartesian
    velocities that are free to change), :math:`J_c(q)` is the contact Jacobian (the Jacobian from the world frame to
    the contact point expressed in the distal link frame (i.e. the link which is in contact)), and :math:`\dot{q}` are
    the joint velocities being optimized.
    Note that because the jacobian is expressed in the distal link frame, the entries of the contact matrix specify
    the velocities that are fixed or can move in that particular frame. For instance, [0,0,1,0,0,0] means that the z
    linear velocity (where z is the z axis of the distal link frame) is fixed.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A = C J_c(q)`,
    :math:`x = \dot{q}`, and :math:`b = 0`.

    This task is useful for instance if we want to keep the feet in contact with the ground.
    
    
    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, contact_matrix=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            contact_matrix (np.array[float[6]], np.array[float[6,6]], None): contact selector matrix (=a diagonal
              square matrix). You can specify only the diagonal elements if you wish. If None, by default it will be
              set to the identity matrix.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(ContactTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set variable
        self.distal_link = self.model.get_link_id(distal_link)
        self.contact_matrix = contact_matrix

        # set QP vector
        self._b = np.zeros(6)

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def contact_matrix(self):
        """Return the contact selector matrix."""
        return self._contact_matrix

    @contact_matrix.setter
    def contact_matrix(self, matrix):
        """Set the contact selector matrix."""
        if matrix is None:
            matrix = np.identity(6)

        # check contact matrix type
        if not isinstance(matrix, (int, float, np.ndarray)):
            raise TypeError("Expecting the given contact matrix to be an int, float, or diagonal np.array, instead "
                            "got: {}".format(type(matrix)))

        # if numpy array, check its shape and make sure it is a diagonal matrix
        if isinstance(matrix, np.ndarray):
            if matrix.shape == (6,):
                matrix = np.diag(matrix)
            elif matrix.shape != (6, 6):
                raise ValueError("Expecting the given contact matrix to be of shape (6,6), instead got a shape of: "
                                 "{}".format(type(matrix)))
            else:
                # make sure the contact matrix is a diagonal matrix
                matrix = np.diag(np.diag(matrix))

        # set the contact matrix
        self._contact_matrix = matrix

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        # get jacobian expressed in the distal link frame
        jacobian = self.model.get_jacobian(link=self.distal_link, frame=self.distal_link)
        self._A = np.dot(self.contact_matrix, jacobian)
