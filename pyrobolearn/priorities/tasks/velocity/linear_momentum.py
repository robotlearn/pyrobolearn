# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the linear momentum task.

The is the linear part of the `CentroidalMomentumTask`. The centroidal momentum task tries to minimize the
difference between the desired and current centroidal moment given by:

.. math:: ||A_G \dot{q} - h_{G,d}||^2

where :math:`A_G \in \mathbb{R}^{6 \times N}` is the centroidal momentum matrix (CMM, see below for description),
:math:`\dot{q}` are the joint velocities being optimized, and :math:`h_{G,d}` is the desired centroidal momentum.

This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=A_G`, :math:`x=\dot{q}`,
and :math:`b = h_{G,d}`.

The centroidal momentum (which is the sum of all body spatial momenta computed wrt the CoM) is given by:

.. math:: h_G = A_G \dot{q}

where :math:`h_G = [k_G^\top, l_G^\top]^\top \in \mathbb{R}^6` is the centroidal momentum (the subscript :math:`G`
denotes the CoM) with :math:`k_G \in \mathbb{R}^3` being the angular momentum and :math:`l_G \in \mathbb{R}^3` the
linear part, :math:`\dot{q}` are the joint velocities, and :math:`A_G \in \mathbb{R}^{6 \times N}` (with :math:`N`
is the number of DoFs) is the centroidal momentum matrix (CMM).

"The CMM is computed from the joint space inertia matrix :math:`H(q)`, given by:

.. math:: A_G = ^1X_G^\top S_1 H(q) = ^1X_G^\top H_1(q)

where :math:`^1X_G^\top \in \mathbb{R}^{6 \times 6}` is the spatial transformation matrix that transfers spatial
momentum from the floating base (Body 1) to the CoM (G), :math:`H(q)` is the full joint space inertia matrix,
:math:`H_1 = S_1 H` is the floating base (Body 1) inertia matrix selected using the selector matrix
:math:`S_1 = [1_{6 \times 6}, 0_{6 \times (N-6)}}`.

The spatial transformation matrix is given by:

.. math::

    ^1X_G^\top = \left[ \begin{array}{cc}
        ^GR_1 &  ^GR_1 S(^1p_G)^\top \\
        0     &  ^GR_1
        \\end{array} \right]

where :math:`^GR_1` is the rotation matrix of :math:`G` wrt the floating base (Body 1),
:math:`^1p_G = ^1R_0 (^0p_G - ^0p_1)` is the position vector from the floating base (Body 1) origin to the CoM
expressed in the floating base (Body 1) frame, :math:`S(\cdot)` provides the skew symmetric cross product matrix
such that :math:`S(p)v = p \cross v`. Note that the orientation of Frame :math:`G` (CoM) could be selected to be
parallel to the ground inertial (i.e. world) frame then the rotation matrix :math:`^GR_1 = ^0R_1`." [2]


The centroidal linear momentum task focuses on the linear part :math:`l_G \in \mathbb{R}^3` in the centroidal
momentum :math:`h_G = [k_G^\top, l_G^\top]^\top \in \mathbb{R}^6`.

The implementation of this class is inspired by [1, 2] (where [1] is licensed under the LGPLv2)

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Pouya Mohammadi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinearMomentumTask(JointVelocityTask):
    r"""(Centroidal) Linear Momentum Task

    The is the linear part of the `CentroidalMomentumTask`. The centroidal momentum task tries to minimize the
    difference between the desired and current centroidal moment given by:

    .. math:: ||A_G \dot{q} - h_{G,d}||^2

    where :math:`A_G \in \mathbb{R}^{6 \times N}` is the centroidal momentum matrix (CMM, see below for description),
    :math:`\dot{q}` are the joint velocities being optimized, and :math:`h_{G,d}` is the desired centroidal momentum.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=A_G`, :math:`x=\dot{q}`,
    and :math:`b = h_{G,d}`.

    The centroidal momentum (which is the sum of all body spatial momenta computed wrt the CoM) is given by:

    .. math:: h_G = A_G \dot{q}

    where :math:`h_G = [k_G^\top, l_G^\top]^\top \in \mathbb{R}^6` is the centroidal momentum (the subscript :math:`G`
    denotes the CoM) with :math:`k_G \in \mathbb{R}^3` being the angular momentum and :math:`l_G \in \mathbb{R}^3` the
    linear part, :math:`\dot{q}` are the joint velocities, and :math:`A_G \in \mathbb{R}^{6 \times N}` (with :math:`N`
    is the number of DoFs) is the centroidal momentum matrix (CMM).

    "The CMM is computed from the joint space inertia matrix :math:`H(q)`, given by:

    .. math:: A_G = ^1X_G^\top S_1 H(q) = ^1X_G^\top H_1(q)

    where :math:`^1X_G^\top \in \mathbb{R}^{6 \times 6}` is the spatial transformation matrix that transfers spatial
    momentum from the floating base (Body 1) to the CoM (G), :math:`H(q)` is the full joint space inertia matrix,
    :math:`H_1 = S_1 H` is the floating base (Body 1) inertia matrix selected using the selector matrix
    :math:`S_1 = [1_{6 \times 6}, 0_{6 \times (N-6)}}`.

    The spatial transformation matrix is given by:

    .. math::

        ^1X_G^\top = \left[ \begin{array}{cc}
            ^GR_1 &  ^GR_1 S(^1p_G)^\top \\
            0     &  ^GR_1
            \\end{array} \right]

    where :math:`^GR_1` is the rotation matrix of :math:`G` wrt the floating base (Body 1),
    :math:`^1p_G = ^1R_0 (^0p_G - ^0p_1)` is the position vector from the floating base (Body 1) origin to the CoM
    expressed in the floating base (Body 1) frame, :math:`S(\cdot)` provides the skew symmetric cross product matrix
    such that :math:`S(p)v = p \cross v`. Note that the orientation of Frame :math:`G` (CoM) could be selected to be
    parallel to the ground inertial (i.e. world) frame then the rotation matrix :math:`^GR_1 = ^0R_1`." [3]

    The centroidal linear momentum task focuses on the linear part :math:`l_G \in \mathbb{R}^3` in the centroidal
    momentum :math:`h_G = [k_G^\top, l_G^\top]^\top \in \mathbb{R}^6`.


    The implementation of this class is inspired by [3, 4] (where [4] is licensed under the LGPLv2)

    References:
        - [1] "Centroidal momentum matrix of a humanoid robot: structure and properties", Orin et al., 2008
        - [2] "Centroidal dynamics of a humanoid robot", Orin et al., 2013
        - [3] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Xin, 2018
        - [4] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, desired_linear_momentum=None, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            desired_linear_momentum (np.array[float[3]], None): desired centroidal linear momentum. If None, it
              will be set to zero.
            weight (float, np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(LinearMomentumTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define desired reference
        self.desired_linear_momentum = desired_linear_momentum

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def desired_linear_momentum(self):
        """Get the desired centroidal linear momentum."""
        return self._l_desired

    @desired_linear_momentum.setter
    def desired_linear_momentum(self, l_d):
        """Set the desired centroidal linear momentum."""
        if l_d is None:
            l_d = np.zeros(3)
        if not isinstance(l_d, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired centroidal linear momentum to be an instance of np.array, "
                            "instead got: {}".format(type(l_d)))
        l_d = np.asarray(l_d)
        if len(l_d) != 3:
            raise ValueError("Expecting the length of the given desired linear centroidal momentum to be of length 3, "
                             "instead got: {}".format(len(l_d)))
        self._l_desired = l_d

    @property
    def x_desired(self):
        """Get the desired centroidal linear momentum."""
        return self._l_desired

    @x_desired.setter
    def x_desired(self, l_d):
        """Set the desired centroidal linear momentum."""
        self.desired_linear_momentum = l_d

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[3]], None): desired centroidal linear momentum.
        """
        self.x_desired = x_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[float[3]]: desired centroidal linear momentum.
        """
        return self.x_desired

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        self._A = self.model.get_centroidal_momentum_matrix()[3:]  # shape: (3, N)
        self._b = self._l_desired  # shape: (3,)
