#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the Zero-Moment Point constraint.

The ZMP constraints can be expressed as:

.. math::

    d_x^{-} \leq -\frac{n^i_y}{f^i_z} \leq d_x^{+} \\
    d_y^{-} \leq \frac{n^i_x}{f^i_z} \leq d_y^{+}

which ensures the stability of the foot/ground contact. The :math:`(d_x^{-}, d_x^{+})` and :math:`(d_y^{-}, d_y^{+})`
defines the size of the sole in the x and y directions respectively. Basically, this means that the ZMP point must be
inside the convex hull in order to have a static stability. The :math:`n^i` are the contact torques around the contact
point :math:`i`, and :math:`f` is the contact force at the contact point :math:`i`.

Notes:
    - the ZMP and CoP are equivalent for horizontal ground surfaces. For irregular ground surfaces they are
    distinct. [2]


QP formulation
--------------

The ZMP constraints given by:

.. math::

    d_x^{-} \leq -\frac{n^i_y}{f^i_z} \leq d_x^{+} \\
    d_y^{-} \leq \frac{n^i_x}{f^i_z} \leq d_y^{+}

can be rewritten as:

.. math::

    d_x^{-} f^i_z + n^i_y \leq 0 \\
    -d_x^{+} f^i_z - n^i_y \leq 0 \\
    d_y^{-} f^i_z - n^i_x \leq 0 \\
    -d_y^{+} f^i_z + n^i_x \leq 0

Thus, it can be rewritten as the inequality constraint :math:`G x \leq h` in QP, with:

.. math::

    G = \left[\begin{array}{cccccc}
        0 & 0 & d_x^{-} & 0 & 1 & 0  \\
        0 & 0 & -d_x^{+} & 0 & -1 & 0  \\
        0 & 0 & d_y^{-} & -1 & 0 & 0  \\
        0 & 0 & -d_y^{+} & 1 & 0 & 0  \\
    \end{array}\right],

:math:`x = [f^i_x, f^i_y, f^i_z, n^i_x, n^i_y, n^i_z]` being the optimized variables, and :math:`h = [0,0,0,0]`.

References:
    - [1] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
    - [2] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
        Implications", Popovic et al., 2005
"""

# TODO: correct this file when there are multiple links

import numpy as np
from scipy.linalg import block_diag

from pyrobolearn.priorities.constraints.constraint import UpperUnilateralConstraint, ForceConstraint
from pyrobolearn.utils.transformation import get_matrix_from_quaternion

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Songyan Xin"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ZMPConstraint(UpperUnilateralConstraint, ForceConstraint):
    r"""Zero-Moment Point constraint.

    "The ZMP is the point on the ground surface about which the horizontal component of the moment of ground
    reaction force is zero. It resolves the ground reaction force distribution to a single point." [1]

    Assumptions: the contact area is planar and has sufficiently high friction to keep the feet from sliding.

    .. math::

        x_{ZMP} &= x_{CoM} - \frac{F_x}{F_z + Mg} z_{CoM} - \frac{\tau_{y}(\vec{r}_{CoM})}{F_z + Mg} \\
        y_{ZMP} &= y_{CoM} - \frac{F_y}{F_z + Mg} z_{CoM} + \frac{\tau_{x}(\vec{r}_{CoM})}{F_z + Mg}

    where :math:`[x_{CoM}, y_{CoM}, z_{CoM}]` is the center of mass position, :math:`M` is the body mass,
    :math:`g` is the gravity value, :math:`F = Ma_{CoM}` is the net force acting on the whole body (including the
    gravity force :math:`-Mg`), :math:`\vec{r}_{CoM}` is the body center of mass, and :math:`\tau(\vec{r}_{CoM})`
    is the net whole-body moment about the center of mass.

    In the case where there are only ground reaction forces (+ the gravity force) acting on the robot, then the
    ZMP point is given by [3]:

    .. math::

        x_{ZMP} &= x_{CoM} - \frac{F_{G.R.X}}{F_{G.R.Z}} z_{CoM} - \frac{\tau_{y}(\vec{r}_{CoM})}{F_{G.R.Z}} \\
        y_{ZMP} &= y_{CoM} - \frac{F_{G.R.Y}}{F_{G.R.Z}} z_{CoM} + \frac{\tau_{x}(\vec{r}_{CoM})}{F_{G.R.Z}}

    where :math:`F_{G.R}` are the ground reaction forces, and the net moment about the CoM
    :math:`\tau(\vec{r}_{CoM})` is computed using the ground reaction forces.

    The ZMP constraints can be expressed as:

    .. math::

        d_x^{-} \leq -\frac{n^i_y}{f^i_z} \leq d_x^{+} \\
        d_y^{-} \leq \frac{n^i_x}{f^i_z} \leq d_y^{+}

    which ensures the stability of the foot/ground contact. The :math:`(d_x^{-}, d_x^{+})` and
    :math:`(d_y^{-}, d_y^{+})` defines the size of the sole in the x and y directions respectively. Basically,
    this means that the ZMP point must be inside the convex hull in order to have a static stability.
    The :math:`n^i` are the contact spatial torques around the contact point :math:`i`, and :math:`f` is the
    contact spatial force at the contact point :math:`i`.

    Notes:
        - the ZMP and CoP are equivalent for horizontal ground surfaces. For irregular ground surfaces they are
        distinct. [1]
        - the FRI coincides with the ZMP when the foot is stationary. [1]
        - the CMP coincides with the ZMP, when the moment about the CoM is zero. [1]


    QP formulation
    --------------

    The ZMP constraints given by:

    .. math::

        d_x^{-} \leq -\frac{n^i_y}{f^i_z} \leq d_x^{+} \\
        d_y^{-} \leq \frac{n^i_x}{f^i_z} \leq d_y^{+}

    can be rewritten as:

    .. math::

        d_x^{-} f^i_z + n^i_y \leq 0 \\
        -d_x^{+} f^i_z - n^i_y \leq 0 \\
        d_y^{-} f^i_z - n^i_x \leq 0 \\
        -d_y^{+} f^i_z + n^i_x \leq 0

    Thus, it can be rewritten as the inequality constraint :math:`G x \leq h` in QP, with:

    .. math::

        G = \left[\begin{array}{cccccc}
            0 & 0 & d_x^{-} & 0 & 1 & 0  \\
            0 & 0 & -d_x^{+} & 0 & -1 & 0  \\
            0 & 0 & d_y^{-} & -1 & 0 & 0  \\
            0 & 0 & -d_y^{+} & 1 & 0 & 0
        \end{array}\right],

    :math:`x = [f^i_x, f^i_y, f^i_z, n^i_x, n^i_y, n^i_z]` being the optimized variables, and :math:`h = [0,0,0,0]`.
    Note that the optimized variables are expressed in the world frame, and thus a rotation to express them in their
    contact local frame is performed beforehand.


    The implementation is based from insights provided in [4].

    References:
        - [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
        Implications", Popovic et al., 2005
        - [2] "Biped Walking Pattern Generation by using Preview Control of ZMP", Kajita et al., 2003
        - [3] "Exploiting Angular Momentum to Enhance Bipedal Center-of-Mass Control", Hofmann et al., 2009
        - [4] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
    """

    def __init__(self, model, x_bounds, y_bounds, link):  # TODO: correct when multiple links
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            x_bounds (tuple[2*float]): lower and upper bound of the size of the sole in the x direction.
            y_bounds (tuple[2*float]): lower and upper bound of the size of the sole in the y direction.
            link (int, str): unique link id or name.
        """
        super(ZMPConstraint, self).__init__(model)

        # set optimization variables
        self._b_upper_bound = np.zeros(4)
        self._zmp_matrix = np.zeros((4, 6))
        self._zmp_matrix[0, 4] = 1
        self._zmp_matrix[1, 4] = -1
        self._zmp_matrix[2, 3] = -1
        self._zmp_matrix[3, 3] = 1
        # the rest of A_ineq is set when setting the bounds

        # set variables
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self._link = self.model.get_link_id(link)

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def x_bounds(self):
        """Get the sole size bounds in the x-direction."""
        return self._x_bounds

    @x_bounds.setter
    def x_bounds(self, bounds):
        """Set the sole size bounds in the x-direction."""
        if not isinstance(bounds, (tuple, list, np.ndarray)):
            raise TypeError("Expecting the given 'x_bounds' to be a tuple, list or np.array of 2 float, but got "
                            "instead: {}".format(bounds))
        if len(bounds) != 2:
            raise ValueError("Expecting the given 'x_bounds' to be of size 2, but got a size of "
                             "{}".format(len(bounds)))
        self._x_bounds = bounds
        self._zmp_matrix[0, 2] = bounds[0]
        self._zmp_matrix[1, 2] = -bounds[1]

    @property
    def y_bounds(self):
        """Get the sole size bounds in the y-direction."""
        return self._y_bounds

    @y_bounds.setter
    def y_bounds(self, bounds):
        """Set the sole size bounds in the y-direction."""
        if not isinstance(bounds, (tuple, list, np.ndarray)):
            raise TypeError("Expecting the given 'y_bounds' to be a tuple, list or np.array of 2 float, but got "
                            "instead: {}".format(bounds))
        if len(bounds) != 2:
            raise ValueError("Expecting the given 'y_bounds' to be of size 2, but got a size of "
                             "{}".format(len(bounds)))
        self._y_bounds = bounds
        self._zmp_matrix[2, 2] = bounds[0]
        self._zmp_matrix[3, 2] = -bounds[1]

    # @property
    # def contacts(self):
    #     """Get the list of contact links."""
    #     return self._contacts
    #
    # @contacts.setter
    # def contacts(self, contacts):
    #     """Set the contact links."""
    #     if not isinstance(contacts, (list, tuple, np.ndarray)):
    #         raise TypeError("Expecting the given 'contacts' to be a list of int/str, but instead got: "
    #                         "{}".format(type(contacts)))
    #     self._contacts = contacts

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the upper unilateral inequality matrix and vector."""
        rot = get_matrix_from_quaternion(self.model.get_orientation(self._link)).T
        rot = block_diag((rot, rot))
        self._A_ineq = self._zmp_matrix.dot(rot)  # (4,6)
