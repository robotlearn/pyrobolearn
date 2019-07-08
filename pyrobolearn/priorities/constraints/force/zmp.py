#!/usr/bin/env python
r"""Provide the Zero-Moment Point constraint.


References:
    - [1] "Motion Planning and Control of Dynamics Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Songyan Xin"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ZMPConstraint(Constraint):
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

        d_x^{-} \leq \frac{n^i_y}{f^i_z} \leq d_x^{+} \\
        d_y^{-} \leq -\frac{n^i_x}{f^i_z} \leq d_y^{+}

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

    References:
        - [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
        Implications", Popovic et al., 2005
        - [2] "Biped Walking Pattern Generation by using Preview Control of ZMP", Kajita et al., 2003
        - [3] "Exploiting Angular Momentum to Enhance Bipedal Center-of-Mass Control", Hofmann et al., 2009
    """

    def __init__(self, model):
        super(ZMPConstraint, self).__init__(model)
