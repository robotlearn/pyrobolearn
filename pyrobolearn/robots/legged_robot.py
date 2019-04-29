#!/usr/bin/env python
"""Provide the Legged robot abstract classes.

Classes that are defined here: LeggedRobot, BipedRobot, QuadrupedRobot, HexapodRobot.
"""

import os
import collections
import numpy as np
from scipy.spatial import ConvexHull

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LeggedRobot(Robot):
    r"""Legged robot

    Legged robots are robots that use some end-effectors to move itself. The movement pattern of these end-effectors
    in the standard regime are rhythmic movements.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.,
                 foot_frictions=None):
        super(LeggedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling=scaling)

        # leg and feet ids
        self.legs = []  # list of legs where a leg is a list of links
        self.feet = []  # list of feet ids

        # set the foot frictions
        if foot_frictions is not None:
            self.set_foot_friction(foot_frictions)

    ##############
    # Properties #
    ##############

    @property
    def num_legs(self):
        """Return the number of legs"""
        return len(self.legs)

    @property
    def num_feet(self):
        """Return the number of feet; this should normally be equal to the number of legs"""
        return len(self.feet)

    ###########
    # Methods #
    ###########

    def get_leg_ids(self, legs=None):
        """
        Return the leg id associated with the given leg index(ices)/name(s).

        Args:
            legs (int, str): leg index(ices) which is [0..num_legs()], or leg name(s)

        Returns:
            int, list[int]: leg id(s)
        """
        if legs is not None:
            if isinstance(legs, int):
                return self.legs[legs]
            elif isinstance(legs, str):
                return self.legs[self.get_link_ids(legs)]
            elif isinstance(legs, (list, tuple)):
                leg_ids = []
                for leg in legs:
                    if isinstance(leg, int):
                        leg_ids.append(self.legs[leg])
                    elif isinstance(leg, str):
                        leg_ids.append(self.legs[self.get_link_ids(leg)])
                    else:
                        raise TypeError("Expecting a str or int for items in legs")
                return leg_ids
        return self.legs

    def get_feet_ids(self, feet=None):
        """
        Return the foot id associated with the given foot index(ices)/name(s).

        Args:
            feet (int, str): foot index(ices) which is [0..num_feet()], or foot name(s)

        Returns:
            int, list[int]: foot id(s)
        """
        if feet is not None:
            if isinstance(feet, int):
                return self.feet[feet]
            elif isinstance(feet, str):
                return self.feet[self.get_link_ids(feet)]
            elif isinstance(feet, (list, tuple)):
                foot_ids = []
                for foot in feet:
                    if isinstance(foot, int):
                        foot_ids.append(self.feet[foot])
                    elif isinstance(foot, str):
                        foot_ids.append(self.feet[self.get_link_ids(foot)])
                    else:
                        raise TypeError("Expecting a str or int for items in feet")
                return foot_ids
        return self.feet

    def set_foot_friction(self, frictions, feet_ids=None):
        """
        Set the foot friction in the simulator.

        Warnings: only available in the simulator.

        Args:
            frictions (float, list of float): friction value(s).
            feet_ids (int, list of int): list of foot/feet id(s).
        """
        if feet_ids is None:
            feet_ids = self.feet
        if isinstance(feet_ids, int):
            feet_ids = [feet_ids]
        if isinstance(frictions, (float, int)):
            frictions = frictions * np.ones(len(feet_ids))
        for foot_id, frict in zip(feet_ids, frictions):
            if isinstance(foot_id, int):
                self.sim.change_dynamics(self.id, foot_id, lateral_friction=frict)
            elif isinstance(foot_id, collections.Iterable):
                for idx in foot_id:
                    self.sim.change_dynamics(self.id, idx, lateral_friction=frict)
            else:
                raise TypeError("Expecting foot_id to be a list of int, or an int. Instead got: "
                                "{}".format(type(foot_id)))

    def center_of_pressure(self, floor_id=None):
        r"""
        Center of Pressure (CoP).

        "The CoP is the point on the ground where the resultant of the ground-reaction force acts". [1]

        This is defined mathematically as:

        .. math::

            x_{CoP} = \frac{\sum_i x_i f^i_n}{\sum{i} f^i_n}
            y_{CoP} = \frac{\sum_i y_i f^i_n}{\sum{i} f^i_n}
            z_{CoP} = \frac{\sum_i z_i f^i_n}{\sum{i} f^i_n}

        where :math:`[x_i, y_i, z_i]` are the coordinates of the contact point :math:`i` on which the normal force
        :math:`f^i_n` acts.

        Notes:
            - the ZMP and CoP are equivalent for horizontal ground surfaces. For irregular ground surfaces they are
            distinct. [2]

        References:
            [1] "Postural Stability of Biped Robots and Foot-Rotation Index (FRI) Point", Goswami, 1999
            [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
            Implications", Popovic et al., 2005
        """
        if floor_id is not None:
            # get contact points between the robot's links and the floor
            points = self.sim.get_contact_points(body1=self.id, body2=floor_id)
            positions = np.array([point[6] for point in points])  # contact positions in world frame
            forces = np.array([point[9] for point in points]).reshape(-1, 1)  # normal force at contact points
            cop = forces * positions / np.sum(forces)
            return cop

        # check if there are force/pressure sensors at the links/joints
        raise NotImplementedError

    def zero_moment_point(self, update_com=False, use_simulator=False):
        r"""
        Zero Moment Point (ZMP).

        "The ZMP is the point on the ground surface about which the horizontal component of the moment of ground
        reaction force is zero. It resolves the ground reaction force distribution to a single point." [1]

        Assumptions: the contact area is planar and has sufficiently high friction to keep the feet from sliding.

        .. math::

            x_{ZMP} &= x_{CoM} - \frac{F_x}{F_z + Mg} z_{CoM} - \frac{\tau_{y}(\vec{r}_{CoM})}{F_z + Mg} \\
            y_{ZMP} &= y_{CoM} - \frac{F_y}{F_z + Mg} z_{CoM} + \frac{\tau_{x}(\vec{r}_{CoM})}{F_z + Mg}

        where :math:`[x_{CoM}, y_{CoM}, z_{CoM}]` is the center of mass position, :math:`F` is the net force acting
        on the whole body, :math:`M` is the body mass, :math:`g` is the gravity value, :math:`\vec{r}_{CoM}` is the
        body center of mass, and :math:`\tau(\vec{r}_{CoM})` is the net whole-body moment about the center of mass.

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
            [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
            Implications", Popovic et al., 2005
            [2] "Biped Walking Pattern Generation by using Preview Control of ZMP", Kajita et al., 2003
        """
        pass

    def foot_rotation_index(self):
        r"""
        Foot Rotation Index (FRI).

        "The FRI is the point (within or outside the support base) where the ground reaction force would have to act
        to keep the foot from accelerating. When the foot is stationary, the FRI coincides with the ZMP." [1]

        .. math::

            x_{FRI} &= \frac{x_f \dot{p}^f_z - z_f \dot{p}^f_x - x_{ZMP} F_{G.R.Z} - \dot{L}^f_y(\vec{r}_f)}
                {\dot{p}^f_z - F_{G.R.Z}} \\
            y_{FRI} &= \frac{y_f \dot{p}^F_z - z_f \dot{p}^f_y - y_{ZMP} F_{G.R.Z} - \dot{L}^f_x(\vec{r}_f)}
                {\dot{p}^f_z - F_{G.R.Z}}

        where :math:`\vec{p}^f` is the linear momentum of the foot's CoM, :math:`F_{G.R}` are the ground reaction
        forces, :math:`[x_f, y_f, z_f]` are the position coordinates of the foot, and :math:`L^f(\vec{r}_f)` is the
        net angular momentum of the foot around the foot.

        Notes:
            - the FRI coincides with the ZMP when the foot is stationary. [1]

        References:
            [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
            Implications", Popovic et al., 2005
        """
        pass

    def centroidal_moment_pivot(self, update_com=False, use_simulator=False):
        r"""
        Centroidal Moment Pivot (CMP).

        "The CMP is the point where the ground reaction force would have to act to keep the horizontal component of
        the whole-body angular momentum constant. When the moment about the CoM is zero, the CMP coincides with the
        ZMP." [1]

        .. math::

            x_{CMP} &= x_{CoM} - \frac{F_{G.R.X}}{F_{G.R.Z}} z_{CoM} \\
            y_{CMP} &= y_{CoM} - \frac{F_{G.R.Y}}{F_{G.R.Z}} z_{CoM}

        .. math::

            x_{CMP} &= x_{ZMP} + \frac{\tau_y(\vec{r}_{CoM})}{F_{G.R.Z}} \\
            y_{CMP} &= y_{ZMP} - \frac{\tau_x(\vec{r}_{CoM})}{F_{G.R.Z}}

        Notes:
            - the CMP coincides with the ZMP, when the moment about the CoM is zero. [1]

        References:
            [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
            Implications", Popovic et al., 2005
        """
        pass

    # def divergent_component_motion(self):
    #     r"""
    #     Divergent Component of Motion, a.k.a 'eXtrapolated Center of Mass'.
    #
    #     .. math:: \xi = x + b \dot{x}
    #
    #     where :math:`\xi = [\xi_x, \xi_y, \xi_z]` is the DCM point, :math:`x = [x,y,z]` and :math:`\dot{x} = [\dot{x},
    #     \dot{y}, \dot{z}]` are the CoM position and velocity, :math:`b > 0` is a time-constant of the DCM dynamics.
    #
    #     References:
    #         [1] "Three-dimensional Bipedal Walking Control Based on Divergent Component of Motion", Englsberger et
    #             al., 2015
    #     """
    #     pass

    # the following methods need to be overwritten in the children classes

    def move(self, velocity):
        """Move the robot at the specified velocity."""
        pass

    def walk_forward(self, speed):
        """Walk forward."""
        pass

    def walk_backward(self, speed):
        """Walk backward."""
        pass

    def walk_left(self, speed):
        """Walk sideways to the left."""
        pass

    def walk_right(self, speed):
        """Walk sideways to the right."""
        pass

    def turn_left(self, speed):
        """Turn left."""
        pass

    def turn_right(self, speed):
        """Turn right."""
        pass

    def draw_support_polygon(self, floor_id, lifetime=1.):  # TODO: improve this by remembering the previous hull
        r"""
        draw the support polygon / convex hull in the simulator.

        Warnings:
            - this is only valid in the simulator.
            - do not call this at a high frequency.

        Args:
            floor_id (int): id of the floor in the simulator.
            lifetime (float): lifetime of the support polygon before it disappears.

        References:
            [1] "A Universal Stability Criterion of the Foot Contact of Legged Robots- Adios ZMP"
        """
        # get contact points between the robot's links and the floor
        points = self.sim.get_contact_points(body1=self.id, body2=floor_id)
        # points = np.array([point[5] for point in points])  # contact position on robot in Cartesian world coordinates
        points = np.array([point[6] for point in points])  # contact position on floor in Cartesian world coordinates

        # compute convex hull
        if len(points) > 2:  # we need at least 3 points to construct the convex hull
            # compute convex hull
            hull = ConvexHull(points[:, :2])
            vertices = points[hull.vertices]  # get the vertices of the convex hull

            # draw support polygon
            for i in range(len(vertices)):
                self.sim.add_user_debug_line(from_pos=vertices[i-1], to_pos=vertices[i], rgb_color=(0, 1, 0), width=3,
                                             lifetime=lifetime)

    # TODO: correct, consider irregular terrain, update visual shape of cones
    def draw_friction_cone(self, floor_id, height=0.2):
        r"""
        Draw the friction cone.

        The friction cone is defined as:

        .. math:: C^i_s = {(f^i_x, f^i_y, f^i_z) \in \mathbb{R}^3 | \sqrt{(f^i_x)^2 + (f^i_y)^2} \leq \mu_i f^i_z }

        where :math:`i` denotes the ith support/contact, :math:`f^i_s` is the contact spatial force exerted at
        the contact point :math:`C_i`, and :math:`\mu_i` is the static friction coefficient at that contact point.

        "A point contact remains in the fixed contact mode while its contact force f^i lies inside the friction cone"
        [1]. Often, the friction pyramid which is the linear approximation of the friction cone is considered as it
        is easier to manipulate it; e.g. present it as a linear constraint in a quadratic optimization problem.

        Warnings:
            - this is only valid in the simulator.
            - do not call this at a high frequency.

        Args:
            floor_id (int): id of the floor in the simulator.
            height (float): maximum height of the cone in the simulator.

        References:
            [1] https://scaron.info/teaching/friction-cones.html
            [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
                for Rectangular Support Areas", Caron et al., 2015
        """
        filename = os.path.dirname(__file__) + '/../worlds/meshes/cone.obj'

        # get contact points between the robot's links and the floor
        points = self.sim.get_contact_points(body1=self.id, body2=floor_id)
        mu = self.sim.get_dynamics_info(floor_id)[1]  # friction coefficient

        ids = []
        for point in points:
            position = point[6]  # contact position on floor in Cartesian world coordinates
            fz_dir = point[7]    # contact normal on floor pointing towards the robot
            fz = point[9]        # normal force applied during the last step
            fy = point[10]       # lateral friction force in the first lateral friction direction
            fy_dir = point[11]   # first lateral friction direction
            fx = point[12]       # lateral friction force in the second lateral friction direction
            fx_dir = point[13]   # second lateral friction direction

            # make sure that fz is bigger than 0
            if not np.allclose(fz, 0):

                # rescale fx, fy, fz
                # TODO uncomment the original calculations
                fx = height  # np.abs(fx / (mu*fz)) * height
                fy = height  # np.abs(fy / (mu*fz)) * height
                fz = height

                position += np.array([0., 0., height * 0.5])
                id_ = self.sim.load_mesh(filename, position, orientation=(0, 1, 0, 0), mass=0.,
                                         scale=(fx, fy, fz), color=(0.5, 0., 0., 0.5), with_collision=False)
                ids.append(id_)

        return ids

    # TODO: add pyramid 3D object, consider irregular terrains, update pyramid visual shape
    def draw_friction_pyramid(self, floor_id, height=0.2):
        r"""
        Draw friction pyramid.

        The friction pyramid is defined as:

        .. math:: P^i_s = {(f^i_x, f^i_y, f^i_z) \in \mathbb{R}^3 | f^i_x \leq \mu_i f^i_z, f^i_y \leq \mu_i f^i_z}

        where where :math:`i` denotes the ith support/contact, :math:`f^i_s` is the contact spatial force exerted at
        the contact point :math:`C_i`, and :math:`\mu_i` is the static friction coefficient at that contact point.
        If the static friction coefficient is given by :math:`\frac{\mu_i}{\sqrt{2}}`, then we are making an inner
        approximation (i.e. the pyramid is inside the cone) instead of an outer approximation (i.e. the cone is inside
        the pyramid). [1]

        This linear approximation is often used as a linear constraint in a quadratic optimization problem along with
        the unilateral constraint :math:`f^i_z \geq 0`.

        Warnings:
            - this is only valid in the simulator.
            - do not call this at a high frequency.

        Args:
            floor_id (int): id of the floor in the simulator.
            height (float): maximum height of the pyramid in the simulator.

        References:
            [1] https://scaron.info/teaching/friction-cones.html
            [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
                for Rectangular Support Areas", Caron et al., 2015
        """
        filename = os.path.dirname(__file__) + '/../worlds/meshes/pyramid.obj'

        # get contact points between the robot's links and the floor
        points = self.sim.get_contact_points(body1=self.id, body2=floor_id)
        mu = self.sim.get_dynamics_info(floor_id)[1]  # friction coefficient

        ids = []
        for point in points:
            position = point[6]  # contact position on floor in Cartesian world coordinates
            fz_dir = point[7]  # contact normal on floor pointing towards the robot
            fz = point[9]  # normal force applied during the last step
            fy = point[10]  # lateral friction force in the first lateral friction direction
            fy_dir = point[11]  # first lateral friction direction
            fx = point[12]  # lateral friction force in the second lateral friction direction
            fx_dir = point[13]  # second lateral friction direction

            # make sure that fz is bigger than 0
            if not np.allclose(fz, 0):
                # rescale fx, fy, fz
                # TODO uncomment the original calculations
                fx = height  # np.abs(fx / (mu*fz)) * height
                fy = height  # np.abs(fy / (mu*fz)) * height
                fz = height

                position += np.array([0., 0., height * 0.5])
                id_ = self.sim.load_mesh(filename, position, orientation=(0, 1, 0, 0), mass=0.,
                                         scale=(fx, fy, fz), color=(0.5, 0., 0., 0.5), with_collision=False)
                ids.append(id_)

        return ids


class BipedRobot(LeggedRobot):
    r"""Biped Robot

    A biped robot is a robot which has 2 legs.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(BipedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_leg_id = 0
        self.right_leg_id = 1

    ##############
    # Properties #
    ##############

    @property
    def left_leg(self):
        """Return the left leg joint ids"""
        return self.legs[self.left_leg_id]

    @property
    def right_leg(self):
        """Return the right leg joint ids"""
        return self.legs[self.right_leg_id]

    @property
    def left_foot(self):
        """Return the left foot id"""
        return self.feet[self.left_leg_id]

    @property
    def right_foot(self):
        """Return the right foot id"""
        return self.feet[self.right_leg_id]


class QuadrupedRobot(LeggedRobot):
    r"""Quadruped robot

    A quadruped robot is a robot which has 4 legs.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(QuadrupedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_front_leg_id = 0
        self.right_front_leg_id = 1
        self.left_back_leg_id = 2
        self.right_back_leg_id = 3

    ##############
    # Properties #
    ##############

    @property
    def left_front_leg(self):
        """Return the left front leg joint ids"""
        return self.legs[self.left_front_leg_id]

    @property
    def right_front_leg(self):
        """Return the right front leg joint ids"""
        return self.legs[self.right_front_leg_id]

    @property
    def left_back_leg(self):
        """Return the left back leg joint ids"""
        return self.legs[self.left_back_leg_id]

    @property
    def right_back_leg(self):
        """Return the right back leg joint ids"""
        return self.legs[self.right_back_leg_id]

    @property
    def left_front_foot(self):
        """Return the left front foot id"""
        return self.feet[self.left_front_leg_id]

    @property
    def right_front_foot(self):
        """Return the right front foot id"""
        return self.feet[self.right_front_leg_id]

    @property
    def left_back_foot(self):
        """Return the left back foot id"""
        return self.feet[self.left_back_leg_id]

    @property
    def right_back_foot(self):
        """Return the right back foot id"""
        return self.feet[self.right_back_leg_id]


class HexapodRobot(LeggedRobot):
    r"""Hexapod Robot

    An hexapod robot is a robot which has 6 legs.
    """

    def __init__(self, simulator, urdf, position, orientation=None, fixed_base=False, scaling=1.):
        super(HexapodRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_front_leg_id = 0
        self.right_front_leg_id = 1
        self.left_middle_leg_id = 2
        self.right_middle_leg_id = 3
        self.left_back_leg_id = 4
        self.right_back_leg_id = 5

    ##############
    # Properties #
    ##############

    @property
    def left_front_leg(self):
        """Return the left front leg ids"""
        return self.legs[self.left_front_leg_id]

    @property
    def right_front_leg(self):
        """Return the right front leg ids"""
        return self.legs[self.right_front_leg_id]

    @property
    def left_middle_leg(self):
        """Return the left middle leg ids"""
        return self.legs[self.left_middle_leg_id]

    @property
    def right_middle_leg(self):
        """Return the right middle leg ids"""
        return self.legs[self.right_middle_leg_id]

    @property
    def left_back_leg(self):
        """Return the left back leg ids"""
        return self.legs[self.left_back_leg_id]

    @property
    def right_back_leg(self):
        """Return the right back leg ids"""
        return self.legs[self.right_back_leg_id]

    @property
    def left_front_foot(self):
        """Return the left front foot id"""
        return self.feet[self.left_front_leg_id]

    @property
    def right_front_foot(self):
        """Return the right front foot id"""
        return self.feet[self.right_front_leg_id]

    @property
    def left_middle_foot(self):
        """Return the left middle foot id"""
        return self.feet[self.left_middle_leg_id]

    @property
    def right_middle_foot(self):
        """Return the right middle foot id"""
        return self.feet[self.right_middle_leg_id]

    @property
    def left_back_foot(self):
        """Return the left back foot id"""
        return self.feet[self.left_back_leg_id]

    @property
    def right_back_foot(self):
        """Return the right back foot id"""
        return self.feet[self.right_back_leg_id]
