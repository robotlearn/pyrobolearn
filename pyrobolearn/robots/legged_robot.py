#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LeggedRobot(Robot):
    r"""Legged robot

    Legged robots are robots that use some end-effectors to move itself. The movement pattern of these end-effectors
    in the standard regime are rhythmic movements.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.,
                 foot_frictions=None):
        """
        Initialize the Legged robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            foot_frictions (float, list of float): foot friction value(s).
        """
        super(LeggedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale=scale)

        # leg and feet ids
        self.legs = []  # list of legs where a leg is a list of links
        self.feet = []  # list of feet ids

        # set the foot frictions
        if foot_frictions is not None:
            self.set_foot_friction(foot_frictions)

        # visual debug
        self.cop_visual = None  # visual sphere for center of pressure
        self.zmp_visual = None  # visual sphere for zero-moment point
        self.fri_visual = None  # visual sphere for foot rotation index
        self.cmp_visual = None  # visual sphere for centroidal moment pivot

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
        Compute and return the center of Pressure (CoP).

        "The CoP is the point on the ground where the resultant of the ground-reaction force acts". [1]

        This is defined mathematically as:

        .. math::

            x_{CoP} = \frac{\sum_i x_i f^i_n}{\sum_i f^i_n}
            y_{CoP} = \frac{\sum_i y_i f^i_n}{\sum_i f^i_n}
            z_{CoP} = \frac{\sum_i z_i f^i_n}{\sum_i f^i_n}

        where :math:`[x_i, y_i, z_i]` are the coordinates of the contact point :math:`i` on which the normal force
        :math:`f^i_n` acts.

        Notes:
            - the ZMP and CoP are equivalent for horizontal ground surfaces. For irregular ground surfaces they are
            distinct. [2]

        Args:
            floor_id (int, None): id of the floor in the simulator. If None, it will use the force/pressure sensors.

        Returns:
            np.array[float[3]], None: center of pressure. None if the robot is not in contact with the ground.

        References:
            - [1] "Postural Stability of Biped Robots and Foot-Rotation Index (FRI) Point", Goswami, 1999
            - [2] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
              Implications", Popovic et al., 2005
        """
        if floor_id is not None:

            cop_key = 'cop_' + str(floor_id)

            # checked if already cached
            if cop_key in self._state:
                return self._state[cop_key]

            # get contact points between the robot's links and the floor
            points = self.sim.get_contact_points(body1=self.id, body2=floor_id)

            # if no contact points
            if len(points) == 0:
                return None

            # compute contact positions (in world frame) and normal force at these points
            positions = np.array([point[6] for point in points])  # contact positions in world frame
            forces = np.array([point[9] for point in points]).reshape(-1, 1)  # normal force at contact points

            # compute CoP and return it
            cop = forces * positions / np.sum(forces)
            cop = np.sum(cop, axis=0)

            # cache it
            self._state[cop_key] = cop

            return cop

        # check if there are force/pressure sensors at the links/joints
        raise NotImplementedError

    def zero_moment_point(self, update_com=False, floor_id=None):
        r"""
        Zero Moment Point (ZMP).

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

        Args:
            update_com (bool): if True, it will compute and update the CoM position.
            floor_id (int, None): id of the floor in the simulator. If None, it will use the force/pressure sensors.

        Returns:
            np.array[float[3]], None: zero-moment point. None if the ground reaction force in z is 0.

        References:
            - [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
              Implications", Popovic et al., 2005
            - [2] "Biped Walking Pattern Generation by using Preview Control of ZMP", Kajita et al., 2003
            - [3] "Exploiting Angular Momentum to Enhance Bipedal Center-of-Mass Control", Hofmann et al., 2009
        """
        # if we need to update the CoM
        if update_com:
            self.com = self.get_center_of_mass_position()

        # if the floor id is given, use the simulator to compute the ZMP (using the contact points)
        if floor_id is not None:

            zmp_key = 'zmp_' + str(floor_id)

            # checked if already cached
            if zmp_key in self._state:
                return self._state[zmp_key]

            # get contact points between the robot's links and the floor
            points = self.sim.get_contact_points(body1=self.id, body2=floor_id)

            # if no contact points
            if len(points) == 0:
                return None

            # compute contact positions in world frame
            positions = np.array([point[6] for point in points])

            # get all the ground reaction forces
            forces_z = np.array([point[9] * point[7] for point in points])  # normal force
            forces_y = np.array([point[10] * point[11] for point in points])  # first lateral friction force
            forces_x = np.array([point[12] * point[13] for point in points])  # second lateral friction force
            forces = forces_x + forces_y + forces_z  # ground reaction forces

            # compute all the moments with respect to the CoM
            moments = np.cross(positions - self.com, forces)

            # sum all the ground reaction forces and moments
            forces = np.sum(forces, axis=0)
            moments = np.sum(moments, axis=0)

            # if no ground reaction forces in z, return None
            if np.isclose(forces[2], 0):
                return None

            # compute ZMP
            zmp = np.copy(self.com)
            zmp[2] = np.mean(positions, axis=0)[2]

            zmp[0] += -forces[0]/forces[2] * self.com[2] - moments[1]/forces[2]
            zmp[1] += -forces[1]/forces[2] * self.com[2] + moments[0]/forces[2]

            # cache it
            self._state[zmp_key] = zmp

            # return ZMP
            return zmp

        # check if there are force/pressure sensors at the links/joints
        raise NotImplementedError

    def foot_rotation_indicator(self):
        r"""
        Foot Rotation Indicator (FRI).

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
            - [1] "Postural Stability of Biped Robots and the Foot-Rotation Indicator (FRI) Point", Goswami, 1999
            - [2] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
              Implications", Popovic et al., 2005
        """
        raise NotImplementedError

    def centroidal_moment_pivot(self, update_com=False, floor_id=None):
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

        Args:
            update_com (bool): if True, it will compute and update the CoM position.
            floor_id (int, None): id of the floor in the simulator. If None, it will use the force/pressure sensors.

        Returns:
            np.array[float[3]], None: centroidal moment pivot point. None if the ground reaction force in z is 0.

        References:
            - [1] "Ground Reference Points in Legged Locomotion: Definitions, Biological Trajectories and Control
              Implications", Popovic et al., 2005
        """
        # update the CoM
        if update_com:
            self.get_center_of_mass_position()

        if floor_id is not None:

            cmp_key = 'cmp_' + str(floor_id)

            # checked if already cached
            if cmp_key in self._state:
                return self._state[cmp_key]

            # get contact points between the robot's links and the floor
            points = self.sim.get_contact_points(body1=self.id, body2=floor_id)

            # if no contact points
            if len(points) == 0:
                return None

            # compute contact positions in world frame
            positions = np.array([point[6] for point in points])

            # get all the ground reaction forces
            forces_z = np.array([point[9] * point[7] for point in points])  # normal force
            forces_y = np.array([point[10] * point[11] for point in points])  # first lateral friction force
            forces_x = np.array([point[12] * point[13] for point in points])  # second lateral friction force
            forces = forces_x + forces_y + forces_z  # ground reaction forces
            forces = np.sum(forces, axis=0)  # sum all the ground reaction forces

            # if no ground reaction forces in z, return None
            if np.isclose(forces[2], 0):
                return None

            # compute CMP
            cmp = np.copy(self.com)
            cmp[2] = np.mean(positions, axis=0)[2]
            cmp[0] -= forces[0] / forces[2] * self.com[2]
            cmp[1] -= forces[1] / forces[2] * self.com[2]

            # cache it
            self._state[cmp_key] = cmp

            # return CMP
            return cmp

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
    #         - [1] "Three-dimensional Bipedal Walking Control Based on Divergent Component of Motion", Englsberger et
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
            - [1] "A Universal Stability Criterion of the Foot Contact of Legged Robots- Adios ZMP"
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
            - [1] https://scaron.info/teaching/friction-cones.html
            - [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
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
            - [1] https://scaron.info/teaching/friction-cones.html
            - [2] "Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone
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

    def draw_cop(self, cop=None, radius=0.05, color=(0, 1, 0, 0.8)):
        """
        Draw the CoP in the simulator.

        Args:
            cop (np.array[float[3]], None, int): center of pressure. If None or int, it will compute the CoP. If None, it
                will compute it using the force sensors. If int, it will be assumed to be the floor's id, and will
                use the simulator to compute the CoP.
            radius (float): radius of the sphere representing the CoP of the robot
            color (tuple of 4 floats): rgba color of the sphere (each value is between 0 and 1). By default it is red.
        """
        if cop is None or isinstance(cop, (int, long)):
            cop = self.center_of_pressure(floor_id=cop)
        if self.cop_visual is None and cop is not None:  # create visual shape if not already created
            cop_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
            self.cop_visual = self.sim.create_body(mass=0, visual_shape_id=cop_visual_shape, position=cop)
        else:  # set CoP position
            if cop is None:
                self.remove_cop()
            else:
                self.sim.reset_base_pose(self.cop_visual, cop, [0, 0, 0, 1])

    def draw_zmp(self, zmp=None, radius=0.05, color=(1, 1, 0, 0.8), update_com=False):
        """
        Draw the ZMP in the simulator.

        Args:
            zmp (np.array[float[3]], None, int): zero-moment point. If None or int, it will compute the ZMP. If None, it
                will compute it using the force sensors. If int, it will be assumed to be the floor's id, and will
                use the simulator to compute the ZMP.
            radius (float): radius of the sphere representing the ZMP of the robot
            color (float[4]): rgba color of the sphere (each value is between 0 and 1). By default it is red.
            update_com (bool): if we should compute the CoM, if None or int is given for the :attr:`zmp`.
        """
        if zmp is None or isinstance(zmp, (int, long)):
            zmp = self.zero_moment_point(update_com=update_com, floor_id=zmp)
        if self.zmp_visual is None and zmp is not None:  # create visual shape if not already created
            zmp_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
            self.zmp_visual = self.sim.create_body(mass=0, visual_shape_id=zmp_visual_shape, position=zmp)
        else:  # set ZMP position
            if zmp is None:
                self.remove_zmp()
            else:
                self.sim.reset_base_pose(self.zmp_visual, zmp, [0, 0, 0, 1])

    def draw_cmp(self, cmp=None, radius=0.05, color=(1, 0, 0, 0.8), update_com=False):
        """
        Draw the CMP in the simulator.

        Args:
            cmp (np.array[float[3]], None, int): central moment pivot. If None or int, it will compute the CMP. If None, it
                will compute it using the force sensors. If int, it will be assumed to be the floor's id, and will
                use the simulator to compute the CMP.
            radius (float): radius of the sphere representing the CMP of the robot
            color (float[4]): rgba color of the sphere (each value is between 0 and 1). By default it is red.
            update_com (bool): if we should compute the CoM, if None or int is given for the :attr:`cmp`.
        """
        if cmp is None or isinstance(cmp, (int, long)):
            cmp = self.centroidal_moment_pivot(update_com=update_com, floor_id=cmp)
        if self.cmp_visual is None and cmp is not None:  # create visual shape if not already created
            cmp_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
            self.cmp_visual = self.sim.create_body(mass=0, visual_shape_id=cmp_visual_shape, position=cmp)
        else:  # set ZMP position
            if cmp is None:
                self.remove_cmp()
            else:
                self.sim.reset_base_pose(self.cmp_visual, cmp, [0, 0, 0, 1])

    def draw_fri(self, fri=None, radius=0.05, color=(1, 0, 0, 0.8), update_com=False):
        """
        Draw the FRI in the simulator.

        Args:
            fri (np.array[float[3]], None, int): central moment pivot. If None or int, it will compute the FRI. If None, it
                will compute it using the force sensors. If int, it will be assumed to be the floor's id, and will
                use the simulator to compute the FRI.
            radius (float): radius of the sphere representing the FRI of the robot
            color (float[4]): rgba color of the sphere (each value is between 0 and 1). By default it is red.
            update_com (bool): if we should compute the CoM, if None or int is given for the :attr:`fri`.
        """
        if fri is None or isinstance(fri, (int, long)):
            fri = self.centroidal_moment_pivot(update_com=update_com, floor_id=fri)
        if self.fri_visual is None and fri is not None:  # create visual shape if not already created
            fri_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
            self.fri_visual = self.sim.create_body(mass=0, visual_shape_id=fri_visual_shape, position=fri)
        else:  # set FRI position
            if fri is None:
                self.remove_cmp()
            else:
                self.sim.reset_base_pose(self.fri_visual, fri, [0, 0, 0, 1])

    def remove_cop(self):
        """
        Remove the CoP from the simulator.
        """
        if self.cop_visual is not None:
            self.sim.remove_body(self.cop_visual)
            self.cop_visual = None

    def remove_zmp(self):
        """
        Remove the ZMP from the simulator.
        """
        if self.zmp_visual is not None:
            self.sim.remove_body(self.zmp_visual)
            self.zmp_visual = None

    def remove_cmp(self):
        """
        Remove the CMP from the simulator.
        """
        if self.cmp_visual is not None:
            self.sim.remove_body(self.cmp_visual)
            self.cmp_visual = None

    def remove_fri(self):
        """
        Remove the FRI from the simulator.
        """
        if self.fri_visual is not None:
            self.sim.remove_body(self.fri_visual)
            self.fri_visual = None

    def update_visuals(self):  # TODO: finish this
        """
        Update all visuals.
        """
        # update robot visuals
        super(LeggedRobot, self).update_visuals()

        # update support polygon

        # update friction cones/pyramids

        # update cop
        if self.cop_visual is not None:
            self.draw_cop()

        # update zmp
        if self.zmp_visual is not None:
            self.draw_zmp()

        # update cmp
        if self.cmp_visual is not None:
            self.draw_cmp()

        # update fri
        if self.fri_visual is not None:
            self.draw_fri()


class BipedRobot(LeggedRobot):
    r"""Biped Robot

    A biped robot is a robot which has 2 legs.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the Biped robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(BipedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

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

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the Quadruped robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(QuadrupedRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

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

    def __init__(self, simulator, urdf, position, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the hexapod robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(HexapodRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

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
