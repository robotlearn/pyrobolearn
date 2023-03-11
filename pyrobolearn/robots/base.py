#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various basic bodies / objects that are present in the simulator / world.

Dependencies:
- `pyrobolearn.simulators`
- `pyrobolearn.utils`
"""

import sys
import copy

from pyrobolearn.simulators import Simulator
from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_matrix_from_quaternion

# define long for Python 3.x
if int(sys.version[0]) == 3:
    long = int


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Body(object):
    r"""Physical (Multi-)Body

    Define a physical body in the simulator/world.
    """

    def __init__(self, simulator, body_id=0, name=None):
        """
        Initialize the Body.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id returned by the simulator.
            name (str): name of the body.
        """
        self.simulator = simulator
        self.id = body_id
        self.name = name
        self._mass = None
        self.joints = None

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator."""
        return self.sim

    @simulator.setter
    def simulator(self, simulator):
        """Set the simulator."""
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given simulator to be an instance of `Simulator`, instead got: "
                            "{}".format(type(simulator)))
        self.sim = simulator

    @property
    def id(self):
        """Return the id."""
        return self._id

    @id.setter
    def id(self, body_id):
        """Set the unique body id."""
        if not isinstance(body_id, (int, long)):
            raise TypeError("Expecting the given 'body_id' to be an integer, instead got: {}".format(type(body_id)))
        if body_id < 0:
            raise ValueError("The given 'body_id' is not a valid one; it should be positive, unique, and returned by "
                             "the simulator.")
        self._id = body_id

    @property
    def name(self):
        """Return the name of the body (or the base if not given)."""
        if self._name is None:
            return self.sim.get_body_info(self.id)
        return self._name

    @name.setter
    def name(self, name):
        """Set the name of the body."""
        if name is None:
            name = self.__class__.__name__.lower()
        if not isinstance(name, str):
            raise TypeError("Expecting the given name to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def is_only_visual(self):
        """Return True if the body doesn't have any collision shapes (i.e. it is only a visual body in the simulator)"""
        return len(self.sim.get_collision_shape_data(self.id)) == 0

    @property
    def is_movable(self):
        """Return True if the body is not fixed in the world. A body is fixed if it has a base mass of 0 and has at
        least one collision shape."""
        return not self.is_only_visual and self.base_mass != 0.

    @property
    def base_link_id(self):
        """Return the base link id."""
        return -1

    @property
    def base_name(self):
        """Return the base name."""
        return self.sim.get_base_name(self.id)

    @property
    def base_mass(self):
        """Return the base mass."""
        return self.sim.get_base_mass(self.id)

    @property
    def base_linear_momentum(self):
        """Return the base linear momentum. Warnings: this is not the same as the total linear momentum."""
        return self.mass * self.linear_velocity

    @property
    def pose(self):
        """Return the body pose."""
        return self.sim.get_base_pose(self.id)

    @property
    def position(self):
        """Return the body position."""
        return self.sim.get_base_position(self.id)

    @position.setter
    def position(self, position):
        """Set the body position. This is only valid in the simulator."""
        self.sim.reset_base_position(self.id, position)

    @property
    def orientation(self):
        """Return the body orientation as a quaternion [x,y,z,w]."""
        return self.sim.get_base_orientation(self.id)

    @orientation.setter
    def orientation(self, quaternion):
        """Set the body orientation given the quaternion [x,y,z,w]. This is only valid in the simulator."""
        self.sim.reset_base_orientation(self.id, quaternion)

    # alias
    quaternion = orientation

    @property
    def rpy(self):
        """Return the orientation as the Roll-Pitch-Yaw angles."""
        return get_rpy_from_quaternion(self.orientation)

    @property
    def rotation_matrix(self):
        """Return the orientation as a rotation matrix."""
        return get_matrix_from_quaternion(self.orientation)

    @property
    def forward_vector(self):
        """Return the vector pointing forward."""
        return self.rotation_matrix[:, 0]

    @property
    def left_vector(self):
        """Return the vector pointing on the left."""
        return self.rotation_matrix[:, 1]

    @property
    def up_vector(self):
        """Return the vector pointing upward."""
        return self.rotation_matrix[:, 2]

    @property
    def linear_velocity(self):
        """Return the linear velocity of the body's base."""
        return self.sim.get_base_linear_velocity(self.id)

    @linear_velocity.setter
    def linear_velocity(self, velocity):
        """Set the linear velocity of the body's base; this only makes sense in the simulator."""
        self.sim.reset_base_linear_velocity(self.id, velocity)

    @property
    def angular_velocity(self):
        """Return the angular velocity of the body's base."""
        return self.sim.get_base_angular_velocity(self.id)

    @property
    def velocity(self):
        """Return the linear and angular velocity of the body."""
        return self.sim.get_base_velocity(self.id)

    @property
    def color(self):
        """Return the color of the object."""
        return self.sim.get_visual_shape_data(self.id)[0][-1]

    @color.setter
    def color(self, color):
        """Set the RGBA color of the object. This is only valid in the simulator."""
        self.sim.change_visual_shape(object_id=self.id, link_id=-1, rgba_color=color)

    @property
    def mass(self):
        """Return the total mass of the body."""
        if self._mass is None:
            self._mass = self.sim.get_mass(self.id)
        return self._mass

    @mass.setter
    def mass(self, mass):
        """Set the mass of the body (its base). This is only valid in the simulator."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, mass=mass)

    @property
    def local_inertia_diagonal(self):
        """Return the local inertia diagonal."""
        return self.sim.get_dynamics_info(self.id, link_id=-1)[2]

    @local_inertia_diagonal.setter
    def local_inertia_diagonal(self, inertia):
        """Set the local inertia diagonal."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, local_inertia_diagonal=inertia)

    @property
    def dimensions(self):
        """Return the dimensions of the body. Warnings: this should not be trusted too much..."""
        return self.sim.get_visual_shape_data(self.id)[0][3]

    @property
    def num_joints(self):
        """Return the total number of joints."""
        return self.sim.num_joints(self.id)

    @property
    def num_links(self):
        """Return the total number of links. This is the same as the number of joints."""
        return self.sim.num_links(self.id)

    @property
    def center_of_mass(self):
        """Return the center of mass."""
        return self.sim.get_center_of_mass_position(self.id)

    @property
    def lateral_friction(self):
        """Return the floor lateral friction coefficient."""
        return self.sim.get_dynamics_info(self.id, link_id=-1)[1]

    @lateral_friction.setter
    def lateral_friction(self, coefficient):
        """Set the floor lateral friction coefficient."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, lateral_friction=coefficient)

    @property
    def rolling_friction(self):
        """Return the floor rolling friction coefficient."""
        return self.sim.get_dynamics_info(self.id, -1)[6]

    @rolling_friction.setter
    def rolling_friction(self, coefficient):
        """Set the floor rolling friction coefficient."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, rolling_friction=coefficient)

    @property
    def spinning_friction(self):
        """Return the floor spinning friction coefficient."""
        return self.sim.get_dynamics_info(self.id, -1)[7]

    @spinning_friction.setter
    def spinning_friction(self, coefficient):
        """Set the spinning friction coefficient."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, spinning_friction=coefficient)

    @property
    def restitution(self):
        """Return the floor restitution (bounciness) coefficient."""
        return self.sim.get_dynamics_info(self.id, -1)[5]

    @restitution.setter
    def restitution(self, coefficient):
        """Set the floor restitution (bounciness) coefficient."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, restitution=coefficient)

    @property
    def contact_damping(self):
        """Return the floor contact damping."""
        return self.sim.get_dynamics_info(self.id, -1)[8]

    @contact_damping.setter
    def contact_damping(self, value):
        """Set the floor contact damping value."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, contact_damping=value)

    @property
    def contact_stiffness(self):
        """Return the floor contact stiffness."""
        return self.sim.get_dynamics_info(self.id, -1)[9]

    @contact_stiffness.setter
    def contact_stiffness(self, value):
        """Set the floor contact stiffness value."""
        self.sim.change_dynamics(body_id=self.id, link_id=-1, contact_stiffness=value)

    # just create setter
    def _set_force(self, force):
        """Set the given force (expressed in the world cartesian frame) on the center of mass of the body."""
        self.apply_external_force(link_id=-1, force=force, position=None, frame=Simulator.WORLD_FRAME)

    force = property(fset=_set_force)

    ###########
    # Methods #
    ###########

    def step(self):
        """
        Perform a step. This can be implemented in the child classes.
        """
        pass

    def set_color(self, color, link_id=-1):
        """Set the given RGBA color to the specified link. This is only valid in the simulator.

        Args:
            color (tuple of 4 float): RGBA color where each channel is between 0 and 1.
            link_id (int): link id. By default, it is the base (-1).
        """
        self.sim.change_visual_shape(object_id=self.id, link_id=link_id, rgba_color=color)

    def apply_external_force(self, force=(0., 0., 0.), link_id=-1, position=None, frame=Simulator.LINK_FRAME):
        """
        Apply the given force on the specified link of the current body.

        Warnings:
            - after each simulation step, the external forces are cleared to 0.
            - this does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            force (np.array[float[3]]): Cartesian forces to be applied on the body
            link_id (int): link id to apply the force, if -1 it will apply the force on the base
            position (np.array[float[3]], None): position on the link where the force is applied (expressed in the
                given cartesian frame, see next attribute :attr:`frame`). If None, it is the center of mass of the body
                (or the link if specified).
            frame (int): allows to specify the coordinate system of force/position. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the world frame.
        """
        self.sim.apply_external_force(body_id=self.id, link_id=link_id, force=force, position=position, frame=frame)

    def apply_external_torque(self, torque=(0., 0., 0.), link_id=-1, frame=Simulator.LINK_FRAME):
        """
        Apply an external torque on the body, or a link of the body. Note that after each simulation step, the external
        torques are cleared to 0.

        Warnings: This does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            torque (float[3]): Cartesian torques to be applied on the body
            link_id (int): link id to apply the torque, if -1 it will apply the torque on the base
            frame (int): Specify the coordinate system of force/position: either `pybullet.WORLD_FRAME` (=2) for
                Cartesian world coordinates or `pybullet.LINK_FRAME` (=1) for local link coordinates.
        """
        self.sim.apply_external_torque(self.id, link_id=link_id, torque=torque, frame=frame)

    def get_dynamics(self, link_id=-1):
        """
        Get dynamic information such as the mass, center of mass, friction and other properties of the specified link.

        Args:
            link_id (int): link/joint index or -1 for the base.

        Returns:
            float: mass in kg
            float: lateral friction coefficient
            np.array[float[3]]: local inertia diagonal. Note that links and base are centered around the center of
                mass and aligned with the principal axes of inertia.
            np.array[float[3]]: position of inertial frame in local coordinates of the joint frame
            np.array[float[4]]: orientation of inertial frame in local coordinates of joint frame
            float: coefficient of restitution
            float: rolling friction coefficient orthogonal to contact normal
            float: spinning friction coefficient around contact normal
            float: damping of contact constraints. -1 if not available.
            float: stiffness of contact constraints. -1 if not available.
        """
        return self.sim.get_dynamics_info(self.id, link_id=link_id)

    def change_dynamics(self, link_id=-1, mass=None, lateral_friction=None, spinning_friction=None,
                        rolling_friction=None, restitution=None, linear_damping=None, angular_damping=None,
                        contact_stiffness=None, contact_damping=None, friction_anchor=None,
                        local_inertia_diagonal=None, joint_damping=None):
        """
        Change dynamic properties of the current body (or link) such as mass, friction and restitution coefficients,
        etc.

        Args:
            link_id (int): link index or -1 for the base.
            mass (float): change the mass of the link (or base for link index -1)
            lateral_friction (float): lateral (linear) contact friction
            spinning_friction (float): torsional friction around the contact normal
            rolling_friction (float): torsional friction orthogonal to contact normal
            restitution (float): bouncyness of contact. Keep it a bit less than 1.
            linear_damping (float): linear damping of the link (0.04 by default)
            angular_damping (float): angular damping of the link (0.04 by default)
            contact_stiffness (float): stiffness of the contact constraints, used together with `contact_damping`
            contact_damping (float): damping of the contact constraints for this body/link. Used together with
                `contact_stiffness`. This overrides the value if it was specified in the URDF file in the contact
                section.
            friction_anchor (int): enable or disable a friction anchor: positional friction correction (disabled by
                default, unless set in the URDF contact section)
            local_inertia_diagonal (np.array[float[3]]): diagonal elements of the inertia tensor. Note that the base and
                links are centered around the center of mass and aligned with the principal axes of inertia so there
                are no off-diagonal elements in the inertia tensor.
            joint_damping (float): joint damping coefficient applied at each joint. This coefficient is read from URDF
                joint damping field. Keep the value close to 0.
                `joint_damping_force = -damping_coefficient * joint_velocity`.
        """
        self.sim.change_dynamics(body_id=self.id, link_id=link_id, mass=mass, lateral_friction=lateral_friction,
                                 spinning_friction=spinning_friction, rolling_friction=rolling_friction,
                                 restitution=restitution, linear_damping=linear_damping,
                                 angular_damping=angular_damping, contact_stiffness=contact_stiffness,
                                 contact_damping=contact_damping, friction_anchor=friction_anchor,
                                 local_inertia_diagonal=local_inertia_diagonal, joint_damping=joint_damping)

    def get_collision_shape_data(self, link_id=-1):
        """
        Get the collision shape data associated with the specified link of the current body.

        Args:
            link_id (int): link index or -1 for the base.

        Returns:
            if not has_collision_shape_data:
                tuple: empty tuple
            else:
                int: object unique id.
                int: link id.
                int: geometry type; GEOM_BOX (=3), GEOM_SPHERE (=2), GEOM_CAPSULE (=7), GEOM_MESH (=5), GEOM_PLANE (=6)
                np.array[float[3]]: depends on geometry type:
                    for GEOM_BOX: extents,
                    for GEOM_SPHERE: dimensions[0] = radius,
                    for GEOM_CAPSULE and GEOM_CYLINDER: dimensions[0] = height (length), dimensions[1] = radius.
                    For GEOM_MESH: dimensions is the scaling factor.
                str: Only for GEOM_MESH: file name (and path) of the collision mesh asset.
                np.array[float[3]]: Local position of the collision frame with respect to the center of mass/inertial
                    frame
                np.array[float[4]]: Local orientation of the collision frame with respect to the inertial frame
        """
        return self.sim.get_collision_shape_data(self.id, link_id=link_id)

    def get_visual_shape_data(self, flags=-1):
        """
        Get the visual shape data associated with the current body. It will output a list of visual shape data.

        Args:
            flags (int, None): VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) will also provide `texture_unique_id`.

        Returns:
            list:
                int: object unique id.
                int: link index or -1 for the base
                int: visual geometry type (TBD)
                np.array[float[3]]: dimensions (size, local scale) of the geometry
                str: path to the triangle mesh, if any. Typically relative to the URDF, SDF or MJCF file location, but
                    could be absolute
                np.array[float[3]]: position of local visual frame, relative to link/joint frame
                np.array[float[4]]: orientation of local visual frame relative to link/joint frame
                list[float[4]]: URDF color (if any specified) in Red / Green / Blue / Alpha
                int: texture unique id of the shape or -1 if None. This field only exists if using
                    VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) flag.
        """
        return self.sim.get_visual_shape_data(self.id, flags=flags)

    def apply_texture(self, texture, link_id=-1):
        """
        Apply the texture on the specified link of the current body.

        Args:
            texture (str): path to the texture.
            link_id (int): link id. If -1, it will be the base.
        """
        texture = self.sim.load_texture(texture)
        self.sim.change_visual_shape(self.id, link_id=link_id, texture_id=texture)

    def get_contacts(self):
        """
        Return all the contacts made by the robot.

        Warnings: note that in reality, you can't know if your link(s) is/are in contact with an object unless there
        is a sensor attached to it. However, this can be useful in simulation to optimize, for instance, trajectories.

        Returns:
            list: list of contact points where each contact point has:
                int: contact flag
                int: unique id of body A (this should be the robot id)
                int: unique id of body B
                int: link index of body A (-1 for base, this should be the same as the given link)
                int: link index of body B (-1 for base)
                float[3]: contact position on A (in Cartesian world coordinates)
                float[3]: contact position on B (in Cartesian world coordinates)
                float[3]: contact normal on B pointing towards A
                float: contact distance (positive for separation and negative for penetration)
                float: normal force applied during the last simulation step
        """
        return self.sim.get_contact_points(body1=self.id)

    def get_link_states(self, link_ids, compute_link_velocity=True, compute_forward_kinematics=True):
        """
        Return the state of the given link(s).

        Warning: note that we do not convert the data here.

        Args:
            link_ids (int, list[int]): link id, or list of desired link ids.
            compute_link_velocity (bool): if True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            if 1 link:
                [0] np.array[float[3]]: Cartesian position of center of mass
                [1] np.array[float[4]]: Cartesian orientation of center of mass
                [2] np.array[float[3]]: local position offset of inertial frame (CoM) expressed in the URDF link frame
                [3] np.array[float[4]]: local orientation (quat. [x,y,z,w]) offset of the inertial frame expressed in
                    URDF link frame
                [4] np.array[float[3]]: world position of the URDF link frame
                [5] np.array[float[4]]: world orientation of the URDF link frame
                [6] np.array[float[3]]: Cartesian world linear velocity
                [7] np.array[float[3]]: Cartesian world angular velocity
            if multiple links: list of above
        """
        if isinstance(link_ids, int):  # one link
            return self.sim.get_link_state(self.id, link_ids, compute_velocity=compute_link_velocity,
                                           compute_forward_kinematics=compute_forward_kinematics)
        # multiple links
        return self.sim.get_link_states(self.id, link_ids, compute_velocity=compute_link_velocity,
                                        compute_forward_kinematics=compute_forward_kinematics)

    def get_joint_states(self, joint_ids):
        """
        Get the state of the given joint(s).

        Args:
            joint_ids (int, list[int]): id of the joint, or list of joint ids.

        Returns:
            for 1 joint:
                float: joint position [rad]
                float: joint velocity [rad/s]
                np.array[6]: joint reaction forces [fx,fy,fz,mx,my,mz]
                float: applied joint motor torque (during the last step)
            for multiple joints: list of each joint state
        """
        if isinstance(joint_ids, int):
            return self.sim.get_joint_state(self.id, joint_ids)
        return self.sim.get_joint_states(self.id, joint_ids)

    def get_joint_info(self, joint_ids):
        """
        Get information about the given joint(s).

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                [0] int:        the same joint id as the input parameter
                [1] str:        name of the joint (as specified in the URDF/SDF/etc file)
                [2] int:        type of the joint which implie the number of position and velocity variables.
                                The types include JOINT_REVOLUTE (=0), JOINT_PRISMATIC (=1), JOINT_SPHERICAL (=2),
                                JOINT_PLANAR (=3), and JOINT_FIXED (=4).
                [3] int:        q index - the first position index in the positional state variables for this body
                [4] int:        dq index - the first velocity index in the velocity state variables for this body
                [5] int:        flags (reserved)
                [6] float:      the joint damping value (as specified in the URDF file)
                [7] float:      the joint friction value (as specified in the URDF file)
                [8] float:      the positional lower limit for slider and revolute joints
                [9] float:      the positional upper limit for slider and revolute joints
                [10] float:     maximum force specified in URDF. Note that this value is not automatically used.
                                You can use maxForce in 'setJointMotorControl2'.
                [11] float:     maximum velocity specified in URDF. Note that this value is not used in actual
                                motor control commands at the moment.
                [12] str:       name of the link (as specified in the URDF/SDF/etc file)
                [13] np.array[float[3]]:  joint axis in local frame (ignored for JOINT_FIXED)
                [14] np.array[float[3]]:  joint position in parent frame
                [15] np.array[float[4]]:  joint orientation in parent frame (x, y, z, w)
                [16] int:       parent link index, -1 for base

            if multiple joints: list of joint information (i.e. list of above)
        """
        if isinstance(joint_ids, int):
            return self.sim.get_joint_info(self.id, joint_ids)
        return [self.sim.get_joint_info(self.id, joint_id) for joint_id in joint_ids]

    #############
    # Operators #
    #############

    # def __repr__(self):
    #     """Return a representation string about the class for debugging and development."""
    #     return self.__class__.__name__

    def __str__(self):
        """Return a readable string about the class."""
        return self.__class__.__name__ + '(' + self.name + ')'

    def __copy__(self):
        """Return a shallow copy of the body. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, body_id=self.id, name=self.name)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the body. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        simulator = copy.deepcopy(self.simulator, memo)
        body = self.__class__(simulator=simulator, body_id=self.id, name=self.name)

        memo[self] = body
        return body


class MovableBody(Body):
    r"""Movable Body

    Define a movable object in the world.
    """

    def __init__(self, simulator, object_id=0, name=None):
        super(MovableBody, self).__init__(simulator, object_id, name=name)
        # # check that the body is movable
        # if not self.is_movable:
        #     raise ValueError("The given id does not correspond to a movable body.")

    # def move(self, position=None, orientation=None):
    #     pass


class ControllableBody(MovableBody):
    r"""Controllable Body

    Define a controllable object in the world.
    """

    def __init__(self, simulator, object_id=0, name=None):
        super(ControllableBody, self).__init__(simulator, object_id, name=name)

    @property
    def actuated_joints(self):
        """Return the total number of actuated joints."""
        if self.joints is None:
            self.joints = self.sim.get_actuated_joint_ids(self.id)
        return self.joints

    @property
    def num_actuated_joints(self):
        """Return the total number of actuated joints. This property should be overwritten in the child class."""
        if self.joints is None:
            return self.sim.num_actuated_joints(self.id)
        return len(self.joints)
