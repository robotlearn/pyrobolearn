#!/usr/bin/env python
"""Define the various basic bodies / objects that are present in the simulator / world.

Dependencies:
- `pyrobolearn.simulators`
- `pyrobolearn.utils`
"""

import numpy as np
# import quaternion

from pyrobolearn.simulators import Simulator
from pyrobolearn.utils.orientation import get_rpy_from_quaternion, get_matrix_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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
    def id(self, id_):
        """Set the unique body id."""
        # if not isinstance(id_, int):
        #     raise TypeError("Expecting the given simulator to be an integer, instead got: {}".format(type(id_)))
        self._id = id_

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

    @property
    def orientation(self):
        """Return the body orientation as a quaternion [x,y,z,w]."""
        return self.sim.get_base_orientation(self.id)

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
        return self.sim.get_visual_shape_data(self.id)[0][-1]

    @property
    def mass(self):
        """Return the total mass of the body."""
        if self._mass is None:
            self._mass = self.sim.get_mass(self.id)
        return self._mass

    @property
    def dimensions(self):
        """Return the dimensions of the body. Warnings: this should not be trusted too much..."""
        return np.array(self.sim.get_visual_shape_data(self.id)[0][3])

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


class MovableBody(Body):
    r"""Movable Body

    Define a movable object in the world.
    """

    def __init__(self, simulator, object_id=0, name=None):
        super(MovableBody, self).__init__(simulator, object_id, name=name)

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
