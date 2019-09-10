#!/usr/bin/env python
"""Provide the common data structures that are shared among the various parsers, generators, and converters.
"""

import copy
import numpy as np
import trimesh
from collections import OrderedDict, Iterable

from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_quaternion_from_rpy, get_matrix_from_rpy, \
    get_rpy_from_matrix, get_matrix_from_axis_angle
from pyrobolearn.utils.inertia import get_inertia_of_box, get_inertia_of_capsule, get_inertia_of_cylinder, \
    get_inertia_of_ellipsoid, get_inertia_of_mesh, get_inertia_of_sphere, combine_inertias


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Simulator(object):
    r"""Simulator data structure."""

    def __init__(self, name=None, worlds=None, physics_engine=None, physics_properties=None):
        """
        Initialize the simulator data structure.

        Args:
            name (str, None): name of the simulator.
            worlds (list[World], None): world data structure instances.
            physics_engine (PhysicsEngine): physics engine instance.
            physics_properties (Physics): the physics properties (gravity, viscosity, friction, etc).
        """
        self.worlds = worlds
        self.engine = physics_engine
        self.physics = physics_properties

    @property
    def worlds(self):
        """Return the worlds."""
        return self._worlds

    @worlds.setter
    def worlds(self, worlds):
        """Set the world data structure instances."""
        if worlds is None:
            worlds = []
        elif isinstance(worlds, World):
            worlds = [worlds]

        if not isinstance(worlds, (list, tuple)):
            raise TypeError("Expecting the given 'worlds' to be a list/tuple of `World` instances, but got instead:"
                            " {}".format(type(worlds)))
        for world in worlds:
            if not isinstance(world, World):
                raise TypeError("Expecting the world to be an instance of `World`, but got instead: "
                                "{}".format(type(world)))
        self._worlds = worlds

    @property
    def world(self):
        """Return the first world."""
        if len(self._worlds) > 0:
            return self._worlds[0]

    @property
    def engine(self):
        """Return the physics engine."""
        return self._engine

    @engine.setter
    def engine(self, engine):
        """Set the physics engine."""
        if engine is not None and not isinstance(engine, PhysicsEngine):
            raise TypeError("Expecting the engine to be an instance of `PhysicsEngine`, but got instead: "
                            "{}".format(type(engine)))
        self._engine = engine

    @property
    def physics(self):
        """Return the physics properties."""
        return self._physics

    @physics.setter
    def physics(self, physics):
        """Set the physics properties."""
        if physics is not None and not isinstance(physics, Physics):
            raise TypeError("Expecting the given 'physics' to be an instance of `Physics`, but got instead: "
                            "{}".format(type(physics)))
        self._physics = physics

    def add_world(self, world):
        r"""
        Append a world to the list of worlds.

        Args:
            world (World): world instance.
        """
        if not isinstance(world, World):
            raise TypeError("Expecting the given 'world' to be an instance of `World`, but got instead: "
                            "{}".format(type(world)))
        self.worlds.append(world)


class PhysicsEngine(object):
    r"""Physics Engine properties.

    This include number of iterations, solver used, tolerance, timesteps, etc.

    MuJoCo:
    - solver: PGS, CG, Newton.
    """

    def __init__(self, timestep=None):
        """
        Initialize the Physics engine parameters.

        Args:
            timestep (float, str): simulation time step in seconds.
        """
        self.timestep = timestep
        self.num_iterations = None
        self.solver = None
        self.tolerance = None

    @property
    def timestep(self):
        """Return the simulation time step."""
        return self._timestep

    @timestep.setter
    def timestep(self, timestep):
        """Set the simulation time step."""
        if timestep is not None:
            timestep = float(timestep)
        self._timestep = timestep


class Frame(object):
    r"""Reference Frame"""

    def __init__(self, position=None, orientation=None, dtype=None, right_handed=True):
        """
        Initialize the frame.

        Args:
            position (list/tuple/np.array[float[3]], str): frame position.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): frame orientation.
            dtype (str): frame type. It can be a {'world', 'body', 'joint', 'inertial'} frame.
            right_handed (bool): if the frame use the right hand rule; the z-axis is given by :math:`z = x \times y`
              using the right hand.
        """
        # forward_axis=(1., 0., 0.), up_axis=(0., 0., 1.)):

        self.position = position
        self.orientation = orientation
        self.dtype = dtype  # world frame, body frame, joint frame, inertial frame, etc.
        self.right_handed = right_handed
        # self.forward_axis = forward_axis
        # self.up_axis = up_axis

    @property
    def position(self):
        """Return the frame position."""
        return self._position

    @position.setter
    def position(self, position):
        """Set the frame position."""
        if position is not None:
            if isinstance(position, str):
                position = [float(p) for p in position.split()]
            position = np.asarray(position)
            if len(position) != 3:
                raise ValueError("Expecting the given position to have a length of 3, but got instead a length of: "
                                 "{}".format(len(position)))
        self._position = position

    @property
    def orientation(self):
        """Return the frame orientation as RPY angles."""
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the frame orientation (which can be given as RPY angles, rotation matrix, a quaternion [x,y,z,w], or
        an axis with its angle)."""
        if orientation is not None:
            if isinstance(orientation, str):
                orientation = [float(o) for o in orientation.split()]
            orientation = np.asarray(orientation)
            if len(orientation) == 4:  # quaternion
                orientation = get_rpy_from_quaternion(orientation)
            elif len(orientation) == 3:  # rpy or rot
                if orientation.shape == (3, 3):
                    orientation = get_rpy_from_matrix(orientation)
            elif len(orientation) == 9:  # rot matrix
                orientation = orientation.reshape(3, 3)
                orientation = get_rpy_from_matrix(orientation)
            elif len(orientation) == 2:  # tuple of (axis, angle)
                axis, angle = orientation[0], orientation[1]
                if isinstance(angle, str):
                    angle = float(angle)
                if isinstance(axis, str):
                    axis = np.array([float(c) for c in axis.split()])
                orientation = get_rpy_from_matrix(get_matrix_from_axis_angle(axis=axis, angle=angle))
        self._orientation = orientation

    @property
    def rpy(self):
        """Return the frame orientation expressed as RPY angles."""
        return self.orientation

    @property
    def quaternion(self):
        """Return the frame orientation expressed as a quaternion [x,y,z,w]."""
        if self._orientation is not None:
            return get_quaternion_from_rpy(self.orientation)

    @property
    def rot(self):
        """Return the frame orientation expressed as a rotation matrix."""
        if self._orientation is not None:
            return get_matrix_from_rpy(self.rpy)

    @property
    def pose(self):
        """Return the frame pose."""
        if self.position is None and self.orientation is None:
            return None
        return self.position, self.orientation

    @pose.setter
    def pose(self, pose):
        """Set the frame pose."""
        if pose is not None:
            if isinstance(pose, str):
                pose = pose.split()
                self.position = pose[:3]
                self.orientation = pose[3:]
            elif isinstance(pose, (tuple, list, np.ndarray)):
                if len(pose) == 2:
                    self.position = pose[0]
                    self.orientation = pose[1]
                elif len(pose) == 6:
                    self.position = pose[:3]
                    self.orientation = pose[3:]
                else:
                    raise ValueError("Expecting the pose to be tuple, list or np.ndarray of length 2 or 6")
            else:
                raise TypeError("Expecting the pose to be a str, list, tuple or np.ndarray")

    @property
    def homogeneous(self):
        """Return the homogeneous matrix based on the position and orientation. Note that if the orientation is
        None it will be set to the identity matrix, and if the position is None, it will be set to the zero vector."""
        R = self.rot if self._orientation is not None else np.identity(3)
        p = self._position if self._position is not None else np.zeros(3)
        return np.vstack((np.hstack((R, p.reshape(-1, 1))),
                          np.array([[0, 0, 0, 1]])))

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the position and orientation of the frame based on the given homogeneous matrix."""
        self.position = matrix[:3, 3]
        self.orientation = matrix[:3, :3]


class Physics(object):
    r"""Physical properties of the world.

    This includes gravity, friction, viscosity, etc.
    """

    def __init__(self, gravity=(0., 0., -9.81), timestep=None):
        """
        Initialize the physical properties of the world.

        Args:
            gravity (list/tuple/np.array[float[3]], str): gravity vector.
            timestep (float, str): simulation time step in seconds.
        """
        # gravity depends on the world frame; the frame axis convention that we use.
        # By default, x points forward, y on the left, and z upward.
        self.gravity = gravity
        self.timestep = timestep

    @property
    def gravity(self):
        """Return the gravity vector."""
        return self._gravity

    @gravity.setter
    def gravity(self, gravity):
        """Set the gravity vector."""
        if gravity is not None:
            if isinstance(gravity, str):
                gravity = [float(g) for g in gravity.split()]
            gravity = np.asarray(gravity).reshape(-1)
            if len(gravity) != 3:
                raise ValueError("Expecting the gravity vector to be of length 3, but got a length of "
                                 "{}.".format(len(gravity)))
        self._gravity = gravity

    @property
    def timestep(self):
        """Return the simulation time step."""
        return self._timestep

    @timestep.setter
    def timestep(self, timestep):
        """Set the simulation time step."""
        if timestep is not None:
            timestep = float(timestep)
        self._timestep = timestep


class World(object):
    r"""World data structure.

    World frame (robotics convention with the right-hand rule):
    - the x axis points forward
    - the y axis points to the left
    - the z axis points upward

    The world has a bunch of multi-body systems, each body is represented by a tree. Each tree has a root element
    (=root link/body) where each element is connected by joints.
    """

    def __init__(self, name=None, position=None, orientation=None):
        """
        Initialize the world data structure which contains the various bodies.

        Args:
            name (str): name of the world. Useful if you have multiple world.
            position (tuple/list of 3 float, np.array[float[3]]): position of the origin of the world wrt the simulator
              reference frame. This is useful if you have multiple worlds.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): world frame orientation wrt
              the simulator reference frame. This is useful if you have multiple worlds.
        """
        self.name = name
        self.trees = OrderedDict()  # {(unique) name: Tree}
        self.physics = None
        self.lights = OrderedDict()
        self.frame = Frame(position=position, orientation=orientation)
        self.floor = None

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def physics(self):
        """Return the physical properties set in the world."""
        return self._physics

    @physics.setter
    def physics(self, physics):
        """Set the physical properties in the world."""
        if physics is not None and not isinstance(physics, Physics):
            raise TypeError("Expecting the physics to be an instance of `Physics`, but got instead: "
                            "{}".format(type(physics)))
        self._physics = physics

    @property
    def gravity(self):
        """Return the gravity vector."""
        if self._physics is not None:
            return self._physics.gravity

    @gravity.setter
    def gravity(self, gravity):
        """Set the gravity vector."""
        if self._physics is None:
            self._physics = Physics()
        self._physics.gravity = gravity

    @property
    def floor(self):
        """Return the floor."""
        return self._floor

    @floor.setter
    def floor(self, floor):
        """Set the floor."""
        if floor is not None and not isinstance(floor, Floor):
            raise TypeError("Expecting the given floor to be an instance of `Floor`, but got instead: "
                            "{}".format(type(floor)))
        self._floor = floor

    @property
    def position(self):
        """Return the world frame position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the world frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the world frame orientation."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the world frame orientation expressed as RPY angles."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the world frame orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the world frame orientation expressed as RPY angles."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the world frame orientation expressed as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the world frame pose."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the world frame pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix


class Light(object):
    r"""Light data structure.

    Type of light: point, directional, and spot
    """

    def __init__(self, name=None, dtype=None, cast_shadows=None, ambient=None, diffuse=None, specular=None,
                 attenuation=None, direction=None, spot=None, position=None, orientation=None, active=True):
        """
        Initialize the Light data structure.

        Args:
            name (str): unique name for the light.
            dtype (str): type of light, select between {'point', 'directional', 'spot'}
            cast_shadows (bool, str): if True, it will cast shadows.
            ambient (tuple[float[4]], np.array[float[4]]): ambient light (RGBA) color.
            diffuse (tuple[float[4]], np.array[float[4]]): diffuse light (RGBA) color.
            specular (tuple[float[4]], np.array[float[4]]): specular light (RGBA) color.
            attenuation: light attenuation
            direction (np.array[float[3]]): direction of the light if dtype='directional' or dtype='spot'.
            spot: spot light parameters
            position (tuple/list of 3 float, np.array[float[3]]): position of the light in the world.
            orientation (np.array[float[3]], str): orientation of the light expressed as roll-pitch-yaw angles.
            active (bool): if True, the light is on.
        """
        self.name = name
        self.dtype = dtype
        self.shadows = cast_shadows
        self.attenuation = attenuation
        self.direction = direction
        self.spot = spot
        self.active = active
        self.frame = Frame(position=position, orientation=orientation)
        self.material = Material(color=ambient, diffuse=diffuse, specular=specular)

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def shadows(self):
        """Return True if we should cast shadows."""
        return self._shadows

    @shadows.setter
    def shadows(self, enable):
        """Set if we should cast shadows."""
        if enable is not None:
            if isinstance(enable, str):
                enable = enable.lower()
                if len(enable) == 1:
                    enable = int(enable)
                elif enable == 'false':
                    enable = 0
                elif enable == 'true':
                    enable = 1
            enable = bool(enable)
        self._shadows = enable

    @property
    def active(self):
        """Return True if the light is active."""
        return self._active

    @active.setter
    def active(self, enable):
        """Set if the light is active or not."""
        if enable is not None:
            if isinstance(enable, str):
                enable = enable.lower()
                if len(enable) == 1:
                    enable = int(enable)
                elif enable == 'false':
                    enable = 0
                elif enable == 'true':
                    enable = 1
            enable = bool(enable)
        self._active = enable

    @property
    def position(self):
        """Return the light frame position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the light frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the light frame orientation."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the light frame orientation expressed as RPY angles."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the light frame orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the light frame orientation expressed as RPY angles."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the light frame orientation expressed as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the light frame pose."""
        return self.frame.pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix

    @pose.setter
    def pose(self, pose):
        """Set the light frame pose."""
        self.frame.pose = pose

    @property
    def direction(self):
        """Return the light direction."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the direction of the light."""
        if direction is not None:
            if isinstance(direction, str):
                direction = [float(d) for d in direction.split()]
            direction = np.asarray(direction).reshape(-1)
        self._direction = direction

    @property
    def ambient(self):
        """Return the light ambient color."""
        return self.material.ambient

    @ambient.setter
    def ambient(self, ambient):
        """Set the light ambient color."""
        self.material.ambient = ambient

    color = ambient

    @property
    def diffuse(self):
        """Return the light diffuse color."""
        return self.material.diffuse

    @diffuse.setter
    def diffuse(self, diffuse):
        """Set the light diffuse color."""
        self.material.diffuse = diffuse

    @property
    def specular(self):
        """Return the light specular color."""
        return self.material.specular

    @specular.setter
    def specular(self, specular):
        """Set the light specular color."""
        self.material.specular = specular


class Floor(object):
    r"""Floor data structure.

    """

    def __init__(self, name=None, dimensions=None, position=None, orientation=None, friction=None, color=None,
                 texture=None):
        """
        Initialize the floor data structure.

        Args:
            name (str): name of the floor.
            dimensions (list/tuple/np.array[float[:3]], str): dimensions of the floor. By default, the given arguments
              are expected to be the (X/2, Y/2, space between square grid lines). If the first and second elements are
              zero, it is expected to create an infinite floor.
            position (list/tuple/np.array[float[3]], str): frame position in the world.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): frame orientation in the world.
            friction (list/tuple/np.array[float[:3]]): friction coefficients. By default, the first coefficient is
              supposed to be the sliding friction, the 2nd one is the torsional friction, and the 3rd one is the
              rolling friction.
            color (list[float[:4]], str): RGB(A) ambient color.
            texture (str): path to the texture.
        """
        self.name = name
        self.dimensions = dimensions
        self.frame = Frame(position, orientation, dtype='world')
        if name is not None:
            name = name + '_material'
        self.material = Material(name=name, color=color, texture=texture)

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def dimensions(self):
        """Return the dimensions of the floor."""
        return self._dims

    @dimensions.setter
    def dimensions(self, dims):
        if dims is not None:
            if isinstance(dims, str):
                dims = [float(p) for p in dims.split()]
            dims = np.asarray(dims)
            if len(dims) > 3:
                raise ValueError("Expecting the given dims to have a length below 3, but got instead a length of: "
                                 "{}".format(len(dims)))
        self._dims = dims

    @property
    def position(self):
        """Return the floor frame position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the floor frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the floor frame orientation."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the floor frame orientation expressed as RPY angles."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the floor frame orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the floor frame orientation expressed as a quaternion [x,y,z,w]."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the floor frame orientation expressed as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the floor frame pose."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the floor frame pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix

    @property
    def color(self):
        """Return the ambient color."""
        return self.material.color

    @property
    def rgb(self):
        """Return the RGB ambient color."""
        return self.material.rgb

    @property
    def rgba(self):
        """Return the RGBA ambient color."""
        return self.material.rgba

    @property
    def texture(self):
        """Return the path to the texture."""
        return self.material.texture


class MultiBody(object):
    r"""Multi-body / Tree data structure.

    The multi-body / tree data structure starts with a root element (=base link) and contains each bodies / joints.
    Each tree represents a multi-body in the world. Its position / orientation is expressed in the world frame.
    """

    def __init__(self, name=None, root=None, position=None, orientation=None):
        """
        Initialize the Multi-body / Tree data structure.

        Args:
            name (str): name of the tree.
            root (Body): root element in the tree.
            position (list/tuple/np.array[float[3]], str): frame position in the world.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): frame orientation in the world.
        """
        self.name = name
        self.root = root
        self.bodies = OrderedDict()  # {name: Body}
        self.joints = OrderedDict()  # {name: Joint}
        self.materials = {}
        self.frame = Frame(position, orientation, dtype='world')

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def num_dofs(self):
        """Return the total number of degrees of freedom."""
        num_dofs = 0
        for joint in self.joints.values():
            num_dofs += joint.num_dofs
        return num_dofs

    @property
    def num_bodies(self):
        """Return the total number of bodies in this multi-body."""
        return len(self.bodies)

    @property
    def num_joints(self):
        """Return the total number of joints in this multi-body (this accounts for fixed joints as well, but not
        free joints)."""
        # return len(self.joints)
        num_joints = 0
        for joint in self.joints.values():
            if joint.dtype != 'floating':
                num_joints += 1
        return num_joints

    @property
    def num_actuated_joints(self):
        """Return the total number of joints which are not fixed nor free."""
        num_actuated_joints = 0
        for joint in self.joints.values():
            if joint.dtype != 'fixed' and joint.dtype != 'floating':
                num_actuated_joints += 1
        return num_actuated_joints

    @property
    def root(self):
        """Return the root body element."""
        if self._root is not None:
            return self._root
        if len(self.bodies):
            return self.bodies[next(iter(self.bodies))]  # get first element

    @root.setter
    def root(self, root):
        """Set the root body element."""
        if root is not None and not isinstance(root, Body):
            raise TypeError("Expecting the given 'body' to be an instance of `Body`, but instead got: "
                            "{}".format(type(root)))
        self._root = root

    @property
    def static(self):
        """Return if the root element in the tree is static or not."""
        if self.root is not None:
            return self.root.static

    @static.setter
    def static(self, static):
        """Set the root element in the tree to be static or not."""
        if self.root is not None:
            self.root.static = static

    @property
    def position(self):
        """Return the tree frame position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the tree frame position."""
        self.frame.position = position
        if self.root is not None:
            self.root.position = position

    @property
    def orientation(self):
        """Return the tree frame orientation."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the tree frame orientation expressed as RPY angles."""
        self.frame.orientation = orientation
        if self.root is not None:
            self.root.orientation = orientation

    @property
    def rpy(self):
        """Return the tree frame orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the tree frame orientation expressed as a quaternion [x,y,z,w]."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the tree frame orientation expressed as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the tree frame pose."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the tree frame pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix
        if self.root is not None:
            self.root.homogeneous = matrix

    def add_body(self, body):
        """Add a body to the tree.

        Args:
            body (Body): body data structure.
        """
        if not isinstance(body, Body):
            raise TypeError("Expecting the given 'body' to be an instance of `Body` but got instead: "
                            "{}".format(type(body)))
        self.bodies[body.name] = body

    def add_joint(self, joint, idx=None):
        """Add a joint to the tree.

        Args:
            joint (Joint): joint data structure.
            idx (int): index to insert a joint. This has a O(N) complexity as we have to copy everything.
        """
        if not isinstance(joint, Joint):
            raise TypeError("Expecting the given 'joint' to be an instance of `Joint`, but got instead: "
                            "{}".format(type(joint)))
        if idx is not None:
            self.joints[joint.name] = joint
        else:
            # this insert
            joints = OrderedDict()
            for i, (joint_name, joint_instance) in enumerate(self.joints.items()):
                if i == idx:
                    joints[joint.name] = joint
                joint[joint_name] = joint_instance


# alias
Tree = MultiBody


class Body(object):
    r"""Body / Link data structure."""

    def __init__(self, body_id, name=None, inertials=None, visuals=None, collisions=None, static=False,
                 position=None, orientation=None, frame_type=None):
        """
        Initialize the Body / Link data structure.

        Args:
            body_id (int): body unique id.
            name (str, None): body name.
            inertials (Inertial, list[Inertial], None): inertial components. Multiple inertial components can be
              provided and by calling the `inertia` property they will be combined together to only form one inertial
              component.
            visuals (Visual, list[Visual], None): visual shapes. Multiple visual shape instances can be provided for a
              specific body.
            collisions (Collision, list[Collision], None): collision shapes. Multiple collision shape instances can
              be provided for a specific body.
            static (bool): if the body is static in the world or not.
            position (list/tuple/np.array[float[3]], str): body frame position. If None, it will look at the visual
              and collision shapes. By default, if the visual shape is defined it will return its position.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): body frame orientation. If
              None, it will look at the collision shapes. By default, if the visual shape is defined it will return
              its orientation.
            frame_type (str): frame type. It can be a {'world', 'body', 'joint', 'inertial'} frame.
        """
        self.id = int(body_id)
        self.name = name

        self.joints = OrderedDict()             # child joints
        self.parent_joints = OrderedDict()      # parent joints

        # set body properties
        self.inertials = inertials
        self.visuals = visuals
        self.collisions = collisions
        self.static = static

        self.frame = Frame(position=position, orientation=orientation, dtype=frame_type)

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def bodies(self):
        """Return the child (inner) bodies / links."""
        bodies = []
        for joint in self.joints.values():
            if joint.child is not None:
                bodies.append(joint.child)
        return bodies

    @property
    def inertials(self):
        """Return the inertial components of the body."""
        return self._inertials

    @inertials.setter
    def inertials(self, inertials):
        """Set the inertial components of the body."""
        if inertials is None:
            inertials = []
        elif isinstance(inertials, Inertial):
            inertials = [inertials]
        elif isinstance(inertials, (list, tuple)):
            for i, inertial in enumerate(inertials):
                if not isinstance(inertial, Inertial):
                    raise TypeError("The {}th element in the given inertials is not an instance of `Inertial`, but: "
                                    "{}".format(i, type(inertial)))
        else:
            raise TypeError("Expecting the given 'inertials' to be a list of `Inertial`, or an instance of "
                            "`Inertial`, but got instead: {}".format(type(inertials)))
        self._inertials = inertials

    @property
    def inertial(self):
        """Return the inertial component of the body."""
        if len(self.inertials) == 1:
            return self.inertials[0]
        return self.combine_inertials(self.inertials)

    @property
    def mass(self):
        """Return the mass."""
        if len(self._inertials) > 0:
            if len(self._inertials) > 1:
                print("WARNING: multiple inertial elements are defined, as such we return the mass of each one.")
                return [inertial.mass for inertial in self._inertials]
            return self.inertial.mass

    @mass.setter
    def mass(self, mass):
        """Set the mass."""
        if self._inertials:
            if len(self._inertials) == 0:
                inertial = Inertial(mass=mass)
                self.add_inertial(inertial)
            elif len(self._inertials) == 1:
                self._inertials[0].mass = mass
            else:
                if not isinstance(mass, (list, tuple, np.ndarray)):
                    raise ValueError("Expecting the mass to be a list/tuple/array of float, instead got: "
                                     "{}".format(type(mass)))
                if len(mass) != len(self._inertials):
                    raise ValueError("Expecting the given mass to be a list of the same size as the number of "
                                     "inertial elements in this class, instead got: len(mass) = {}, and "
                                     "len(inertials) = {}".format(len(mass), len(self._inertials)))
                for m, inertial in zip(mass, self._inertials):
                    inertial.mass = m

    @property
    def visuals(self):
        """Return the visual shapes of the body."""
        return self._visuals

    @visuals.setter
    def visuals(self, visuals):
        """Set the visual shapes of the body."""
        if visuals is None:
            visuals = []
        elif isinstance(visuals, Visual):
            visuals = [visuals]
        elif isinstance(visuals, (list, tuple)):
            for i, visual in enumerate(visuals):
                if not isinstance(visual, Visual):
                    raise TypeError("The {}th element in the given visuals is not an instance of `Visual`, but: "
                                    "{}".format(i, type(visual)))
        else:
            raise TypeError("Expecting the given 'visuals' to be a list of `Visual`, or an instance of "
                            "`Visual`, but got instead: {}".format(type(visuals)))
        self._visuals = visuals

    @property
    def visual(self):
        """Return the first visual shape of the body."""
        if len(self._visuals) > 0:
            return self._visuals[0]

    @property
    def collisions(self):
        """Return the collision shapes of the body."""
        return self._collisions

    @collisions.setter
    def collisions(self, collisions):
        """Set the collision shape of the body."""
        if collisions is None:
            collisions = []
        elif isinstance(collisions, Collision):
            collisions = [collisions]
        elif isinstance(collisions, (list, tuple)):
            for i, collision in enumerate(collisions):
                if not isinstance(collision, Collision):
                    raise TypeError("The {}th element in the given collisions is not an instance of `Collision`, but: "
                                    "{}".format(i, type(collision)))
        else:
            raise TypeError("Expecting the given 'collisions' to be a list of `Collision`, or an instance of "
                            "`Collision`, but got instead: {}".format(type(collisions)))
        self._collisions = collisions

    @property
    def collision(self):
        """Return the first collision shape of the body."""
        if len(self._collisions) > 0:
            return self._collisions[0]

    @property
    def static(self):
        """Return if the body is static in the world or not."""
        return self._static

    @static.setter
    def static(self, static):
        """Set if the body is static in the world or not."""
        self._static = bool(static)

    @property
    def position(self):
        """Return the body frame position."""
        if self.frame.position is not None:
            return self.frame.position
        if self.visual is not None and self.visual.position is not None:
            return self.visual.position
        if self.collision is not None:
            return self.collision.position

    @position.setter
    def position(self, position):
        """Set the body frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the body frame orientation expressed as RPY angles."""
        if self.frame.orientation is not None:
            return self.frame.orientation
        if self.visual is not None and self.visual.orientation is not None:
            return self.visual.orientation
        if self.collision is not None:
            return self.collision.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the body frame orientation (which can be expressed as a rotation matrix, RPY angles. or a quaternion
        [x,y,z,w]."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the body frame orientation expressed as RPY angles."""
        if self.frame.orientation is not None:
            return self.frame.rpy
        if self.visual is not None and self.visual.orientation is not None:
            return self.visual.rpy
        if self.collision is not None:
            return self.collision.rpy

    @property
    def quaternion(self):
        """Return the body frame orientation expressed as a quaternion [x,y,z,w]."""
        if self.frame.orientation is not None:
            return self.frame.quaternion
        if self.visual is not None and self.visual.orientation is not None:
            return self.visual.quaternion
        if self.collision is not None:
            return self.collision.quaternion

    @property
    def rot(self):
        """Return the body frame orientation expressed as a rotation matrix."""
        if self.frame.orientation is not None:
            return self.frame.rot
        if self.visual is not None and self.visual.orientation is not None:
            return self.visual.rot
        if self.collision is not None:
            return self.collision.rot

    @property
    def pose(self):
        """Return the body frame pose."""
        if self.frame.pose is not None:
            return self.frame.pose
        if self.visual is not None and self.visual.pose is not None:
            return self.visual.pose
        if self.collision is not None:
            return self.collision.pose

    @pose.setter
    def pose(self, pose):
        """Set the body frame pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix

    def add_collision(self, collision):
        """
        Add a collision shape to the list of collision shapes.

        Args:
            collision (Collision): collision instance.
        """
        if not isinstance(collision, Collision):
            raise TypeError("Expecting the given 'collision' to be an instance of `Collision`, but got instead: "
                            "{}".format(type(collision)))
        self.collisions.append(collision)

    def add_visual(self, visual):
        """
        Add a visual shape to the list of visual shapes.

        Args:
            visual (Visual): visual instance.
        """
        if not isinstance(visual, Visual):
            raise TypeError("Expecting the given 'visual' to be an instance of `Visual`, but got instead: "
                            "{}".format(type(visual)))
        self.visuals.append(visual)

    def add_inertial(self, inertial):
        """
        Add an inertial element to the list of inertials.

        Args:
            inertial (Inertial): inertial element.
        """
        if not isinstance(inertial, Inertial):
            raise TypeError("Expecting the given 'inertial' to be an instance of `Inertial`, but got instead: "
                            "{}".format(type(inertial)))
        self.inertials.append(inertial)

    @staticmethod
    def combine_inertials(inertials):
        """
        Combine the given inertial elements.

        Args:
            inertials (list[Inertial]): list of Inertial elements.

        Returns:
            Inertial: combined inertial element.
        """
        if len(inertials) == 0:
            return None
        inertial = copy.deepcopy(inertials[0])
        for i in range(1, len(inertials)):
            inertial += inertials[i]
        return inertial

    def add_child_joint(self, joint):
        """
        Add child joint.

        Args:
            joint (Joint): joint instance.
        """
        if not isinstance(joint, Joint):
            raise TypeError("Expecting the given 'joint' to be an instance of `Joint` but instead got: "
                            "{}".format(type(joint)))
        self.joints[joint.id] = joint

    def add_parent_joint(self, joint):
        """
        Add parent joint.

        Args:
            joint (Joint): parent joint instance.
        """
        if not isinstance(joint, Joint):
            raise TypeError("Expecting the given 'joint' to be an instance of `Joint` but instead got: "
                            "{}".format(type(joint)))
        self.parent_joints[joint.id] = joint


class Joint(object):
    r"""Joint data structure.

    Joint types: fixed, floating/free, prismatic, revolute/hinge, continuous, gearbox, revolute2, ball, screw,
    universal, and planar.

    - fixed: no motions is allowed; both links are rigidly attached to each other.
    - floating/free: allows motion for all 6 degrees of motion.
    - prismatic: allows motion along 1 translational DoF.
    - revolute/hinge: allows rotational motion around one axis (1 DoF).
    - continuous: a revolute/hinge joint that doesn't have lower or upper limits.
    - gearbox: geared revolute joint.
    - revolute2: two revolute joints connected in series
    - ball: a ball and socket joint which allows rotational motions around the 3 axis (3 DoFs).
    - screw: a single DoF joint wich coupled sliding and rotational motion
    - universal: like a ball joint, but constrains one DoF
    - planar: allows motion in a plane perpendicular to the axis.

    - URDF: continuous, fixed, floating, planar, prismatic, revolute
    - SDF: ball, fixed, gearbox, prismatic, revolute, revolute2, screw, universal
    - Dart: ball, free (=floating), euler, prismatic, weld (=fixed), revolute, universal
    - MuJoCo: ball, free (=floating), hinge (=revolute), slide (=prismatic)

    By default, we follow the convention expressed in URDF to describe the frames. That is, the child joint frame is
    described with respect to the parent joint/link frame.

    References:
        - [1] http://wiki.ros.org/urdf/XML/joint
    """

    def __init__(self, joint_id, name=None, dtype=None, limits=None, parent=None, child=None, axis=None,
                 position=None, orientation=None, friction=None, stiffness=None, damping=None, effort=None,
                 velocity=None):
        """
        Initialize the Joint class.

        Args:
            joint_id (int): unique joint id.
            name (str): name of the joint.
            dtype (str): joint type which can be selected from: [fixed, floating/free, prismatic, revolute/hinge,
              continuous, gearbox, revolute2, ball, screw, universal, planar].
            limits (tuple[float]], str): joint position limits.
            parent (Body, None): parent body/link that is connected to the joint.
            child (Body, None): child body/link that is connected to the joint.
            axis (tuple/list[float[3]], str): joint axis.
            position (list/tuple/np.array[float[3]], str): joint frame position.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): joint frame orientation.
            friction (float, str): joint friction loss due to dry friction.
            stiffness (float, str): joint stiffness coefficient.
            damping (float, str): joint damping coefficient.
            effort (float, str): joint maximum allowed effort.
            velocity (float, str): joint maximum allowed velocity.
        """
        self.id = int(joint_id)
        self.num_dofs = 0
        self.name = name
        self.dtype = dtype
        self.limits = limits
        self.parent = parent
        self.child = child
        self.axis = axis
        self.frame = Frame(position=position, orientation=orientation)
        self.friction = friction
        self.damping = damping
        self.effort = effort
        self.velocity = velocity

        self.init_position = None
        self.init_velocity = None

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def dtype(self):
        """Return the joint type."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """Set the joint type."""
        if dtype is not None:
            if not isinstance(dtype, str):
                raise TypeError("Expecting dtype to be a string")
            dtype = dtype.lower().strip()

            if dtype in {'fixed', 'weld'}:
                dtype = 'fixed'
                self.num_dofs = 0
            elif dtype in {'hinge', 'revolute'}:
                dtype = 'revolute'
                self.num_dofs = 1
            elif dtype in {'slide', 'prismatic'}:
                dtype = 'prismatic'
                self.num_dofs = 1
            elif dtype in {'free', 'floating'}:
                dtype = 'floating'
                self.num_dofs = 6
            elif dtype == 'ball':
                self.num_dofs = 3
            elif dtype == 'continuous':
                self.num_dofs = 1

        self._dtype = dtype

    @property
    def num_dofs(self):
        """Return the number of DoFs for the specified joint."""
        return self._num_dofs

    @num_dofs.setter
    def num_dofs(self, dofs):
        """Set the number of DoFs for the specified joint."""
        self._num_dofs = int(dofs)

    @property
    def limits(self):
        """Return the joint limits."""
        return self._limits

    @limits.setter
    def limits(self, limits):
        """Set the joint position limits."""
        if limits is not None:
            if isinstance(limits, str):
                limits = [lim for lim in limits.split()]
            limits = tuple(limits)
            limits = [float(lim) for lim in limits]
            if len(limits) != 2:
                raise ValueError("Expecting 2 floats for the joint limits")
        self._limits = limits

    @property
    def parent(self):
        """Return the parent body."""
        return self._parent

    @parent.setter
    def parent(self, parent):
        """Set the parent body."""
        if parent is not None and not isinstance(parent, Body):
            raise TypeError("Expecting the given 'parent' to be an instance of `Body` but got instead: "
                            "{}".format(type(parent)))
        self._parent = parent

    @property
    def child(self):
        """Return the child body."""
        return self._child

    @child.setter
    def child(self, child):
        """Set the child body."""
        if child is not None and not isinstance(child, Body):
            raise TypeError("Expecting the given 'child' to be an instance of `Body` but got instead: "
                            "{}".format(type(child)))
        self._child = child

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        """Set the joint axis."""
        if axis is not None:
            if isinstance(axis, str):
                axis = [float(ax) for ax in axis.split()]
            axis = tuple(axis)
            if len(axis) != 3:
                raise ValueError("Expecting the joint axis to be defined using 3 floats (x,y,z)")
        self._axis = axis

    @property
    def position(self):
        """Return the joint frame position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the joint frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the joint frame orientation expressed as RPY angles."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the joint frame orientation (which can be expressed as a rotation matrix, RPY angles. or a quaternion
        [x,y,z,w]."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the joint frame orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the joint frame orientation expressed as a quaternion [x,y,z,w]."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the joint frame orientation expressed as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the joint frame pose."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the joint frame pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix

    @property
    def friction(self):
        """Return the joint friction coefficient."""
        return self._friction

    @friction.setter
    def friction(self, friction):
        """Set the joint friction coefficient."""
        if friction is not None:
            friction = float(friction)
        self._friction = friction

    @property
    def damping(self):
        """Return the joint damping coefficient."""
        return self._damping

    @damping.setter
    def damping(self, damping):
        """Set the joint damping coefficient."""
        if damping is not None:
            damping = float(damping)
        self._damping = damping

    @property
    def effort(self):
        """Return the joint maximum effort."""
        return self._effort

    @effort.setter
    def effort(self, effort):
        """Set the joint maximum effort."""
        if effort is not None:
            effort = float(effort)
        self._effort = effort

    @property
    def velocity(self):
        """Return the joint maximum velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        """Set the joint maximum velocity."""
        if velocity is not None:
            velocity = float(velocity)
        self._velocity = velocity

    @property
    def init_position(self):
        """Return the initial joint position."""
        return self._init_position

    @init_position.setter
    def init_position(self, position):
        """Set the initial joint position."""
        if position is not None:
            if isinstance(position, str):
                position = np.asarray([float(s) for s in position.split()])
                if len(position) == 1:
                    position = position[0]
            elif isinstance(position, (tuple, list, np.ndarray)):
                position = np.asarray([float(s) for s in position])
                if len(position) == 1:
                    position = position[0]
            elif not isinstance(position, (float, int)):
                raise TypeError("Expecting the init_position to be a float, int, list, tuple or np.ndarray")
        self._init_position = position

    @property
    def init_velocity(self):
        """Return the joint initial velocity."""
        return self._init_velocity

    @init_velocity.setter
    def init_velocity(self, velocity):
        """Set the joint initial velocity."""
        if velocity is not None:
            if isinstance(velocity, str):
                velocity = np.asarray([float(s) for s in velocity.split()])
                if len(velocity) == 1:
                    velocity = velocity[0]
            elif isinstance(velocity, (tuple, list, np.ndarray)):
                velocity = np.asarray([float(s) for s in velocity])
                if len(velocity) == 1:
                    velocity = velocity[0]
            elif not isinstance(velocity, (float, int)):
                raise TypeError("Expecting the init_velocity to be a float, int, list, tuple or np.ndarray")
        self._init_velocity = velocity


class Inertia(object):
    r"""Inertia data structure

    This class represents an inertia matrix.
    """

    def __init__(self, ixx=1., iyy=1., izz=1., ixy=0., ixz=0., iyz=0., inertia=None):
        """
        Initialize the Inertia.

        Args:
            ixx (float, str): Ixx component of the inertia.
            iyy (float, str): Iyy component of the inertia.
            izz (float, str): Izz component of the inertia.
            ixy (float, str): Ixy component of the inertia.
            ixz (float, str): Ixz component of the inertia.
            iyz (float, str): Iyz component of the inertia.
            inertia (list/np.array[float[3/6]], np.array[float[3,3]], str, None): inertia matrix. If specified, the
              previous attributes won't be taken into account.
        """
        if inertia is not None:
            self.inertia = inertia
        else:
            self.ixx = ixx
            self.iyy = iyy
            self.izz = izz
            self.ixy = ixy
            self.ixz = ixz
            self.iyz = iyz

    @property
    def inertia(self):
        """Return the 6 components of the inertia [ixx, iyy, izz, ixy, ixz, iyz]."""
        return np.array([self.ixx, self.iyy, self.izz, self.ixy, self.ixz, self.iyz])

    @inertia.setter
    def inertia(self, inertia):
        """Set the inertia matrix."""
        if isinstance(inertia, str):
            inertia = [float(c) for c in inertia.split()]
        elif not isinstance(inertia, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'inertia' to be a tuple, list, np.array of 3/6/9 floats, but got "
                            "instead: {}".format(type(inertia)))
        inertia = np.asarray(inertia)
        if inertia.ndim == 2:  # 3x3
            inertia = inertia.reshape(-1)
        if len(inertia) == 3:
            self.ixx, self.iyy, self.izz = inertia
        elif len(inertia) == 6:
            self.ixx, self.iyy, self.izz, self.ixy, self.ixz, self.iyz = inertia
        elif len(inertia) == 9:
            I = inertia
            self.ixx, self.iyy, self.izz, self.ixy, self.ixz, self.iyz = I[0], I[4], I[8], I[1], I[2], I[5]
        else:
            raise ValueError("Expecting the given 'inertia' to be have a length of 3, 6, or 9, but got instead a "
                             "length of: {}".format(len(inertia)))

    @property
    def full_inertia(self):
        """Return the full inertia matrix."""
        return np.array([[self.ixx, self.ixy, self.ixz],
                         [self.ixy, self.iyy, self.iyz],
                         [self.ixz, self.iyz, self.izz]])

    @property
    def diagonal_inertia(self):
        """
        Return the aligned diagonal inertia (this corresponds to the eigenvalues of the full inertia matrix).

        Returns:
            np.array[float[3]]: principal moments of the inertia.

        References:
            - [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_matrix_in_different_reference_frames
        """
        inertia = self.full_inertia
        evals, evecs = np.linalg.eigh(inertia)
        return evals

    @property
    def principal_inertia(self):
        """
        Return the principal moments of the inertia, and the direction of the principal axes of the body.

        Returns:
            np.array[float[3]]: principal moments of the inertia.
            np.array[float[3,3]]: principal axes of the body.
        """
        inertia = self.full_inertia
        evals, evecs = np.linalg.eigh(inertia)
        return evals, evecs

    @property
    def principal_axes(self):
        """Return the directions of the principal axes of the body as a 3x3 matrix where each column represents an
        axis."""
        inertia = self.full_inertia
        evals, evecs = np.linalg.eigh(inertia)
        return evecs

    @property
    def ixx(self):
        """Return the Ixx component of the inertia."""
        return self._ixx

    @ixx.setter
    def ixx(self, ixx):
        """Set the Ixx component of the inertia."""
        if ixx is not None:
            ixx = float(ixx)
        self._ixx = ixx

    @property
    def iyy(self):
        """Return the Iyy component of the inertia."""
        return self._iyy

    @iyy.setter
    def iyy(self, iyy):
        """Set the Iyy component of the inertia."""
        if iyy is not None:
            iyy = float(iyy)
        self._iyy = iyy

    @property
    def izz(self):
        """Return the Izz component of the inertia."""
        return self._izz

    @izz.setter
    def izz(self, izz):
        """Set the Izz component of the inertia."""
        if izz is not None:
            izz = float(izz)
        self._izz = izz

    @property
    def ixy(self):
        """Return the Ixy component of the inertia."""
        return self._ixy

    @ixy.setter
    def ixy(self, ixy):
        """Set the Ixy component of the inertia."""
        if ixy is not None:
            ixy = float(ixy)
        self._ixy = ixy

    @property
    def ixz(self):
        """Return the Ixz component of the inertia."""
        return self._ixz

    @ixz.setter
    def ixz(self, ixz):
        """Set the Ixz component of the inertia."""
        if ixz is not None:
            ixz = float(ixz)
        self._ixz = ixz

    @property
    def iyz(self):
        """Return the Iyz component of the inertia."""
        return self._iyz

    @iyz.setter
    def iyz(self, iyz):
        """Set the Iyz component of the inertia."""
        if iyz is not None:
            iyz = float(iyz)
        self._iyz = iyz


class Inertial(object):
    r"""Inertial properties.

    The inertial element groups the mass, inertia, and the body CoM position and orientation. By default, we follow
    the convention expressed in URDF to describe the frames. That is, the inertial frame is described with respect to
    the link frame.

    Moments of inertia of popular shapes:

    - box: I = 1./12 * mass * np.array([h**2 + d**2, w**2 + d**2, w**2 + h**2]), where w=width, h=height, d=depth.
    - capsule: from https://www.gamedev.net/articles/programming/math-and-physics/capsule-inertia-tensor-r3856/
        - ixx = m_c * (h**2/12. + r**2/4.) + m_s * (2*r**2/5. + h**2/2. + 3*h*r/8.)
        - iyy = m_c * (h**2/12. + r**2/4.) + m_s * (2*r**2/5. + h**2/2. + 3*h*r/8.)
        - izz = m_c * r**2/2. + m_s * 2 * r**2 / 5.
        - where m_c = mass of the cylinder, m_s = mass of the sphere, r = radius of hemispheres, h = height of the
          cylinder.
    - cylinder: I = 1./12 * mass * np.array([3*r**2 + h**2, 3*r**2 + h**2, r**2]), where r=radius, h=height.
    - ellipsoid: I = 1./5 * mass * np.array([b**2 + c**2, a**2 + c**2, a**2 + b**2]), where a,b,c are the X,Y,Z radius.
    - sphere: I = 2./5 * mass * radius**2 * np.ones(3)
    - mesh: use ``trimesh`` library, after loading the mesh, you can access the moments of inertia with
      ``mesh.moment_inertia``.

    References:
        - [1] http://wiki.ros.org/urdf/XML/link
    """

    def __init__(self, mass=None, inertia=None, position=(0., 0., 0.), orientation=None):
        """
        Initialize the Inertial instance.

        Args:
            mass (float, str): mass value (in kg)
            inertia (str, list/tuple[float[3/6/9]], np.array[float[3/6/9]], np.array[float[3,3]], dict): inertia matrix
              represented in the body frame.
            position (np.array[float[3]], str): position of the inertial frame (center of mass).
            orientation (np.array[float[3]], str): orientation of the inertial frame expressed as roll-pitch-yaw angles.
        """
        self.mass = mass
        self.inertia = inertia
        self.frame = Frame(position=position, orientation=orientation)

    @property
    def mass(self):
        """Return the mass."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        """Set the mass."""
        if mass is not None:
            mass = float(mass)
        self._mass = mass

    @property
    def inertia(self):
        """Return the inertia instance."""
        return self._inertia

    @inertia.setter
    def inertia(self, inertia):
        """Set the inertia matrix."""
        if inertia is not None:
            if isinstance(inertia, str):
                inertia = Inertia(inertia=inertia)
            if isinstance(inertia, (list, tuple, np.ndarray)):
                if isinstance(inertia, np.ndarray) and inertia.ndim == 2:
                    if inertia.shape != (3, 3):
                        raise ValueError("Expecting a 3x3 inertia matrix")
                    inertia = Inertia(ixx=inertia[0, 0], ixy=inertia[0, 1], ixz=inertia[0, 2],
                                      iyy=inertia[1, 1], iyz=inertia[1, 2], izz=inertia[2, 2])
                else:
                    if len(inertia) == 3:
                        inertia = Inertia(ixx=inertia[0], iyy=inertia[1], izz=inertia[2])
                    elif len(inertia) == 6:
                        inertia = Inertia(ixx=inertia[0], iyy=inertia[1], izz=inertia[2], ixy=inertia[3],
                                          ixz=inertia[4], iyz=inertia[5])
                    elif len(inertia) == 9:
                        inertia = Inertia(ixx=inertia[0], ixy=inertia[1], ixz=inertia[2], iyy=inertia[4],
                                          iyz=inertia[5], izz=inertia[8])
            elif isinstance(inertia, dict):
                inertia = Inertia(**inertia)
        self._inertia = inertia

    @property
    def full_inertia(self):
        """Return the full inertia matrix."""
        rot = self.rot
        return rot.dot(self._inertia.full_inertia).dot(rot.T)

    @property
    def principal_inertia(self):
        """
        Return the principal moments of the inertia, and the direction of the principal axes of the body.

        Returns:
            np.array[float[3]]: principal moments of the inertia.
            np.array[float[3,3]]: principal axes of the body.
        """
        evals, evecs = self.inertia.principal_inertia
        return evals, self.rot.dot(evecs)

    @property
    def diagonal_inertia(self):
        """
        Return the aligned inertia = principal moments of inertia.

        Returns:
            np.array[float[3]]: principal moments of the inertia.

        References:
            - [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_matrix_in_different_reference_frames
        """
        return self.inertia.diagonal_inertia

    @property
    def principal_axes(self):
        """Return the directions of the principal axes of the body as a 3x3 matrix where each column represents an
        axis."""
        evecs = self.inertia.principal_axes
        return self.rot.dot(evecs)

    @property
    def position(self):
        """Return the CoM position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the CoM position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the inertial orientation."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the inertial orientation as RPY angles."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the inertial orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the inertial orientation expressed as a quaternion [x,y,z,w]."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the inertial orientation given as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the inertial pose."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the inertial pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix

    @staticmethod
    def compute_mass_from_density(shape, dimensions=None, density=1000, volume=None, mesh=None):
        """
        Compute the mass from the density and the shape type.

        Args:
            shape (str): shape type which can be selected from {'sphere', 'box', 'capsule', 'cylinder', 'ellipsoid',
              'mesh'}.
            dimensions (list[float[:3]]): shape dimensions, or scale factor if mesh. If the volume is not provided,
              it will use the specified dimensions.
            density (float): density (by default, it is the density of water ~ 1000kg/m^3).
            volume (float, None): if provided, the returned mass will be the given density times the volume.
            mesh (str, trimesh.Trimesh): mesh filename or mesh instance. Only valid if :attr:`shape` = 'mesh'.

        Returns:
            float, None: mass (in kg). Return None if the specified shape is not supported.
        """
        if volume is None:
            volume = Inertial.compute_volume(shape=shape, dimensions=dimensions, mesh=mesh)
        if volume is not None:
            return density * volume

    @staticmethod
    def compute_volume(shape, dimensions, mesh=None):
        """
        Compute the volume given the shape type and its dimensions.

        Args:
            shape (str): shape type which can be selected from {'sphere', 'box', 'capsule', 'cylinder', 'ellipsoid',
              'mesh'}.
            dimensions (list[float[:3]]): shape dimensions, or scale factor if mesh.
            mesh (str, trimesh.Trimesh): mesh filename or mesh instance. Only valid if :attr:`shape` = 'mesh'.

        Returns:
            float, None: total volume of the specified shape. Return None if the specified shape is not supported.
        """
        if shape == 'box':
            w, h, d = dimensions  # width, height, depth
            volume = w * h * d
        elif shape == 'capsule':
            r, h = dimensions  # radius, height
            sphere_volume = 4. / 3 * np.pi * r ** 3
            cylinder_volume = np.pi * r ** 2 * h
            volume = sphere_volume + cylinder_volume
        elif shape == 'cylinder':
            r, h = dimensions  # radius, height
            volume = np.pi * r ** 2 * h
        elif shape == 'ellipsoid':
            a, b, c = dimensions
            volume = 4. / 3 * np.pi * a * b * c
        elif shape == 'mesh':
            if isinstance(dimensions, (list, tuple, np.ndarray)):
                dimensions = dimensions[0]
            scale = dimensions  # scale
            if isinstance(mesh, str):
                mesh = trimesh.load(mesh)
            elif not isinstance(mesh, trimesh.Trimesh):
                raise TypeError("Expecting the given 'mesh' to be an instance of `trimesh.Trimesh`, instead got: "
                                "{}".format(type(mesh)))
            mesh.apply_scale(scale)
            volume = mesh.volume  # in m^3  (in trimesh: mash.mass = mash.volume, i.e. density = 1)
            # volume *= scale ** 3  # the scale is for each dimension
            mesh.apply_scale(1./scale)
        elif shape == 'sphere':
            if isinstance(dimensions, (list, tuple, np.ndarray)):
                dimensions = dimensions[0]
            r = dimensions  # radius
            volume = 4. / 3 * np.pi * r ** 3
        else:
            volume = None
        return volume

    @staticmethod
    def compute_inertia(shape, dimensions, mass=None, density=1000, mesh=None):
        """
        Compute the principal moments of inertia for the specified shape.

        Args:
            shape (str): shape type, can be selected from {}
            dimensions (list[float[:3]]): dimensions of the shape.
            mass (float, None): mass of the shape. If not provided, the density will be used.
            density (float): density (by default, it is the density of water ~ 1000kg/m^3).
            mesh (str, trimesh.Trimesh): mesh filename or mesh instance. Only valid if :attr:`shape` = 'mesh'.

        Returns:
            np.array[float[3]], None: principal moments of inertia. None if the specified shape is not supported.
        """
        # compute mass if necessary
        if mass is None:
            mass = Inertial.compute_mass_from_density(shape=shape, dimensions=dimensions, density=density, mesh=mesh)

        # compute inertia
        if shape == 'box':
            inertia = get_inertia_of_box(mass, size=dimensions, full=False)
        elif shape == 'capsule':
            r, h = dimensions  # radius, height
            inertia = get_inertia_of_capsule(mass, radius=r, height=h, full=False)
        elif shape == 'cylinder':
            r, h = dimensions  # radius, height
            inertia = get_inertia_of_cylinder(mass, radius=r, height=h, full=False)
        elif shape == 'ellipsoid':
            a, b, c = dimensions
            inertia = get_inertia_of_ellipsoid(mass, a=a, b=b, c=c, full=False)
        elif shape == 'mesh':
            if isinstance(dimensions, (list, tuple, np.ndarray)):
                dimensions = dimensions[0]
            scale = dimensions  # scale
            inertia = get_inertia_of_mesh(mesh=mesh, mass=mass, scale=scale, full=False)
        elif shape == 'sphere':
            if isinstance(dimensions, (list, tuple, np.ndarray)):
                dimensions = dimensions[0]
            radius = dimensions
            inertia = get_inertia_of_sphere(mass=mass, radius=radius, full=False)
        else:
            inertia = None
        return inertia

    def __add__(self, other):
        """
        Combine two Inertial elements together.

        This is done in 3 steps:
        1. find the combined CoM
        2. find the moments of inertia of each object through that point using the parallel axis theorem [1]
        3. combine the moments by adding the new tensors.

        Args:
            other (Inertial): other inertial elements.

        Returns:
            Inertial: the combined inertial element.

        References:
            - [1] Parallel axis theorem: https://en.wikipedia.org/wiki/Parallel_axis_theorem
        """
        # check the type
        if not isinstance(other, Inertial):
            raise TypeError("Expecting the given 'other' inertial element to be an instance of `Inertial` but got "
                            "instead: {}".format(type(other)))

        # check the attribute of each Inertial
        m1, m2 = self.mass, other.mass
        I1, I2 = self.inertia, other.inertia
        p1, p2 = self.position, other.position
        r1, r2 = self.rot, other.rot
        if m1 is None or m2 is None:
            raise ValueError("The mass is not specified for this inertial element or the other one.")
        if I1 is None or I2 is None:
            raise ValueError("The inertia is not specified for this inertial element or the other one.")
        if p1 is None or p2 is None:
            raise ValueError("The CoM position is not specified for this inertial element or the other one.")
        if r1 is None or r2 is None:
            rotations = None
        else:
            if r1 is None:
                r1 = np.identity(3)
            if r2 is None:
                r2 = np.identity(3)
            rotations = [r1, r2]

        # combine inertial elements and return it
        mass, com, inertia = combine_inertias(coms=[p1, p2], masses=[m1, m2], inertias=[I1, I2], rotations=rotations)
        inertial = Inertial(mass=mass, inertia=inertia, position=com)
        return inertial

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        inertial = self.__add__(other)
        self.mass = inertial.mass
        self.inertia = inertial.inertia
        self.frame = Frame(position=inertial.position, orientation=inertial.orientation)


class Geometry(object):  # Shape
    """Geometry: plane, sphere, box, mesh, cylinder, ellipsoid, capsule, cone, heightmap, etc.

    The Geometry represents a Shape that is set to the `Visual` and `Collision` elements.

    Here are different shapes available in different file formats:
    - URDF: box, cylinder, mesh, sphere
    - SDF: box, cylinder, heightmap, image, mesh, plane, polyline, sphere
    - Skel: box, capsule, cone, cylinder, ellipsoid, mesh, multi_sphere, sphere
    - MuJoCo: box, capsule, cylinder, ellipsoid, hfield (=height field), mesh, plane, sphere

    Depending on the type, the size can be a float, or a list of float:
    - box: size = float[3] (length in each direction)
    - cylinder: size = float[2] (radius and length/height)
    - capsule: size = float[2] (radius and length/height)
    - ellipsoid: size = float[3] (radius in X, Y, Z)
    - mesh: size = scale = float[3] (scale in each direction), or usually size = scale = float (total scale factor)
    - plane: size = float[2] (length in X and Y), if float[3] the third element represents the distance between the
      lines on the plane grid.
    - sphere: size = float (radius)
    """

    def __init__(self, dtype=None, size=None, filename=None):
        r"""
        Initialize the Geometry which is used for the `Visual` and `Collision` elements.

        Args:
            dtype (str): primitive shape type, this can be selected from ['box', 'cylinder', 'capsule', 'cone',
              'ellipsoid', 'mesh', 'plane', 'sphere', 'heightmap'].
            size (list[float[:3]], str): size/dimension of the shape.
            filename (str): path to the mesh file.
        """
        self.dtype = dtype
        self.size = size            # depending on the type it can be different size
        self.filename = filename

    @property
    def size(self):
        """Return the size/dimension of the geometric shape."""
        return self._size

    @size.setter
    def size(self, size):
        """Set the size/dimension of the geometric shape."""
        if size is not None:
            if isinstance(size, str):
                size = [float(s) for s in size.split()]
                if len(size) == 1:
                    size = size[0]
            elif isinstance(size, (tuple, list, np.ndarray)):
                size = [float(s) for s in size]
                if len(size) == 1:
                    size = size[0]
            elif not isinstance(size, (float, int)):
                raise TypeError("Expecting the size to be a float, int, list, tuple or np.ndarray")
        self._size = size

    # alias
    dimensions = size

    @property
    def format(self):
        """Return the filename format extension for the mesh."""
        if self.filename is not None:
            return self.filename.split('.')[-1]


# alias
Shape = Geometry


class Visual(object):
    r"""visual parameters for body.

    By default, we follow the convention expressed in URDF to describe the frames. That is, the visual frame is
    described with respect to the link frame.

    References:
        - [1] http://wiki.ros.org/urdf/XML/link
    """

    def __init__(self, name=None, dtype=None, size=None, color=None, filename=None, position=None, orientation=None,
                 material_name=None, texture=None, diffuse=None, specular=None, emissive=None):
        """
        Initialize the visual shape of a body.

        Args:
            name (str): name of the visual shape.
            dtype (str): primitive shape type.
            size (list[float[:3]], str): size/dimension of the shape.
            color (list[float], str): RGB(A) ambient color.
            filename (str): path to the mesh file.
            position (list/tuple/np.array[float[3]], str): visual frame position.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): visual frame orientation.
            material_name (str): name of the material.
            color (list[float], str): RGB(A) ambient color.
            texture (str): path to the texture.
            diffuse (list[float], str): diffuse color of an object; the color of an object under a pure white light.
            specular (list[float], str): specular color of an object; the color and intensity of a highlight from a
              specular reflection (higher values make an object more shiny).
            emissive (list[float], str): emissive color of an object; the color where the light appears to being
              emitted from the object.
        """
        self.name = name
        self.geometry = Geometry(dtype=dtype, size=size, filename=filename)
        self.frame = Frame(position=position, orientation=orientation)
        self.material = Material(name=material_name, color=color, texture=texture, diffuse=diffuse, specular=specular,
                                 emissive=emissive)

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def dtype(self):
        """Return the primitive shape type."""
        return self.geometry.dtype

    @dtype.setter
    def dtype(self, dtype):
        """Set the primitive shape type."""
        self.geometry.dtype = dtype

    @property
    def size(self):
        """Return the shape size / dimension."""
        return self.geometry.size

    @size.setter
    def size(self, size):
        """Set the shape size / dimension."""
        self.geometry.size = size

    @property
    def filename(self):
        """Return the path to the mesh file."""
        return self.geometry.filename

    @filename.setter
    def filename(self, filename):
        """Set the path to the mesh file."""
        self.geometry.filename = filename

    @property
    def color(self):
        """Return the ambient color."""
        return self.material.color

    @color.setter
    def color(self, color):
        """Set the ambient color."""
        self.material.color = color

    @property
    def rgb(self):
        """Return the RGB ambient color."""
        return self.material.rgb

    @property
    def rgba(self):
        """Return the RGBA ambient color."""
        return self.material.rgba

    @property
    def texture(self):
        """Return the path to the texture."""
        return self.material.texture

    @texture.setter
    def texture(self, texture):
        """Set the texture to the material."""
        self.material.texture = texture

    @property
    def format(self):
        """Return the filename format extension for the mesh."""
        if self.filename is not None:
            return self.filename.split('.')[-1]

    @property
    def position(self):
        """Return the visual frame position."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the visual frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the visual frame orientation (expressed as RPY angles)."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the visual frame orientation (which can be expressed as a rotation matrix, RPY angles, or a quaternion
        [x,y,z,w]."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the visual frame orientation expressed as RPY angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the visual frame orientation expressed as a quaternion [x,y,z,w]."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the visual frame orientation expressed as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the visual frame pose."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the visual frame pose."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix


class Collision(object):
    r"""Collision parameters for body.

    By default, we follow the convention expressed in URDF to describe the frames. That is, the collision frame is
    described with respect to the link frame.

    References:
        - [1] http://wiki.ros.org/urdf/XML/link
    """

    def __init__(self, name=None, dtype=None, size=None, filename=None, position=None, orientation=None):
        """
        Initialize the collision shape of a body.

        Args:
            name (str): name of the collision shape.
            dtype (str): primitive shape type.
            size (list[float[:3]], str): size/dimension of the shape.
            filename (str): path to the mesh file for the collision shape.
            position (list/tuple/np.array[float[3]], str): collision frame position.
            orientation (list/tuple/np.array[float[3/4/9]], np.array[float[3,3]], str): collision frame orientation.
        """
        self.name = name
        self.geometry = Geometry(dtype=dtype, size=size, filename=filename)
        self.frame = Frame(position=position, orientation=orientation)

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def dtype(self):
        """Return the primitive shape type."""
        return self.geometry.dtype

    @dtype.setter
    def dtype(self, dtype):
        """Set the primitive shape type."""
        self.geometry.dtype = dtype

    @property
    def size(self):
        """Return the shape size / dimension."""
        return self.geometry.size

    @size.setter
    def size(self, size):
        """Set the shape size / dimension."""
        self.geometry.size = size

    @property
    def filename(self):
        """Return the path to the mesh file."""
        return self.geometry.filename

    @filename.setter
    def filename(self, filename):
        """Set the path to the mesh file."""
        self.geometry.filename = filename

    @property
    def format(self):
        """Return the format of the mesh file."""
        if self.filename is not None:
            return self.filename.split('.')[-1]

    @property
    def position(self):
        """Return the position of the collision frame."""
        return self.frame.position

    @position.setter
    def position(self, position):
        """Set the collision frame position."""
        self.frame.position = position

    @property
    def orientation(self):
        """Return the collision frame orientation as RPY angles."""
        return self.frame.orientation

    @orientation.setter
    def orientation(self, orientation):
        """Set the collision frame orientation (which can be represented as RPY angles, rotation matrix, or a
        quaternion [x,y,z,w])."""
        self.frame.orientation = orientation

    @property
    def rpy(self):
        """Return the orientation of the collision frame as a roll-pitch-yaw angles."""
        return self.frame.rpy

    @property
    def quaternion(self):
        """Return the orientation of the collision frame as a quaternion [x,y,z,w]."""
        return self.frame.quaternion

    @property
    def rot(self):
        """Return the orientation of the collision frame as a rotation matrix."""
        return self.frame.rot

    @property
    def pose(self):
        """Return the pose of the collision frame."""
        return self.frame.pose

    @pose.setter
    def pose(self, pose):
        """Set the pose of the collision frame."""
        self.frame.pose = pose

    @property
    def homogeneous(self):
        """Return the frame homogeneous matrix. Note that if the orientation is None it will be set to the identity
        matrix, and if the position is None, it will be set to the zero vector."""
        return self.frame.homogeneous

    @homogeneous.setter
    def homogeneous(self, matrix):
        """Set the given homogeneous matrix."""
        self.frame.homogeneous = matrix


class Material(object):
    r"""Material info.

    Type of colors:
    - ambient: color of an object when no lights are pointing at it.
    - diffuse: color of an object under a pure white light.
    - specular: color and intensity of a highlight from a specular reflection (higher values make an object more shiny).
    - emissive: color where the light appears to being emitted from the object.
    """

    def __init__(self, name=None, color=None, texture=None, diffuse=None, specular=None, emissive=None):
        """
        Initialize the Material.

        Args:
            name (str): name of the material.
            color (list[float[:4]], str): RGB(A) ambient color.
            texture (str): path to the texture.
            diffuse (list[float], str): diffuse color of an object; the color of an object under a pure white light.
            specular (list[float], str): specular color of an object; the color and intensity of a highlight from a
              specular reflection (higher values make an object more shiny).
            emissive (list[float], str): emissive color of an object; the color where the light appears to being
              emitted from the object.
        """
        self.name = name
        self.color = color  # ambient color
        self.texture = texture

        # RGBA color
        self.diffuse = diffuse
        self.specular = specular
        self.emissive = emissive

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def texture(self):
        """Return the texture."""
        return self._texture

    @texture.setter
    def texture(self, texture):
        """Set the texture."""
        if texture is not None and not isinstance(texture, str):
            raise TypeError("Expecting the given 'texture' to be a string, instead got: {}".format(type(texture)))
        self._texture = texture

    @staticmethod
    def _check_color(color):
        """Check the given color (its type and length) and convert it to a tuple of float."""
        if color is not None:
            if isinstance(color, str):  # e.g. '0.5 0.1 1. 1.'
                color = [float(c) for c in color.split()]
            if isinstance(color, (list, tuple)):
                if len(color) < 3 or len(color) > 4:
                    raise ValueError("Expecting the color to be of length 3 (RGB) or 4 (RGBA), but got a length of: "
                                     "{}".format(len(color)))
                color = tuple(color)
            else:
                raise TypeError("Expecting the color to be a tuple or list of 3 or 4 float, but got instead: "
                                "type={}".format(type(color)))
        return color

    @property
    def color(self):
        """Return the ambient color."""
        return self._color

    @color.setter
    def color(self, color):
        """Set the ambient color."""
        self._color = self._check_color(color)

    # alias
    ambient = color

    @property
    def rgb(self):
        """Return the RGB ambient color."""
        if self.color is None:
            return None  # 0.5, 0.5, 0.5
        return tuple(self.color[:3])

    @property
    def rgba(self):
        """Return the RGBA ambient color."""
        if self.color is None:
            return None  # 0.5, 0.5, 0.5, 1.
        if len(self.color) == 3:
            return tuple(self.color) + (1.,)
        return tuple(self.color)

    @property
    def diffuse(self):
        """Return the diffuse color."""
        return self._diffuse

    @diffuse.setter
    def diffuse(self, diffuse):
        """Set the diffuse color."""
        self._diffuse = self._check_color(diffuse)

    @property
    def specular(self):
        """Return the specular color."""
        return self._specular

    @specular.setter
    def specular(self, specular):
        """Set the specular color."""
        self._specular = self._check_color(specular)

    @property
    def emissive(self):
        """Return the emissive color."""
        return self._emissive

    @emissive.setter
    def emissive(self, emissive):
        """Set the emissive color."""
        self._emissive = self._check_color(emissive)


class Sensor(object):
    r"""Sensor (abstract) class.

    """

    def __init__(self, name=None, sensors=[]):
        """
        Initialize the sensor.

        Args:
            name (str): name of the sensor.
            sensors (list[Sensor]): inner list of sensors.
        """
        self.name = name
        self.sensors = sensors

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def num_sensors(self):
        """Return the number of inner sensors."""
        return len(self.sensors)


class Actuator(object):  # Motor
    r"""Actuator/Motor (abstract) class.

    """

    def __init__(self, name=None, actuators=[]):
        """
        Initialize the actuator/motor.

        Args:
            name (str): name of the actuator/motor.
            actuators (list[Sensor]): inner list of actuators.
        """
        self.name = name
        self.actuators = actuators

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("Expecting the given 'name' to be a string, instead got: {}".format(type(name)))
        self._name = name

    @property
    def num_actuators(self):
        """Return the number of inner actuators."""
        return len(self.actuators)


class Heightmap(object):
    r"""Heightmap (abstract) class.

    """
    pass


class Constraint(object):
    r"""Constraint data structure.

    This allows to define a constraint.
    """
    pass
