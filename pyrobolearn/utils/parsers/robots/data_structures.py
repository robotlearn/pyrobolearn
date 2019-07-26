#!/usr/bin/env python
"""Provide the data structures that are shared among the various parsers and converters.
"""

import numpy as np
from collections import OrderedDict

from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_quaternion_from_rpy, get_matrix_from_rpy


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

    def __init__(self, world=None, physics_engine=None):
        self.world = world
        self.engine = physics_engine
        self.physics = physics_properties

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, world):
        if world is not None and not isinstance(world, World):
            raise TypeError("Expecting the world to be an instance of `World`, but got instead: "
                            "{}".format(type(world)))
        self._world = world

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        if engine is not None and not isinstance(engine, PhysicsEngine):
            raise TypeError("Expecting the engine to be an instance of `PhysicsEngine`, but got instead: "
                            "{}".format(type(engine)))
        self._engine = engine


class PhysicsEngine(object):
    r"""Physics Engine properties.

    This include number of iterations, solver used, tolerance, timesteps, etc.
    """

    def __init__(self, timestep=None):
        self.timestep = timestep
        self.num_iterations = None
        self.solver = None
        self.tolerance = None


class World(object):
    r"""World data structure.

    World frame (robotics convention with the right-hand rule):
    - the x axis points forward
    - the y axis points to the left
    - the z axis points upward
    """

    def __init__(self, name=None):
        self.name = name
        self.trees = OrderedDict()
        self.physics = None

    @property
    def physics(self):
        return self._physics

    @physics.setter
    def physics(self, physics):
        if physics is not None and not isinstance(physics, Physics):
            raise TypeError("Expecting the physics to be an instance of `Physics`, but got instead: "
                            "{}".format(type(physics)))
        self._physics = physics


class Light(object):
    r"""Light data structure.

    Type of light: point, directional, and spot
    """

    def __init__(self, name=None, dtype=None, cast_shadows=None, diffuse=None, specular=None, attenuation=None,
                 direction=None, spot=None, position=None, orientation=None):
        """
        Initialize the Light data structure.

        Args:
            name (str): unique name for the light.
            dtype (str): type of light, select between {'point', 'directional', 'spot'}
            cast_shadows (bool): if True, it will cast shadows.
            diffuse (tuple of 4 float, np.array[4]): diffuse light (RGBA) color.
            specular (tuple of 4 float, np.array[4]): specular light (RGBA) color.
            attenuation: light attenuation
            direction (np.array[3]): direction of the light if dtype='directional' or dtype='spot'.
            spot: spot light parameters
            position (tuple/list of 3 float, np.array[3]): position of the light in the world.
            orientation (tuple/list of 3 float, np.array[3]): orientation of the light in the world.
        """
        self.name = name
        self.dtype =dtype
        self.shadows = cast_shadows
        self.diffuse = diffuse
        self.specular = specular
        self.attenuation = attenuation
        self.direction = direction
        self.spot = spot
        self.position = position
        self.orientation = orientation


class Physics(object):
    r"""Physical properties of the world.

    This includes gravity, friction, viscosity, etc.
    """

    def __init__(self, gravity=(0., 0., -9.81)):
        self.gravity = gravity


class Frame(object):
    r"""Reference Frame"""

    def __init__(self, position=None, orientation=None, dtype=None, right_handed=True, forward_axis=(1., 0., 0.),
                 up_axis=(0., 0., 1.)):
        self.position = position
        self.orientation = orientation
        self.dtype = dtype  # world frame, body frame, joint frame, inertial frame, etc.
        self.right_handed = right_handed
        self.forward_axis = forward_axis
        self.up_axis = up_axis


class Tree(object):
    r"""Tree data structure."""

    def __init__(self, name=None, root=None):
        self.name = name
        self.root = root
        self.bodies = OrderedDict()
        self.joints = OrderedDict()
        self.materials = {}
        self.position = None
        self.orientation = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if position is not None:
            if isinstance(position, str):
                position = [float(p) for p in position.split()]
            position = np.asarray(position)
        self._position = position

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if orientation is not None:
            if isinstance(orientation, str):
                orientation = [float(o) for o in orientation.split()]
            if len(orientation) == 4:  # quaternion
                orientation = get_rpy_from_quaternion(orientation)
            if len(orientation) == 3:  # rpy
                pass
            orientation = np.asarray(orientation)
        self._orientation = orientation

    @property
    def rpy(self):
        return self._orientation

    @property
    def quaternion(self):
        return get_quaternion_from_rpy(self._orientation)

    @property
    def rot(self):
        return get_matrix_from_rpy(self.rpy)

    @property
    def pose(self):
        return self.position, self.orientation

    @pose.setter
    def pose(self, pose):
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


class Body(object):
    r"""Body / Link data structure."""

    def __init__(self, body_id, name=None):
        self.id = body_id
        self.name = name

        # inner bodies / links
        self.bodies = []
        self.joints = OrderedDict()

        self.inertial = None
        self.visual = None
        self.collision = None


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
    """

    def __init__(self, joint_id, name=None, dtype=None, limits=None, parent=None, child=None, axis=None,
                 position=None, orientation=None, friction=None, damping=None, effort=None, velocity=None):
        self.id = joint_id
        self.name = name
        self.dtype = dtype
        self.limits = limits
        self.parent = parent
        self.child = child
        self.axis = axis
        self.position = position
        self.orientation = orientation
        self.friction = friction
        self.damping = damping
        self.effort = effort
        self.velocity = velocity

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        if limits is not None:
            if isinstance(limits, str):
                limits = [lim for lim in limits.split()]
            limits = tuple(limits)
            limits = [float(lim) for lim in limits]
            if len(limits) != 2:
                raise ValueError("Expecting 2 floats for the joint limits")
        self._limits = limits

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        if axis is not None:
            if isinstance(axis, str):
                axis = [float(ax) for ax in axis.split()]
            axis = tuple(axis)
            if len(axis) != 3:
                raise ValueError("Expecting the joint axis to be defined using 3 floats (x,y,z)")
        self._axis = axis

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if position is not None:
            if isinstance(position, str):
                position = [float(p) for p in position.split()]
            position = np.asarray(position)
        self._position = position

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if orientation is not None:
            if isinstance(orientation, str):
                orientation = [float(o) for o in orientation.split()]
            if len(orientation) == 4:  # quaternion
                orientation = get_rpy_from_quaternion(orientation)
            if len(orientation) == 3:  # rpy
                pass
            orientation = np.asarray(orientation)
        self._orientation = orientation

    @property
    def rpy(self):
        return self._orientation

    @property
    def quaternion(self):
        return get_quaternion_from_rpy(self._orientation)

    @property
    def rot(self):
        return get_matrix_from_rpy(self.rpy)

    @property
    def pose(self):
        return self.position, self.orientation

    @pose.setter
    def pose(self, pose):
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
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, friction):
        if friction is not None:
            friction = float(friction)
        self._friction = friction

    @property
    def damping(self):
        return self._damping

    @damping.setter
    def damping(self, damping):
        if damping is not None:
            damping = float(damping)
        self._damping = damping

    @property
    def effort(self):
        return self._effort

    @effort.setter
    def effort(self, effort):
        if effort is not None:
            effort = float(effort)
        self._effort = effort

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        if velocity is not None:
            velocity = float(velocity)
        self._velocity = velocity


class Inertia(object):
    r"""Inertia data structure"""

    def __init__(self, ixx=1., iyy=1., izz=1., ixy=0., ixz=0., iyz=0.):
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz
        self.ixy = ixy
        self.ixz = ixz
        self.iyz = iyz

    @property
    def full_inertia(self):
        return np.array([[self.ixx, self.ixy, self.ixz],
                         [self.ixy, self.iyy, self.iyz],
                         [self.ixz, self.iyz, self.izz]])

    @property
    def diagonal_inertia(self):
        """Aligned inertia.

        Returns:
            np.array[3]: principal moments of the inertia.

        References:
            - [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_matrix_in_different_reference_frames
        """
        inertia = self.full_inertia
        evals, evecs = np.linalg.eigh(inertia)
        return evals

    @property
    def principal_inertia(self):
        """Return the principal moments of the inertia (np.array[3]), and the direction of the principal axes of the
        body (np.array[3,3])."""
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
        return self._ixx

    @ixx.setter
    def ixx(self, ixx):
        if ixx is None:
            ixx = float(ixx)
        self._ixx = ixx

    @property
    def iyy(self):
        return self._iyy

    @iyy.setter
    def iyy(self, iyy):
        if iyy is not None:
            iyy = float(iyy)
        self._iyy = iyy

    @property
    def izz(self):
        return self._izz

    @izz.setter
    def izz(self, izz):
        if izz is not None:
            izz = float(izz)
        self._izz = izz

    @property
    def ixy(self):
        return self._ixy

    @ixy.setter
    def ixy(self, ixy):
        if ixy is not None:
            ixy = float(ixy)
        self._ixy = ixy

    @property
    def ixz(self):
        return self._ixz

    @ixz.setter
    def ixz(self, ixz):
        if ixz is not None:
            ixz = float(ixz)
        self._ixz = ixz

    @property
    def iyz(self):
        return self._iyz

    @iyz.setter
    def iyz(self, iyz):
        if iyz is not None:
            iyz = float(iyz)
        self._iyz = iyz


class Inertial(object):
    r"""Inertial parameters."""

    def __init__(self, mass=None, inertia=None, position=(0., 0., 0.), orientation=(0., 0., 0.)):
        """

        Args:
            mass (float): mass value (in kg)
            inertia (str, list / tuple of 3/6/9 float, np.ndarray[3/6/9], np.ndarray[3,3]): inertia matrix represented
                in the body frame.
            position (np.array[3], str): position of the center of mass.
            orientation (np.array[3], str): rotation expressed as roll-pitch-yaw angles.
        """
        self.mass = mass
        self.inertia = inertia
        self.position = position
        self.orientation = orientation

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass):
        if mass is not None:
            mass = float(mass)
        self._mass = mass

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, inertia):
        if inertia is not None:
            if isinstance(inertia, str):
                inertia = inertia.split()
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
        rot = self.rot
        return rot.dot(self._inertia.full_inertia).dot(rot.T)

    @property
    def principal_inertia(self):
        """Return the principal moments of the inertia (np.array[3]), and the direction of the principal axes of the
        body (np.array[3,3])."""
        evals, evecs = self.inertia.principal_inertia
        return evals, self.rot.dot(evecs)

    @property
    def diagonal_inertia(self):
        """Aligned inertia.

        Returns:
            np.array[3]: principal moments of the inertia.

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
        return self._position

    @position.setter
    def position(self, position):
        if position is not None:
            if isinstance(position, str):
                position = [float(p) for p in position.split()]
            position = np.asarray(position)
        self._position = position

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if orientation is not None:
            if isinstance(orientation, str):
                orientation = [float(o) for o in orientation.split()]
            if len(orientation) == 4:  # quaternion
                orientation = get_rpy_from_quaternion(orientation)
            if len(orientation) == 3:  # rpy
                pass
            orientation = np.asarray(orientation)
        self._orientation = orientation

    @property
    def rpy(self):
        return self._orientation

    @property
    def quaternion(self):
        return get_quaternion_from_rpy(self._orientation)

    @property
    def rot(self):
        return get_matrix_from_rpy(self.rpy)

    @property
    def pose(self):
        return self.position, self.orientation

    @pose.setter
    def pose(self, pose):
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


class Geometry(object):  # Shape
    """Geometry: plane, sphere, box, mesh, cylinder, ellipsoid, capsule, heightmap, etc."""

    def __init__(self, dtype=None, size=None, filename=None):
        self.dtype = dtype
        self.size = size            # depending on the type it can be different size
        self.filename = filename

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
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

    @property
    def format(self):
        """Return the filename format extension for the mesh."""
        if self.filename is not None:
            return self.filename.split('.')[-1]


# alias
Shape = Geometry


class Visual(object):
    r"""visual parameters for body."""

    def __init__(self, name=None, dtype=None, size=None, color=None, filename=None, position=None, orientation=None,
                 material=None):
        self.name = name
        self.geometry = Geometry(dtype=dtype, size=size, filename=filename)
        self.color = color
        self.position = position
        self.orientation = orientation
        self.material = material

    @property
    def dtype(self):
        return self.geometry.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.geometry.dtype = dtype

    @property
    def size(self):
        return self.geometry.size

    @size.setter
    def size(self, size):
        self.geometry.size = size

    @property
    def filename(self):
        return self.geometry.filename

    @filename.setter
    def filename(self, filename):
        self.geometry.filename = filename

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        if color is not None:
            if isinstance(color, str):  # e.g. '0.5 0.1 1. 1.'
                color = (float(c) for c in color.split())
            if not isinstance(color, (list, tuple)):
                raise TypeError("Expecting the color to be a tuple or list of 3 or 4 float")
            else:
                color = tuple(color)
        self._color = color

    @property
    def format(self):
        """Return the filename format extension for the mesh."""
        if self.filename is not None:
            return self.filename.split('.')[-1]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if position is not None:
            if isinstance(position, str):
                position = [float(p) for p in position.split()]
            position = np.asarray(position)
        self._position = position

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if orientation is not None:
            if isinstance(orientation, str):
                orientation = [float(o) for o in orientation.split()]
            if len(orientation) == 4:  # quaternion
                orientation = get_rpy_from_quaternion(orientation)
            if len(orientation) == 3:  # rpy
                pass
            orientation = np.asarray(orientation)
        self._orientation = orientation

    @property
    def rpy(self):
        return self._orientation

    @property
    def quaternion(self):
        return get_quaternion_from_rpy(self._orientation)

    @property
    def rot(self):
        return get_matrix_from_rpy(self.rpy)

    @property
    def pose(self):
        return self.position, self.orientation

    @pose.setter
    def pose(self, pose):
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


class Collision(object):
    r"""Collision parameters for body."""

    def __init__(self, name=None, dtype=None, size=None, filename=None, position=None, orientation=None):
        self.name = name
        self.geometry = Geometry(dtype=dtype, size=size, filename=filename)
        self.position = position
        self.orientation = orientation

    @property
    def dtype(self):
        return self.geometry.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.geometry.dtype = dtype

    @property
    def size(self):
        return self.geometry.size

    @size.setter
    def size(self, size):
        self.geometry.size = size

    @property
    def filename(self):
        return self.geometry.filename

    @filename.setter
    def filename(self, filename):
        self.geometry.filename = filename

    @property
    def format(self):
        if self.filename is not None:
            return self.filename.split('.')[-1]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if position is not None:
            if isinstance(position, str):
                position = [float(p) for p in position.split()]
            position = np.asarray(position)
        self._position = position

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if orientation is not None:
            if isinstance(orientation, str):
                orientation = [float(o) for o in orientation.split()]
            if len(orientation) == 4:  # quaternion
                orientation = get_rpy_from_quaternion(orientation)
            if len(orientation) == 3:  # rpy
                pass
            orientation = np.asarray(orientation)
        self._orientation = orientation

    @property
    def rpy(self):
        return self._orientation

    @property
    def quaternion(self):
        return get_quaternion_from_rpy(self._orientation)

    @property
    def rot(self):
        return get_matrix_from_rpy(self.rpy)

    @property
    def pose(self):
        return self.position, self.orientation

    @pose.setter
    def pose(self, pose):
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


class Material(object):
    r"""Material info."""

    def __init__(self, name=None, color=None, texture=None):
        self.name = name
        self.color = color
        self.texture = texture

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        if color is not None:
            if isinstance(color, str):  # e.g. '0.5 0.1 1. 1.'
                color = (float(c) for c in color.split())
            if not isinstance(color, (list, tuple)):
                raise TypeError("Expecting the color to be a tuple or list of 3 or 4 float")
            else:
                color = tuple(color)
        self._color = color

    @property
    def rgb(self):
        if self.color is None:
            return 0.5, 0.5, 0.5
        return tuple(self.color[:3])

    @property
    def rgba(self):
        if self.color is None:
            return 0.5, 0.5, 0.5, 1.
        if len(self.color) == 3:
            return tuple(self.color) + (1.,)
        return tuple(self.color)


class Sensor(object):
    pass


class Heightmap(object):
    pass
