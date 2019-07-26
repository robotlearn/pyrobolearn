#!/usr/bin/env python
"""Provide the data structures that are shared among the various parsers and converter.
"""

import numpy as np
from collections import OrderedDict

from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_quaternion_from_rpy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class World(object):
    r"""World data structure."""

    def __init__(self, trees=None):
        self.trees = trees


class Tree(object):
    r"""Tree data structure."""

    def __init__(self, name=None, root=None):
        self.name = name
        self.root = root
        self.bodies = OrderedDict()
        self.joints = OrderedDict()
        self.materials = {}


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

    Joint types: fixed, revolute/hinge, continuous
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
    def diagonal_inertia(self):
        return np.array([self.ixx, self.iyy, self.izz])

    @property
    def full_inertia(self):
        return np.array([[self.ixx, self.ixy, self.ixz],
                         [self.ixy, self.iyy, self.iyz],
                         [self.ixz, self.iyz, self.izz]])

    @property
    def ixx(self):
        return self._ixx

    @ixx.setter
    def ixx(self, ixx):
        if ixx is not None:
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

    def __init__(self, mass=None, inertia=None, position=None, orientation=None):
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
    def aligned_inertia(self):
        raise NotImplementedError

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


class Visual(object):
    r"""visual parameters for body."""

    def __init__(self, name=None, dtype=None, size=None, color=None, filename=None, position=None, orientation=None,
                 material=None):
        self.name = name
        self.dtype = dtype
        self.size = size     # depending on the type it can be different size
        self.color = color
        self.filename = filename
        self.position = position
        self.orientation = orientation
        self.material = material

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        if size is not None:
            if isinstance(size, str):
                size = [float(s) for s in size.split()]
            elif isinstance(size, (tuple, list, np.ndarray)):
                size = [float(s) for s in size]
            elif not isinstance(size, (float, int)):
                raise TypeError("Expecting the size to be a float, int, list, tuple or np.ndarray")
        self._size = size

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


class Collision(object):
    r"""Collision parameters for body."""

    def __init__(self, name=None, dtype=None, size=None, filename=None, position=None, orientation=None):
        self.name = name
        self.dtype = dtype
        self.size = size
        self.filename = filename
        self.position = position
        self.orientation = orientation

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        if size is not None:
            if isinstance(size, str):
                size = [float(s) for s in size.split()]
            elif isinstance(size, (tuple, list, np.ndarray)):
                size = [float(s) for s in size]
            elif not isinstance(size, (float, int)):
                raise TypeError("Expecting the size to be a float, int, list, tuple or np.ndarray")
        self._size = size

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


class Material(object):
    r"""Material info."""

    def __init__(self, name=None, color=None):
        self.name = name
        self.color = color

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
