#!/usr/bin/env python
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

from abc import ABCMeta, abstractmethod

from pyrobolearn.states.state import State
from pyrobolearn.worlds.world import World
from pyrobolearn.robots import Object


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ObjectState(State):
    """Object state (abstract)
    """
    __metaclass__ = ABCMeta

    def __init__(self, obj, world=None):
        super(ObjectState, self).__init__()
        if not isinstance(obj, (Object, int)):
            raise TypeError("Expecting an instance of Object, or an identifier from the simulator/world.")
        if isinstance(obj, int):
            if not isinstance(world, World):
                # try to look for the world in global variables
                if 'world' in globals() and isinstance(globals()['world'], World): # O(1)
                    world = globals()['world']
                else:
                    raise ValueError("When giving the object identifier, the world need to be given as well.")
            obj = Object(world.getSimulator(), obj)
        self.obj = obj

    @abstractmethod
    def _read(self):
        pass


class PositionState(ObjectState):
    """Position of an object.
    """
    def __init__(self, obj, world=None):
        super(PositionState, self).__init__(obj, world)
        self.data = self.obj.position

    def _read(self):
        self.data = self.obj.position


class OrientationState(ObjectState):
    """Orientation of an object.
    """
    def __init__(self, obj, world=None):
        super(OrientationState, self).__init__(obj, world)
        self.data = self.obj.orientation

    def _read(self):
        self.data = self.obj.orientation


class VelocityState(ObjectState):
    """Velocity of an object.
    """
    def __init__(self, obj, world=None):
        super(VelocityState, self).__init__(obj, world)
        self.data = self.obj.velocity

    def _read(self):
        self.data = self.obj.velocity
