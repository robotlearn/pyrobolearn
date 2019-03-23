#!/usr/bin/env python
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

from abc import ABCMeta, abstractmethod

from pyrobolearn.states.state import State
from pyrobolearn.worlds import World
from pyrobolearn.robots import Body


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BodyState(State):
    """Body state (abstract)
    """
    __metaclass__ = ABCMeta

    def __init__(self, body, world=None):
        """
        Initialize the body state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
        """

        super(BodyState, self).__init__()
        if not isinstance(body, (Body, int)):
            raise TypeError("Expecting an instance of Body, or an identifier from the simulator/world.")
        if isinstance(body, int):
            if not isinstance(world, World):
                # try to look for the world in global variables
                if 'world' in globals() and isinstance(globals()['world'], World):  # O(1)
                    world = globals()['world']
                else:
                    raise ValueError("When giving the body identifier, the world need to be given as well.")
            body = Body(world.simulator, body)
        self.body = body

    @abstractmethod
    def _read(self):
        pass


class PositionState(BodyState):
    """Position of a body in the world.
    """

    def __init__(self, body, world=None):
        """
        Initialize the position state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
        """
        super(PositionState, self).__init__(body, world)
        self.data = self.body.position

    def _read(self):
        self.data = self.body.position


class OrientationState(BodyState):
    """Orientation of a body in the world.
    """

    def __init__(self, body, world=None):
        """
        Initialize the orientation state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
        """
        super(OrientationState, self).__init__(body, world)
        self.data = self.body.orientation

    def _read(self):
        self.data = self.body.orientation


class VelocityState(BodyState):
    """Velocity of a body in the world
    """

    def __init__(self, body, world=None):
        """
        Initialize the velocity state.

        Args:
            body (Body, int): body or unique body id.
            world (None, World): world instance if the body id was given.
        """
        super(VelocityState, self).__init__(body, world)
        self.data = self.body.velocity

    def _read(self):
        self.data = self.body.velocity
