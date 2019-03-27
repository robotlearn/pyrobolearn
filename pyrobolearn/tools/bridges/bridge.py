#!/usr/bin/env python
"""Define the Bridge class.

The Bridge class links an interface with something else. This can be the world (such as a robot, a camera,
a controller), a state / action, an environment, or a task. This is the main parent class from which any other
bridges inherit from.
"""

from pyrobolearn.tools.interfaces import Interface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Bridge(object):
    r"""Bridge class

    The Bridge class links an interface with something else in the world, such as a robot, a camera, a controller, etc.
    This is the main parent class from which any other bridges inherit from.

    More specifically, the interface can be run in a separate thread or in the same thread than the main one.
    The interface collects the input data/events and/or outputs the given data/events, but the interface does not
    know how to connect to the rest of the code. The bridge is the one that possesses this knowledge
    (see Bridge/Adapter design pattern). There can be multiple interfaces running (like for the audio and webcam),
    and each interface can have multiple bridges. However, each bridge links only one specific interface to something
    else.

    For instance, let's say we have an Xbox controller interface, and we want to control a quadcopter robot and
    a wheeled robot. Everytime pushing the joystick forward results the interface to remember such event, but this
    does not result in anything in the simulator. The associated bridge such as `BridgeXboxUAV` and
    `BridgeUAVWheeled` knows what should be done if the joystick is moved forward. In the case of the quadcopter,
    the bridge could for instance increase the speed of the propellers, and for the wheeled robot could result
    to move this one forward. There could be other bridges linking the Xbox controller to other things in the world,
    like the world camera, a visual object, etc. An interface can have multiple bridges, e.g. a bridge linking the
    same Xbox controller to different cameras in the world. Pushing the joystick will result in the same movements
    for all these cameras.
    """

    def __init__(self, interface, priority=None):
        """
        Initialize the bridge.

        Args:
            interface (Interface): interface instance.
            priority (None, int): priority number.
        """
        self.interface = interface
        self.priority = priority

    ##############
    # Properties #
    ##############

    @property
    def interface(self):
        """Return the interface instance associated to the bridge."""
        return self._interface

    @interface.setter
    def interface(self, interface):
        """Set the interface associated to the bridge."""
        if not isinstance(interface, Interface):
            raise TypeError("Expecting interface to be an instance of Interface, instead got {}".format(interface))
        self._interface = interface

    @property
    def priority(self):
        """Return the priority number."""
        return self._priority

    @priority.setter
    def priority(self, priority):
        """Set the priority number."""
        if priority is not None:
            if not isinstance(priority, int):
                raise TypeError("Expecting the priority to be an integer, instead got {}".format(priority))
        self._priority = priority

    ###########
    # Methods #
    ###########

    def step(self, update_interface=False):
        """Main function that has to be overwritten by the user."""
        raise NotImplementedError

    def __call__(self, update_interface=False):
        self.step(update_interface=update_interface)
