#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define contact sensors attached to a link.
"""

from pyrobolearn.robots.sensors.links import LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ContactSensor(LinkSensor):
    r"""Contact sensor.

    This sensor return 1 if in contact with an object, and 0 otherwise.
    """

    def __init__(self, simulator, body_id, link_id=-1, noise=None, ticks=1, latency=None):
        """
        Initialize the contact sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            link_id (int): unique id of the link.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
        """
        super(ContactSensor, self).__init__(simulator, body_id=body_id, link_id=link_id, noise=noise, ticks=ticks,
                                            latency=latency)

    def get_contact_points(self):
        """
        Get the contact points.

        Returns:
            list: list of contacts
        """
        contacts = self.sim.get_contact_points(bodyA=self.body_id, link1_id=self.link_id)
        return contacts

    def is_in_contact(self):
        """
        Returns true if in contact.
        """
        return len(self.get_contact_points()) > 0

    def _sense(self, apply_noise=True):
        """
        Sense using the contact sensor.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            bool: True if there is contact.
        """
        # there is already some noise from the simulator
        return self.is_in_contact()


# class PressureSensor(LinkSensor):
#     r"""Pressure sensor.
#
#     Compared to the binary contact sensor, this sensor returns a continuous pressure value.
#     """
#
#     def __init__(self, simulator, body_id, link_id):
#         super(PressureSensor, self).__init__(simulator, body_id, link_id)
#
#     def _sense(self, apply_noise=True):
#         raise NotImplementedError
#
#
# class TouchSensor(LinkSensor):
#     r"""Touch Sensor
#     """
#     pass
#
#
# class SkinPressureSensor
