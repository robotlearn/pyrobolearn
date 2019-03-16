#!/usr/bin/env python
"""Define contact sensors attached to a link.
"""

from links import LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ContactSensor(LinkSensor):
    r"""Contact sensor.

    This sensor return 1 if in contact with an object, and 0 otherwise.
    """

    def __init__(self, simulator, body_id, link_id, position, orientation, refresh_rate=1):
        super(ContactSensor, self).__init__(simulator, body_id, link_id, position, orientation, refresh_rate)

    def get_contact_points(self):
        """Get the contact points.

        Returns:
            list: list of contacts
        """
        contacts = self.sim.getContactPoints(bodyA=self.body_id, linkIndexA=self.link_id)
        return contacts

    def is_in_contact(self):
        """
        Returns true if in contact.
        """
        return len(self.get_contact_points()) > 0

    # alias
    _sense = is_in_contact


class PressureSensor(LinkSensor):
    r"""Pressure sensor.

    Compared to the binary contact sensor, this sensor returns a continuous pressure value.
    """

    def __init__(self, simulator, body_id, link_id):
        super(PressureSensor, self).__init__(simulator, body_id, link_id)

    def _sense(self):
        raise NotImplementedError


class TourchSensor(LinkSensor):
    r"""Touch Sensor
    """


# class SkinPressureSensor
