#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define miscellaneous sensors.
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


class TemperatureSensor(LinkSensor):
    r"""Temperature sensor

    A temperature sensor measures the temperature and converts it into an electrical signal.
    """
    raise NotImplementedError


class UltrasonicSensor(LinkSensor):
    r"""Ultrasonic Sensor

    "Ultrasonic transducers or ultrasonic sensors are a type of acoustic sensor divided into three broad categories:
    transmitters, receivers and transceivers. Transmitters convert electrical signals into ultrasound, receivers
    convert ultrasound into electrical signals, and transceivers can both transmit and receive ultrasound." [1]

    References:
        - [1] Ultrasonic transducer (Wikipedia): https://en.wikipedia.org/wiki/Ultrasonic_transducer
    """
    raise NotImplementedError


class TransmitterSensor(UltrasonicSensor):
    r"""Transmitter sensor

    "The transmitter sensor is an acoustic sensor which converts the electrical signals into ultrasounds." [1]

    References:
        - [1] Ultrasonic transducer (Wikipedia): https://en.wikipedia.org/wiki/Ultrasonic_transducer
    """
    raise NotImplementedError


class ReceiverSensor(UltrasonicSensor):
    r"""Receiver sensor

    "The receiver sensor is an acoustic sensor which converts ultrasound into electrical signals." [1]

    References:
        - [1] Ultrasonic transducer (Wikipedia): https://en.wikipedia.org/wiki/Ultrasonic_transducer
    """
    raise NotImplementedError


class TransceiverSensor(UltrasonicSensor):
    r"""Receiver sensor

    "The transceiver sensor can both transmit and receive ultrasound by converting into/from electrical signals." [1]

    References:
        - [1] Ultrasonic transducer (Wikipedia): https://en.wikipedia.org/wiki/Ultrasonic_transducer
    """
    raise NotImplementedError


class HumiditySensor(LinkSensor):
    r"""Humidity sensor

    A humidity sensor measures and reports the moisture and air temperature.
    """
    raise NotImplementedError


class GPSSensor(LinkSensor):
    r"""GPS sensor

    "The GPS is a satellite-based radionavigation system. Autonomous robots use a GPS sensors to get the latitude,
    longitude, time, speed, and heading." [1]

    References:
        - [1] Global Positioning System (Wikipedia): https://en.wikipedia.org/wiki/Global_Positioning_System
    """
    raise NotImplementedError


class MagnetometerSensor(LinkSensor):
    r"""Magnetometer Sensor

    "A magnetometer is a device that measures magnetism - the direction, strength, or relative change of a magnetic
    field at a particular location." [1]

    References:
        - [1] Magnetometer (Wikipedia): https://en.wikipedia.org/wiki/Magnetometer
    """
    raise NotImplementedError
