#!/usr/bin/env python
"""Define miscellaneous sensors.
"""

from links import Sensor, LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class HeightmapSensor(Sensor):
    r"""Heightmap Sensor

    Sensor that detects the heights of its surrounding using a grid map.
    """

    def __init__(self, simulator, body_id, link_id, width, height, num_rays_width, num_rays_height,
                 position=None, orientation=None):
        """Initialize the heightmap sensor.

        Note that `num_rays_width * num_rays_height` has to be smaller than `simulator.MAX_RAY_INTERSECTION_BATCH_SIZE`.
        In pybullet, this is currently set to be 256.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique id of the body
            link_id (int): unique id of the link
            position: local position of the sensor with respect to the given link
            orientation: local orientation of the sensor with respect to the given link
            width (float): width of the map (along the left-right axis of the body, measured in meters)
            height (float): height of the map (along the front-back axis of the body, measured in meters)
            num_rays_width (int): number of rays along the width dimension (left-right axis). This will be the 'width'
                of the returned heightmap.
            num_rays_height (int): number of rays along the height dimension (front-back axis). This will be
                the 'height' of the returned heightmap.
        """
        super(HeightmapSensor, self).__init__(simulator, body_id, link_id, position, orientation)

        # Check arguments
        if not isinstance(num_rays_width, int):
            raise TypeError("num_rays_width needs to be an integer")
        if not isinstance(num_rays_height, int):
            raise TypeError("num_rays_height needs to be an integer")
        if num_rays_width * num_rays_height > self.sim.MAX_RAY_INTERSECTION_BATCH_SIZE:  # pybullet = 256
            raise ValueError("num_rays_width * num_rays_height can not be bigger"
                             " than {}".format(self.sim.MAX_RAY_INTERSECTION_BATCH_SIZE))
        if num_rays_width < 2:
            raise ValueError("num_rays_width must be equal or bigger than 2")
        if num_rays_height < 2:
            raise ValueError("num_rays_height must be equal or bigger than 2")

        # construct the grid
        self.width_step = float(width) / num_rays_width
        self.height_step = float(height) / num_rays_height

    def get_ray_from_positions(self):
        # using width and height
        pass

    def get_ray_to_positions(self):
        pass

    def _sense(self, normalize=False, display=False):
        """Return the heightmap.

        Returns:
            np.array: Height map with shape [width, height] where the values are the heights (in meters).
        """
        # calculate width and height
        collisions = self.sim.rayTestBatch(self.get_ray_from_positions(), self.get_ray_to_positions())
        return collisions


class TemperatureSensor(LinkSensor):
    pass


class UltrasonicSensor(LinkSensor):
    pass


class HumiditySensor(LinkSensor):
    pass
