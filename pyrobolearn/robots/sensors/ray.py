#!/usr/bin/env python
"""Define ray sensors; sensors that cast rays into the world and return the range of the nearest object that were
intersected with these ones.
"""

import numpy as np

from pyrobolearn.robots.sensors.links import LinkSensor
from pyrobolearn.utils.transformation import get_rotated_point_from_quaternion

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RaySensor(LinkSensor):
    r"""Ray sensor

    This sensor casts a single ray into the world, check for intersection, and return the range of the nearest object.
    """

    def __init__(self, simulator, body_id, to_position, link_id=-1, noise=None, ticks=1, latency=None, position=None,
                 orientation=None):
        """
        Initialize the Ray sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            to_position (np.array[float[3]]): position where the ray should stop with respect to the new local link
                frame (specified by :attr:`position` and :attr:`orientation`).
            link_id (int): unique id of the link.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
            position (np.array[float[3]], None): local position of the sensor with respect to the given link. If None,
                it will be the zero vector.
            orientation (np.array[float[4]], None): local orientation of the sensor with respect to the given link
                (expressed as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
        """
        super(RaySensor, self).__init__(simulator, body_id=body_id, link_id=link_id, noise=noise, ticks=ticks,
                                        latency=latency, position=position, orientation=orientation)
        if isinstance(to_position, (tuple, list)):
            to_position = np.asarray(to_position)
        if not isinstance(to_position, np.ndarray):
            raise TypeError("Expecting the given 'to_position' to be a np.array, but got instead: "
                            "{}".format(type(to_position)))
        if to_position.shape != (3,):
            raise ValueError("Expecting the shape of the given 'to_position' to be (3,), but got instead a shape of: "
                             "{}".format(to_position.shape))
        self.to_position = to_position

    def _sense(self, apply_noise=True):
        """
        Sense using the ray sensor.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            float: hit fraction along the ray in range [0,1] along the ray.
        """
        if self.simulator.supports_sensors("ray"):
            return self.simulator.get_sensor("ray", self.body_id, self.link_id).sense()
        position = self.position + get_rotated_point_from_quaternion(self.orientation, self.to_position)
        hit = self.sim.ray_test(from_position=self.position, to_position=position)[2]
        if apply_noise:
            hit = self._noise(hit)
        return hit

    def render(self, enable=True, color=None):
        """Render the ray in the simulator; they are only visual and attached to the sensor (link). The visual shape
        is updated at runtime (each time you call this function).

        Args:
            enable (bool): if we should render or not.
            color (None, tuple/list of 4 float, np.ndarray[float[4]]): RGBA color of all the rays, where each channel
                is between 0 and 1.
        """
        pass


class RayBatchSensor(LinkSensor):
    r"""Ray batch sensor.

    This sensor casts a batch of rays into the world, check for intersections, and return the range of the nearest
    objects. This can be used for sonars, laser scanning range sensors (such as LIDAR), and others.

    Note that the number of rays must be smaller than `simulator.MAX_RAY_INTERSECTION_BATCH_SIZE`. In pybullet, this
    is currently set to 16,384.
    """

    def __init__(self, simulator, body_id, to_positions, link_id=-1, noise=None, ticks=1, latency=None, position=None,
                 orientation=None):
        """
        Initialize the Ray batch sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            to_positions (np.array[N,3]): position where each ray should stop with respect to the new local link frame
                (specified by :attr:`position` and :attr:`orientation`).
            link_id (int): unique id of the link.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
            position (np.array[float[3]], None): local position of the sensor with respect to the given link. If None,
                it will be the zero vector.
            orientation (np.array[float[4]], None): local orientation of the sensor with respect to the given link
                (expressed as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
        """
        super(RayBatchSensor, self).__init__(simulator, body_id=body_id, link_id=link_id, noise=noise, ticks=ticks,
                                             latency=latency, position=position, orientation=orientation)
        if isinstance(to_positions, (tuple, list)):
            to_positions = np.asarray(to_positions)
        if not isinstance(to_positions, np.ndarray):
            raise TypeError("Expecting the given 'to_positions' to be a np.array, but got instead: "
                            "{}".format(type(to_positions)))
        if to_positions.ndim != 2:
            raise ValueError("Expecting the given 'to_positions' to be 2D array, but got instead a {}D "
                             "array".format(to_positions.ndim))
        if to_positions.shape[1] != 3:
            raise ValueError("Expecting the shape of the given 'to_positions' to be (N,3), but got instead a shape "
                             "of: {}".format(to_positions.shape))
        if len(to_positions) > self.sim.MAX_RAY_INTERSECTION_BATCH_SIZE:
            raise ValueError("The number of 'to_positions' (={}) is bigger than the maximum amount authorized "
                             "(={})".format(len(to_positions), self.sim.MAX_RAY_INTERSECTION_BATCH_SIZE))
        self.to_positions = to_positions

    def _sense(self, apply_noise=True):
        """
        Sense using the ray batch sensor.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            np.array[N]: hit fractions along each ray in range [0,1] along the ray.
        """
        if self.simulator.supports_sensors("ray_batch"):
            return self.simulator.get_sensor("ray_batch", self.body_id, self.link_id).sense()
        position = self.position + get_rotated_point_from_quaternion(self.orientation, self.to_positions)
        rays = self.sim.ray_test_batch(from_positions=self.position, to_positions=position)
        hit = np.array([ray[2] for ray in rays])
        if apply_noise:
            hit = self._noise(hit)
        return hit

    def render(self, enable=True, color=None):
        """Render the batch of rays in the simulator; they are only visual and attached to the sensor (link). The
        visual shape of each ray is updated at runtime (each time you call this function).

        Args:
            enable (bool): if we should render or not.
            color (None, tuple/list of 4 float, np.ndarray[float[4]]): RGBA color of all the rays, where each channel
                is between 0 and 1.
        """
        pass


class HeightmapSensor(LinkSensor):
    r"""Heightmap Sensor

    Sensor that detects the heights of its surrounding using a grid map.

    Warnings: this is only valid in the simulator.
    """

    def __init__(self, simulator, body_id, link_id, width, height, num_rays_width=2, num_rays_height=2,
                 max_ray_length=100, position=None, orientation=None):  # TODO: use orientation initially
        """
        Initialize the heightmap sensor. This is only valid in the simulator.

        Note that `num_rays_width * num_rays_height` has to be smaller than `simulator.MAX_RAY_INTERSECTION_BATCH_SIZE`.
        In pybullet, this is currently set to 16,384.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique id of the body
            link_id (int): unique id of the link
            width (float): width of the map (along the left-right axis (i.e. y axis) of the body, measured in meters)
            height (float): height of the map (along the front-back axis (i.e. x axis) of the body, measured in meters)
            num_rays_width (int): number of rays along the width dimension (left-right axis). This will be the 'width'
                of the returned heightmap. This must be bigger or equal to 2.
            num_rays_height (int): number of rays along the height dimension (front-back axis). This will be
                the 'height' of the returned heightmap. This must be bigger or equal to 2.
            max_ray_length (float): maximum length of each ray.
            position (np.array[float[3]], None): local position of the sensor with respect to the given link. If None,
                it will be the zero vector. This position represents the center of the map.
            orientation (np.array[float[4]], None): local orientation of the sensor with respect to the given link
                (expressed as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
        """
        super(HeightmapSensor, self).__init__(simulator, body_id=body_id, link_id=link_id, position=position,
                                              orientation=orientation)

        # Check arguments

        # set num_rays_width and num_rays_height
        num_rays_width, num_rays_height = int(num_rays_width), int(num_rays_height)
        if num_rays_width * num_rays_height > self.sim.MAX_RAY_INTERSECTION_BATCH_SIZE:  # pybullet = 16,384
            raise ValueError("num_rays_width * num_rays_height can not be bigger"
                             " than {}".format(self.sim.MAX_RAY_INTERSECTION_BATCH_SIZE))
        if num_rays_width < 2:
            raise ValueError("num_rays_width must be equal or bigger than 2, but got: {}".format(num_rays_width))
        if num_rays_height < 2:
            raise ValueError("num_rays_height must be equal or bigger than 2, but got: {}".format(num_rays_height))
        self._num_rays_width = num_rays_width
        self._num_rays_height = num_rays_height

        # set max_ray_length
        if not isinstance(max_ray_length, (float, int)):
            raise TypeError("Expecting 'max_ray_length' to be an int or float, but got instead: "
                            "{}".format(type(max_ray_length)))
        max_ray_length = float(max_ray_length)
        if max_ray_length <= 0.:
            raise ValueError("Expecting 'max_ray_length' to be positive, but got instead: {}".format(max_ray_length))
        self._max_ray_length = max_ray_length

        # set width and height
        width, height = float(width), float(height)
        if width <= 0.:
            raise ValueError("Expecting the 'width' to be bigger than 0, but got: {}".format(width))
        if height <= 0.:
            raise ValueError("Expecting the 'height' to be bigger than 0, but got: {}".format(height))
        self._width = width
        self._height = height

        self._z_array = np.ones(self._num_rays_width * self._num_rays_height)

    def get_ray_from_to_positions(self):
        """
        Return the world positions for the rays to start and end.

        Returns:
            np.array[N,3]: list of starting positions for the rays
            np.array[N,3]: list of ending positions for the rays
        """
        pos = self.position
        w2, h2 = self._width / 2., self._height / 2.

        x, y = np.meshgrid(np.linspace(pos[1] - w2, pos[1] + w2, self._num_rays_width),
                           np.linspace(pos[0] - h2, pos[0] + h2, self._num_rays_height))
        x, y = x.ravel(), y.ravel()
        from_z = pos[2] * self._z_array
        to_z = from_z - self._max_ray_length

        from_positions = np.vstack((x, y, from_z)).T  # (N, 3)
        to_positions = np.vstack((x, y, to_z)).T  # (N, 3)

        return from_positions, to_positions

    def _sense(self, apply_noise=True):
        """
        Return the heightmap.

        Returns:
            np.array[width, height]: Height map with shape [width, height] where the values are the hit fractions [0,1],
                you can multiply it by :attr:`max_ray_length` to get the depth in meters.
        """
        from_positions, to_positions = self.get_ray_from_to_positions()
        rays = self.sim.ray_test_batch(from_positions=from_positions, to_positions=to_positions)
        hit = np.array([ray[2] for ray in rays]).reshape(self._num_rays_width, self._num_rays_height)
        if apply_noise:
            hit = self._noise(hit)
        return hit

    def render(self, enable=True, color=None):
        """Render the grid map in the simulator; each point/intersection in the grid is represented as a visual
        sphere. The position of these spheres is updated at runtime (each time you call this function).

        Args:
            enable (bool): if we should render or not.
            color (None, tuple/list of 4 float, np.ndarray[float[4]]): RGBA color of all the rays, where each channel
                is between 0 and 1.
        """
        pass
