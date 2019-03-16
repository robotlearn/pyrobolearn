#!/usr/bin/env python
"""Define the various basic objects that are present in the simulator/world.
"""

import numpy as np
import quaternion

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Object(object):
    r"""Object

    Define an object in the simulator/world.
    """

    def __init__(self, simulator, object_id):
        self.sim = simulator
        self.id = object_id

    @property
    def name(self):
        return self.sim.getBodyInfo(self.id)

    @property
    def position(self):
        return np.array(self.sim.getBasePositionAndOrientation(self.id)[0])

    @property
    def quaternion(self):
        quat = self.sim.getBasePositionAndOrientation(self.id)[1]
        return quaternion.quaternion(quat[3], *quat[:3])

    # alias
    orientation = quaternion

    @property
    def rpy(self):
        quat = self.sim.getBasePositionAndOrientation(self.id)[1]
        y, p, r = self.sim.getEulerFromQuaternion(quat)
        return np.array([r, p, y])

    @property
    def rotation(self):
        quat = self.sim.getBasePositionAndOrientation(self.id)[1]
        rot = self.sim.getMatrixFromQuaternion(quat)
        return np.array(rot).reshape(3, 3)

    @property
    def state(self):
        pos, quat = self.sim.getBasePositionAndOrientation(self.id)
        rpy = self.sim.getEulerFromQuaternion(quat)[::-1]
        # return np.array(pos), quaternion.quaternion(quat[3], *quat[:3])
        return np.array(pos+rpy)

    @property
    def linear_velocity(self):
        return np.array(self.sim.getBaseVelocity(self.id)[0])

    @property
    def angular_velocity(self):
        return np.array(self.sim.getBaseVelocity(self.id)[1])

    @property
    def velocity(self):
        lin, ang = self.sim.getBaseVelocity(self.id)
        return np.array(lin+ang)

    @property
    def color(self):
        return self.sim.getVisualShapeData(self.id)[0][-1]

    # alias
    rgba_color = color

    @property
    def mass(self):
        links = [-1] + list(range(self.sim.getNumJoints(self.id)))
        return np.sum([self.sim.getDynamicsInfo(self.id, linkId)[0] for linkId in links])

    @property
    def dimensions(self):
        return np.array(self.sim.getVisualShapeData(self.id)[0][3])


class MovableObject(Object):
    r"""Movable Object

    Define a movable object in the world.
    """

    def __init__(self, simulator, object_id):
        super(MovableObject, self).__init__(simulator, object_id)

    def move(self, new_position=None, new_orientation=None):
        pass


class ControllableObject(MovableObject):
    r"""Controllable Object

    Define a controllable object in the world.
    """

    def __init__(self, simulator, object_id):
        super(ControllableObject, self).__init__(simulator, object_id)
