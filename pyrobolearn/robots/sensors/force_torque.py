#!/usr/bin/env python
"""Define the force torque joint sensors used in robotics.
"""

import numpy as np

from pyrobolearn.robots.sensors.joints import JointSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ForceTorqueSensor(JointSensor):
    r"""Force-Torque (FT or F/T) Sensor

    The F/T sensor allows to measure the forces and torques applied to it.
    """

    def __init__(self, simulator, body_id, joint_id, position, orientation, rate=1):
        super(ForceTorqueSensor, self).__init__(simulator, body_id, joint_id, position, orientation, rate)
        self.sim.enable_joint_force_torque_sensor(body_id, joint_id, enableSensor=True)

    def _sense(self):
        return np.array(self.sim.getJointState(self.body_id, self.joint_id)[2])
