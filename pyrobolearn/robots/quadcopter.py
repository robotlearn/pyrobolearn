#!/usr/bin/env python
"""Provide the Quadcopter robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.uav import RotaryWingUAV
from pyrobolearn.utils.transformation import get_matrix_from_quaternion
from pyrobolearn.utils.units import inches_to_meters, rpm_to_rad_per_second

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: several methods such as the calculation of the thrust force should be done in `uav.py` file, not here
class Quadcopter(RotaryWingUAV):
    r"""Quadcopter

    WARNING: Currently, in pybullet there is no air, so we simulate the thrust force.

    Based on momentum theory, we can calculate the thrust [5,6,7] to be:

    .. math::

        F &= A \Delta p = A (P_e - P_0) \\
        F &= A [(p_0 + \frac{1}{2} \rho v_e^2) - (p_0 + \frac{1}{2} \rho v_0^2)] \\
        F &= \frac{\rho A}{2} (v_e^2 - v_0^2)

    where :math:`\Delta p` is the difference between the pressure ahead and behind the propeller, :math:`A = \pi r^2`
    is the propeller disk area with radius :math:`r = \frac{D}{2}` and diameter :math:`D`, :math:`p_0` is the static
    pressure and :math:`\frac{1}{2} \rho v^2` is the dynamic pressure, :math:`v_0` is the velocity of the aircraft
    (aka propeller forward speed, inflow velocity, forward airspeed, or freestream velocity), and :math:`v_e` is the
    exit velocity. At the second line above, we used Bernoulli's equation to relate the pressure and velocity.

    According to [6], the dynamic thrust is given by:

    .. math:: F = \rho A [ (\frac{RPM}{60} pitch)^2 - (\frac{RPM}{60} pitch) v_0 ] (k_1 \frac{D}{pitch})^k_2

    where :math:`pitch` is the propeller pitch (i.e. the distance a propeller would move in 1 revolution if it were
    moving through a soft solid). If :math:`v_0 = 0` we have a static thrust.

    Notes:
    * From [6], "for dynamic thrust, consider the equation to be an underestimate of what the propeller is actually
        doing"
    * :math:`1 RPM = \frac{2\pi}{60} rad/s`
    * the air density :math:`\rho` is :math:`1.225kg/m^3` at sea level and at :math:`15C`.

    References:
        [1] https://www.wilselby.com/research/ros-integration/
        [2] https://github.com/wilselby/ROS_quadrotor_simulator
        [3] https://github.com/prfraanje/quadcopter_sim
        [4] https://github.com/ethz-asl/rotors_simulator

        [5] "Propeller Thrust" (NASA): https://www.grc.nasa.gov/WWW/K-12/airplane/propth.html
        [6] "Static thrust calculation": https://quadcopterproject.wordpress.com/static-thrust-calculation/
        [7] "Propeller Static & Dynamic Thrust Calculation":
            https://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
            https://www.electricrcaircraftguy.com/2014/04/propeller-static-dynamic-thrust-equation-background.html
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.2),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/quadcopter/quadcopter.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.2)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.2,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Quadcopter, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'quadcopter'

        self.gravity = 9.81
        self.air_density = 1.225

        # from urdf
        self.radius = 0.14
        self.diameter = 2. * self.radius
        self.area = np.pi * self.radius**2

        self.max_velocity = 770  # rad/sec

        # joints 1 and 3 are CCW, and joints 2 and 4 are CW
        # CCW = +1, CW = -1
        self.turning_directions = [+1, -1, +1, -1]

        # Propeller pitches are around 0.0762m (3 inches) and 0.127m (5 inches)
        # (from https://www.dronezon.com/learn-about-drones-quadcopters/how-a-quadcopter-works-with-propellers-and\
        # -motors-direction-design-explained/)
        # The larger the prop (either increasing diameter, or pitch or both), the more energy it takes to spin it.
        # When buying propellers, they follow the syntax `LxP` where `L` is the length/diameter of the propeller
        # (in inches) and `P` is the propeller pitch (also in inches).
        # 1 inch = 0.0254m --> 0.28m = 11 inches
        # The value of the pitch below has been set by looking online for quadcopter props with 11 inches of length
        self.propeller_pitch = inches_to_meters(4.7)  # inches_to_meters(5.)

        # some constants
        self.k1 = 1./3.29546
        self.k2 = 1.5

    def calculate_thrust_force(self, angular_speed, area, propeller_pitch, v0=0, air_density=1.225):
        r"""
        Calculate the thrust force generated by the propeller (based on [6]).

        Args:
            angular_speed (float): angular speed of the propeller [rad/s]. If RPM, convert it to rad/s using the
                formula :math:`1RPM = \frac{2\pi}{60} rad/s`.
            area (float): area of the propeller [m^2]
            propeller_pitch (float): "distance a propeller would move in 1 revolution if it were moving through a
                soft solid" [m]
            v0 (float): initial velocity
            air_density (float): density of air [kg/m^3]. By default, it is the density of the air at sea level and at
                15 degrees Celsius. Note that this varies with the temperature, humidity, and pressure. It decreases
                with increasing altitude.

        Returns:
            float: thrust force generated by the propeller [N]
        """
        tmp = angular_speed / (2*np.pi) * propeller_pitch
        diameter = (4. * area / np.pi)**0.5
        return air_density * area * (tmp**2 - tmp*v0) * (self.k1 * diameter / propeller_pitch)**self.k2

    def set_propeller_velocities(self, velocities, max_velocity=True, forces=True):
        """
        Set the joint velocities and apply the thrust force on the propeller link corresponding to the given
        joint id(s).

        Args:
            velocities (np.array[4]): velocity of each propeller
            forces (float, np.float[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
            max_velocity (float, bool, None): if True, it will make sure that the given velocity(ies) are below their
                authorized maximum value(s) (inferred from the URDF, or set previously by the user). If you already
                did the check outside the method or if you don't want limits, set this variable to False.

        Returns:
            None
        """
        if len(velocities) != 4:
            raise ValueError("Expecting a velocity for each propeller")

        joint_ids = self.joints

        # call parent method
        super(Quadcopter, self).set_joint_velocities(velocities, joint_ids, forces, max_velocity)

        # calculate thrust force of the given joints, and apply it on the link
        for jnt, d, v in zip(joint_ids, self.turning_directions, velocities):
            if max_velocity and v > self.max_velocity:
                v = self.max_velocity

            # compute propeller speed v0
            state = self.sim.get_link_state(self.id, jnt, compute_velocity=True)  # , compute_forward_kinematics=True)
            R = get_matrix_from_quaternion(state[1])
            linear_velocity = np.array(state[-2])
            propeller_up_vec = R.dot(np.array([0., 0., 1.]))
            v0 = linear_velocity.dot(propeller_up_vec)
            # v0 = 0    # static thrust

            # compute thrust
            f = self.calculate_thrust_force(v * d, self.area, self.propeller_pitch, v0)
            # f = self.mass * self.gravity / 4.

            # apply force in the simulation
            self.apply_external_force(force=[0, 0, f], link_id=jnt, position=(0., 0., 0.))

    def get_stationary_joint_velocity(self):
        fg = self.mass * self.gravity / 4.
        p = self.propeller_pitch
        return (2*np.pi / p) * (fg / (self.air_density * self.area) * (p / (self.k1 * self.diameter))**self.k2)**0.5

    def get_stationary_rpm(self):
        fg = self.mass * self.gravity / 4.
        p = self.propeller_pitch
        return (60 / p) * (fg / (self.air_density * self.area) * (p / (self.k1 * self.diameter))**self.k2)**0.5

    # def gravityCompensate(self):
    #     pass


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Quadcopter(sim)

    # print information about the robot
    robot.print_info()

    rpm = robot.get_stationary_rpm()
    print("Stationary RPM: {}".format(rpm))
    v = rpm_to_rad_per_second(rpm + 20)
    v = [v, -v, v, -v]

    # run simulation
    for i in count():
        robot.set_propeller_velocities(v)
        # step in simulation
        world.step(sleep_dt=1./240)
