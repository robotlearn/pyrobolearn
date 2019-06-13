#!/usr/bin/env python
"""Provide the techpod platform.
"""

import os
import json
import numpy as np

from pyrobolearn.robots.uav import FlappingWingUAV


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = "Fei et al."
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SimpleNamespace:
    """A simple object subclass that provides attribute access to its namespace, as well as a meaningful repr.

    Taken from: https://docs.python.org/3/library/types.html
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Wing(object):
    r"""Wing

    Python code translated from C++ code provided in [1].

    References:
        [1] https://github.com/purdue-biorobotics/flappy/blob/master/flappy/envs/Wing.cpp
        [2] "Flappy Hummingbird: An Open Source Dynamic Simulation of Flapping Wing Robots and Animals", Fei et al.,
            2019
    """

    def __init__(self, wing_index, wing_length, mean_chord, r33, r22, r11, r00, z_cp2, z_cp1, z_cp0, z_rd,
                 shoulder_width, stroke_plane_offset):
        r"""
        Initialize the wing and compute the dynamics on it.

        Args:
            wing_index (int): wing index (0 or 1). This is used to determine the sign of the axis for the motors.
                Set it to 0 for the left wing, and 1 for the right wing.
            wing_length (float): length of wing.
            mean_chord (float): mean of the chord. "In aeronautics, a chord is the imaginary straight line joining the
                leading and trailing edges of an aerofoil. The chord length is the distance between the trailing edge
                and the point on the leading edge where the chord intersects the leading edge." (from Wikipedia)
            r33 (float): wing geometry constant.
            r22 (float): wing geometry constant.
            r11 (float): wing geometry constant.
            r00 (float): wing geometry constant.
            z_cp2 (float): wing geometry constant.
            z_cp1 (float): wing geometry constant.
            z_cp0 (float): wing geometry constant.
            z_rd (float): wing geometry constant.
            shoulder_width (float): should width
            stroke_plane_offset (float): stroke plane offset.
        """
        self.air_density = 1.18009482370369
        self.wing_index = wing_index
        self.wing_length = wing_length
        self.mean_chord = mean_chord
        self.r33 = r33
        self.r22 = r22
        self.r11 = r11
        self.r00 = r00
        self.z_cp2 = z_cp2
        self.z_cp1 = z_cp1
        self.z_cp0 = z_cp0
        self.z_rd = z_rd
        self.sign = pow(-1, wing_index)

        self.r_w = wing_length
        self.d_0 = shoulder_width
        self.d_s = stroke_plane_offset
        self.r_cp = self.r_w * r33/r22  # span-wise center of pressure

        # total force and moments
        self.span_wise_center_of_pressure = 0
        self.cord_wise_center_of_pressure = 0
        self.normal_force = 0
        self.aero_moment = 0
        self.rotational_damping_moment = 0

        # rotational damping moment coefficient
        self.Crd = 5.0

    def do_nothing(self):
        # total force and moments
        self.span_wise_center_of_pressure = 0
        self.cord_wise_center_of_pressure = 0
        self.normal_force = 0
        self.aero_moment = 0
        self.rotational_damping_moment = 0

    def update_aero_force(self):
        """Update the aerodynamic forces applied on the wing; i.e. the normal force, aerodynamic moment and rotational
        damping moment (the equations are provided in [2]).
        """
        self.update_velocity_coefficients()
        self.update_angle_of_attack()
        self.CN = self.get_CN(self.alpha)
        self.d_cp = self.get_center_of_pressure(self.alpha)
        self.update_velocity_squared_coefficients()

        # the normal force, aerodynamic moment and rotational damping moment are given in [2]
        self.normal_force = 0.5 * self.air_density * self.mean_chord * self.CN * \
                            (self.a_u2 * self.r_w**3 * self.r22 + self.a_u1 * self.r_w**2 * self.r11 +
                             self.a_u0 * self.r_w * self.r00)
        self.aero_moment = -0.5 * self.air_density * self.d_cp * self.CN * self.mean_chord**2 * \
                           (self.a_u2 * self.r_w**3 * self.z_cp2 + self.a_u1 * self.r_w**2 * self.z_cp1 +
                            self.a_u0 * self.r_w * self.z_cp0)

        if self.normal_force != 0:
            self.cord_wise_center_of_pressure = -self.aero_moment / self.normal_force
        else:
            self.cord_wise_center_of_pressure = 0

        self.span_wise_center_of_pressure = self.r_cp

        self.rotational_damping_moment = -0.125 * self.air_density * np.abs(self.dtheta) * self.dtheta * self.Crd * \
                                         self.r_w * self.mean_chord**4 * self.z_rd

    def update_state(self, body_velocity_rpy, body_velocity, stroke_plane_angle, stroke_plane_velocity, stroke_angle,
                     stroke_velocity, deviation_angle, deviation_velocity, rotate_angle, rotate_velocity):
        """Update the state of the wing."""
        self.velocity = body_velocity
        self.drpy = body_velocity_rpy

        # stroke plane
        self.Phi = stroke_plane_angle
        self.dPhi = stroke_plane_velocity

        # stroke
        self.psi = stroke_angle
        self.dpsi = stroke_velocity

        # deviation
        self.phi = deviation_angle
        self.dphi = deviation_velocity

        # wing rotation
        self.theta = rotate_angle
        self.dtheta = rotate_velocity

        # wing trigonometry pre calculation
        self.s_Phi = np.sin(self.Phi)
        self.c_Phi = np.cos(self.Phi)
        self.s_psi = np.sin(self.psi)
        self.c_psi = np.cos(self.psi)
        self.s_phi = np.sin(self.phi)
        self.c_phi = np.cos(self.phi)

    def update_velocity_coefficients(self):
        """Update the velocity coefficients."""
        u, v, w = self.velocity
        p, q, r = self.drpy
        s_Phi, c_Phi, s_psi, c_psi, s_phi, c_phi = self.s_Phi, self.c_Phi, self.s_psi, self.c_psi, self.s_phi, \
                                                   self.c_phi
        dPhi, dpsi, dphi = self.dPhi, self.dpsi, self.dphi
        d_0, d_s = self.d_0, self.d_s

        # velocity coefficients
        self.u_o1 = self.sign * (p * c_psi * c_Phi - dPhi * s_psi) + (q * s_psi + r * c_psi * s_Phi + dphi)
        self.u_o0 = self.sign * (-(u + q * d_s) * c_phi * s_Phi - r * d_0 * s_phi * s_psi * c_Phi - (v - p * d_s) *
                                 s_phi * c_psi + w * s_phi * s_psi * s_Phi + p * d_0 * c_phi * c_Phi) + \
                    ((u + q * d_s) * s_phi * s_psi * c_Phi + r * d_0 * c_phi * s_Phi + w * c_phi * c_Phi + p * d_0 *
                     s_phi * s_psi * s_Phi)
        self.u_i1 = self.sign * (p * s_phi * s_psi * c_Phi + r * c_phi * c_Phi + dPhi * s_phi * c_psi) + \
                    (-p * c_phi * s_Phi - q * s_phi * c_psi + r * s_phi * s_psi * s_Phi + dpsi * c_phi)
        self.u_i0 = self.sign * (r * d_0 * c_psi * c_Phi - (v - p * d_s) * s_psi - w * c_psi * s_Phi) + \
                    (-(u + q * d_s) * c_psi * c_Phi - p * d_0 * c_psi * s_Phi)

    def update_angle_of_attack(self):
        """Update the angle of attack (AoA)."""
        # AoA correction double
        self.u_i = self.u_i1 * self.r_cp + self.u_i0

        if self.u_i != 0:
            self.delta_alpha = np.arctan((self.u_o1 * self.r_cp + self.u_o0) / (self.u_i1 * self.r_cp + self.u_i0))
        else:
            self.delta_alpha = 0

        # geometric AoA
        self.alpha_0 = self.theta + np.double(np.sign(self.u_i)) * np.pi / 2

        self.alpha = self.alpha_0 - self.delta_alpha

    @staticmethod
    def get_CN(alpha):
        # see equation (8) in paper [2]
        return 1.8 * np.sin(2 * alpha) * np.cos(alpha) + 1.95 * np.sin(alpha) - 1.5 * np.cos(2 * alpha) * np.sin(alpha)

    @staticmethod
    def get_center_of_pressure(alpha):
        return 0.46 - 0.332 * np.cos(alpha) - 0.037 * np.cos(3 * alpha) - 0.013 * np.cos(5 * alpha)

    def update_velocity_squared_coefficients(self):
        # velocity squared coefficients
        self.a_u2 = self.u_i1**2 + self.u_o1**2
        self.a_u1 = 2 * self.u_i1 * self.u_i0 + 2 * self.u_o1 * self.u_o0
        self.a_u0 = self.u_i0**2 + self.u_o0**2


class Actuator:
    r"""Actuator

    Code copied-pasted from [1] (I didn't change anything except adding some whitespaces).

    References:
        [1] https://github.com/purdue-biorobotics/flappy/blob/master/flappy/envs/Wing.cpp
        [2] "Flappy Hummingbird: An Open Source Dynamic Simulation of Flapping Wing Robots and Animals", Fei et al.,
            2019
    """

    def __init__(self, motor_properties):

        config = SimpleNamespace(**motor_properties)
        self.resistance = config.resistance
        self.torque_constant = config.torque_constant
        self.gear_ratio = config.gear_ratio
        self.mechanical_efficiency = config.mechanical_efficiency
        self.friction_coefficient = config.friction_coefficient
        self.damping_coefficient = config.damping_coefficient
        self.inertia = config.inertia

        self.inertia_torque = 0
        self.damping_torque = 0
        self.friction_torque = 0
        self.magnetic_torque = 0
        self.motor_torque = 0

        self.voltage = 0

        self.current = 0
        self.back_EMF = 0
        self.output_torque = 0
        self.config = config
        self.reset()

    def update_driver_voltage(self, voltage):
        self.voltage = voltage

    def update_torque(self, stroke_velocity, stroke_acceleration):
        psi_dot = stroke_velocity
        psi_ddot = stroke_acceleration
        motor_vel = psi_dot * self.gear_ratio
        motor_accel = psi_ddot * self.gear_ratio
        if psi_dot > 0:
            sign = 1
        elif psi_dot < 0:
            sign = -1
        else:
            sign = 0

        self.back_EMF = self.torque_constant * motor_vel
        self.current = (self.voltage - self.back_EMF) / self.resistance

        self.inertia_torque = self.inertia * motor_accel
        self.damping_torque = self.damping_coefficient * motor_vel
        self.friction_torque = self.friction_coefficient * sign
        self.magnetic_torque = self.torque_constant * self.current

        self.motor_torque = self.magnetic_torque - self.inertia_torque - self.damping_torque - self.friction_torque

        self.output_torque = self.motor_torque * self.gear_ratio * self.mechanical_efficiency

    def get_torque(self):
        return self.output_torque

    def reset(self):
        self.inertia_torque = 0
        self.damping_torque = 0
        self.friction_torque = 0
        self.magnetic_torque = 0
        self.motor_torque = 0

        self.current = 0
        self.back_EMF = 0
        self.output_torque = 0


class Flappy(FlappingWingUAV):
    r"""Flappy Hummingbird UAV (from Purdue University)

    This is the main class for the flappy hummingbird (a flapping wing micro aerial vehicle (FWMAV)). Most of the code
    as well as the URDF model comes from [2,3].

    The flappy hummingbird has 2 wings and each one has 2 degrees of freedom; the stroke and rotation angles.

    Warnings: Currently, in pybullet there is no air, so we simulate all the forces acting on the flappy vehicle as
    described in the paper and code [2,3]. The gravity is carried out by pybullet.

    References:
        [1] "Design Optimization and System Integration of Robotic Hummingbird", Zhang et al., 2017
        [2] "Flappy Hummingbird: An Open Source Dynamic Simulation of Flapping Wing Robots and Animals", Fei et al.,
            2019
        [3] https://github.com/purdue-biorobotics/flappy
    """

    def __init__(self, simulator, position=(0, 0, 0.5), orientation=(0, 0, 0, 1), fixed_base=False, scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/flappy/flappy.urdf',
                 config=os.path.dirname(__file__) + '/urdfs/flappy/config/mav_config.json'):
        super(Flappy, self).__init__(simulator, urdf, position, orientation, fixed_base)

        with open(config) as f:
            config = json.load(f)[0]
            config = SimpleNamespace(**config)

        # create wings
        self.left_wing = Wing(0, config.wing_length, config.mean_chord, config.r33, config.r22, config.r11, config.r00,
                              config.z_cp2, config.z_cp1, config.z_cp0, config.z_rd, config.left_shoulder_width,
                              config.stroke_plane_offset)
        self.right_wing = Wing(1, config.wing_length, config.mean_chord, config.r33, config.r22, config.r11, config.r00,
                               config.z_cp2, config.z_cp1, config.z_cp0, config.z_rd, config.right_shoulder_width,
                               config.stroke_plane_offset)

        # create motors
        self.left_motor = Actuator(config.left_motor_properties)
        self.right_motor = Actuator(config.right_motor_properties)

        # joints
        self.left_wing_joints = [self.get_link_ids(link) for link in ['left_leading_edge', 'left_wing']
                                 if link in self.link_names]
        self.right_wing_joints = [self.get_link_ids(link) for link in ['right_leading_edge', 'right_wing']
                                  if link in self.link_names]
        self.wings = [self.left_wing_joints, self.right_wing_joints]
        # joints = [left stroke, left rotate, right stroke, right rotate]
        self.wing_joints = self.left_wing_joints + self.right_wing_joints

        # dummy variables for now
        # self.driver_update_time = 0
        # self.dt_driver = 1./1e3
        self.prev_t = None
        self.prev_joint_velocities = None

    def apply_voltage(self, t, input_voltage):  # step(self, t, input_voltage):
        # get the joint positions for the left and right wing joints
        joint_positions = self.get_joint_positions(self.wing_joints)
        joint_velocities = self.get_joint_velocities(self.wing_joints)
        # joint_accelerations = self.get_joint_accelerations(self.wing_joints)
        if self.prev_t is None:
            joint_accelerations = np.zeros(len(self.wing_joints))
        else:
            joint_accelerations = (joint_velocities - self.prev_joint_velocities) / (t - self.prev_t)

        # update aerodynamic forces
        self.left_wing.update_state(self.angular_velocity, self.linear_velocity, 0, 0, joint_positions[0],
                                    joint_velocities[0], 0, 0, joint_positions[1], joint_velocities[1])
        self.right_wing.update_state(self.angular_velocity, self.linear_velocity, 0, 0, joint_positions[2],
                                     joint_velocities[2], 0, 0, joint_positions[3], joint_velocities[3])
        self.left_wing.update_aero_force()
        self.right_wing.update_aero_force()

        # update voltage
        # if t >= self.driver_update_time:
        #     self.driver_update_time += self.dt_driver
        self.left_motor.update_driver_voltage(input_voltage[0])
        self.right_motor.update_driver_voltage(input_voltage[1])

        # update torque (left and right strokes)
        self.left_motor.update_torque(joint_velocities[0], joint_accelerations[0])
        self.right_motor.update_torque(joint_velocities[2], joint_accelerations[2])

        # apply stroke torque
        torques = np.zeros(self.num_dofs)
        torques[0] = self.left_motor.get_torque()
        torques[2] = self.right_motor.get_torque()
        self.set_joint_torques(torques)

        # get aero forces
        left_normal_force = np.array([self.left_wing.normal_force, 0, 0])  # in wing x direction
        right_normal_force = np.array([self.right_wing.normal_force, 0, 0])
        left_cop = np.array([0, self.left_wing.span_wise_center_of_pressure,
                             (-1) * self.left_wing.cord_wise_center_of_pressure])
        right_cop = np.array([0, (-1) * self.right_wing.span_wise_center_of_pressure,
                              (-1) * self.right_wing.cord_wise_center_of_pressure])
        left_rot_damping_moment = np.array([0, self.left_wing.rotational_damping_moment, 0])  # in wing y direction
        right_rot_damping_moment = np.array([0, self.right_wing.rotational_damping_moment, 0])

        # apply aero force and moment on wing
        # self.apply_external_force(left_normal_force, link_id=1, position=left_cop, frame=BulletSim.LINK_FRAME)
        # self.apply_external_force(right_normal_force, link_id=3, position=right_cop, frame=BulletSim.LINK_FRAME)
        # self.apply_external_torque(left_rot_damping_moment, link_id=1, frame=BulletSim.LINK_FRAME)
        # self.apply_external_torque(right_rot_damping_moment, link_id=3, frame=BulletSim.LINK_FRAME)

        # save
        self.prev_t = t
        self.prev_joint_velocities = joint_velocities


# Test
if __name__ == "__main__":
    import time
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Flappy(sim)

    # print information about the robot
    robot.print_info()
    robot.add_joint_slider(robot.left_wing_joints)

    # run simulation
    for i in count():
        robot.update_joint_slider()
        # signal = 3 * np.sin(2. * np.pi * i/1000) * np.ones(2)
        # robot.apply_voltage(time.time(), signal)
        # robot.set_joint_positions(1. * np.sin(2 * np.pi * i/240), joint_ids=0)
        # step in simulation
        world.step(sleep_dt=1./240)
