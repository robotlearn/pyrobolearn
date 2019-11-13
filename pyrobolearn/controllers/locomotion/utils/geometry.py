#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tf import transformations
'''
geometry types:

position: [x,y,z]
quaternion: [qx,qy,qz,qw]
pose: [x,y,z,qx,qy,qz,qw]

twist: [vx,vy,vz,wx,wy,wz]
wrench: [fx,fy,fz,tx,ty,tz]

spatial_velocity = [wx,wy,wz,vx,vy,vz]
spatial_force = [tx,ty,tz,fx,fy,fz] 

'''
# homogeneous_vector = lambda P: np.append(P,1)
def homogeneous_vector(P):
    return np.hstack((P, 1))


# homogeneous_matrix = lambda rot=np.identity(3), pos=np.zeros(3): np.vstack((np.append(rot[0, :], pos[0]), np.append(rot[1, :], pos[1]), np.append(rot[2, :], pos[2]), np.array([0, 0, 0, 1])))
def homogeneous_matrix(rot=np.identity(3), pos=np.zeros(3)):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot[:3, :3]
    transform_matrix[:3, -1] = pos[:3]
    return transform_matrix

def transform_matrix(rot=np.identity(3), pos=np.zeros(3)):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot[:3, :3]
    transform_matrix[:3, -1] = pos[:3]
    return transform_matrix


def pose2transform(pose):
    position, quaternion = pose[:3], pose[-4:]
    rotation_matrix = transformations.quaternion_matrix(quaternion)
    rotation_matrix[:3, -1] = position
    return rotation_matrix

def transform2pose(transform):
    position = transform[:3,-1]
    quaternion = transformations.quaternion_from_matrix(transform)
    pose = np.concatenate((position, quaternion))
    return pose


def positionPD(pos_des, pos_cur, vel_des=np.zeros(3), vel_cur=np.zeros(3), acc_des=np.zeros(3), kp=100, kd=0.0):
    return kp * (pos_des - pos_cur) + kd * (vel_des - vel_cur) + acc_des

def rotationPD(rot_des, rot_cur, omega_des=np.zeros(3), omega_cur=np.zeros(3), omega_dot_des=np.zeros(3), kp=200,
               kd=0.0):
    vex = lambda M: 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    return kp * vex(rot_des.dot(rot_cur.T) - np.identity(3)) + kd * (omega_des - omega_cur) + omega_dot_des

def quaternion_error(quat_des, quat_cur):
    skew = lambda V: np.array([[0, -V[2], V[1]], [V[2], 0, -V[0]], [-V[1], V[0], 0]])
    diff_quat = quat_cur[-1] * quat_des[:3] - quat_des[-1] * quat_cur[:3] - skew(quat_des[:3]).dot(quat_cur[:3])
    return diff_quat

def quaternionPD(quat_des, quat_cur, omega_des=np.zeros(3), omega_cur=np.zeros(3), omega_dot_des=np.zeros(3), kp=100,
                 kd=0.0):
    return kp * quaternion_error(quat_des=quat_des, quat_cur=quat_cur) + kd * (omega_des - omega_cur) + omega_dot_des

def posePD(pose_des, pose_cur, spatial_velocity_des=np.zeros(6), spatial_velocity_cur=np.zeros(6), spatial_acceleration_des=np.zeros(6), kp_linear=100, kd_linear=10, kp_angular=100, kd_angular=10):
    error_linear = positionPD(pos_cur=pose_cur[:3], pos_des=pose_des[:3],
                              vel_cur=spatial_velocity_cur[-3:],
                              vel_des=spatial_velocity_des[-3:],
                              acc_des=spatial_acceleration_des[-3:],
                              kp=kp_linear, kd=kd_linear)
    error_angular = quaternionPD(quat_cur=pose_cur[-4:], quat_des=pose_des[-4:],
                                 omega_cur=spatial_velocity_cur[:3],
                                 omega_des=spatial_velocity_des[:3],
                                 omega_dot_des=spatial_acceleration_des[:3],
                                 kp=kp_angular, kd=kd_angular)
    error = np.concatenate((error_angular, error_linear))
    return error
