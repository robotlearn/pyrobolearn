# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Test with MuJoCo.
"""

import os
import time
import numpy as np
from itertools import count
# from pyrobolearn.simulators.bullet import Bullet
from pyrobolearn.simulators.mujoco import Mujoco
import mujoco_py as mujoco


# sim = Bullet(render=True)
sim = Mujoco(render=True, update_dynamically=True)
print("Gravity: {}".format(sim.get_gravity()))
print("Timestep: {}".format(sim.get_time_step()))
# sim.set_gravity(np.zeros(3))

# load floor
# floor = sim.load_floor(dimension=20)

print("qpos (before loading): ", sim.sim.data.qpos)

# create box
# box = sim.create_primitive_object(sim.GEOM_BOX, position=(0, 0, 2), mass=1, rgba_color=(1, 0, 0, 1))
# sphere = sim.create_primitive_object(sim.GEOM_SPHERE, position=[0.5, 0., 1.], mass=0, radius=0.05,
#                                      rgba_color=(1, 0, 0, 0.5))
# cylinder = sim.create_primitive_object(sim.GEOM_CYLINDER, position=(0, 2, 2), mass=1)
# capsule = sim.create_primitive_object(sim.GEOM_CAPSULE, position=(0, -2, 2), mass=1, rgba_color=(0, 0, 1, 1))

print("qpos (after loading sphere): ", sim.sim.data.qpos)

# print("Sphere id: ", sphere)
# print("Num bodies before loading robot: ", sim.num_bodies())

# load robot
path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/rrbot/pendulum.urdf'
# path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/rrbot/rrbot.urdf'
path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/franka/franka.urdf'
# path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/kuka/kuka_iiwa/iiwa14.urdf'
path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/hyq2max/hyq2max.urdf'
# path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/anymal/anymal.urdf'
# path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/centauro/centauro_stick.urdf'
robot_name = path.split('/')[-1].split('.')[0]
robot = sim.load_urdf(path, position=(0, 0, 0.8), use_fixed_base=True)

print("qpos (after loading robot): ", sim.sim.data.qpos)

print("Base position: ", sim.get_base_position(robot))

print("Num bodies after loading robot: ", sim.num_bodies())

# print("Robot")
# print("base name: ", sim.get_body_info(robot))
# print("mass: ", sim.get_base_mass(body_id=robot))

# sim.remove_body(sphere)
print(sim.sim.data.qpos)

# sim.step()
model = sim.model
mjc_sim = sim.sim
data = mjc_sim.data

# The ones that appear in the following are because of the floor
print("nbody", model.nbody - 1)  # total number of links
# print("nuser_body", model.nuser_body)
print("njnt", model.njnt)    # total number of joints
print("nq", model.nq)            # total number of generalized coordinates (=num_actuated_joints); for free joints 7
print("nv", model.nv)            # generalized velocities (nq - 1)
print("na", model.na)
print("nu", model.nu)
print("qpos", data.qpos)
print("qvel", data.qvel)
print("act", data.act)
# print("qpos", data.qpos, len(data.qpos))  # nqx1
print("body_dofnum: ", model.body_dofnum)
print("body_mass: ", model.body_mass)
print("body_subtreemass", model.body_subtreemass)
print("subtree_com", data.subtree_com)
print("body_xpos: ", data.body_xpos)
print("body pos: ", model.body_pos)
print("body_xquat: ", data.body_xquat)
# print("get_xpos: ", data.get_body_xpos(sim._bodies[sphere].tag_name))
print("xfrc_applied: ", data.xfrc_applied)

# joints
# print("jnt_type: ", [["free", "ball", "slide", "hinge"][idx] for idx in model.jnt_type])
# print("jnt_qposadr: ", model.jnt_qposadr)

data.body_xpos[1] = np.array(range(3))

num_joints = sim.num_joints(robot)
num_actuated_joints = sim.num_actuated_joints(robot)
num_links = sim.num_links(robot)

joint_ids = sim.get_joint_type_ids(robot, list(range(num_joints)))
joint_ids = np.array([i for i in range(num_joints) if joint_ids[i] != sim.JOINT_FIXED])

# define amplitude and angular velocity when moving the sphere
w = 0.01/2
r = 0.2

print("\nncam: ", model.ncam)
print("cam_xpos: ", data.cam_xpos)
print("cam_xmat: ", data.cam_xmat)
print("cam_fovy: ", model.cam_fovy)
print("Masses: ", sim.get_link_masses(robot))
print("Names: ", sim.get_link_names(robot))
print("Num links: ", num_links)
print("Num joints: ", num_joints)
print("Num actuated joints: ", num_actuated_joints)
print("Contacts: ", data.contact)
print("Sim state: ", mjc_sim.get_state())

print("time: ", data.time)

for i in range(num_joints):
    print(sim.get_joint_info(robot, i))

for i in range(num_links):
    print(sim.get_link_state(robot, i))

print("Jacobian: ", sim.calculate_jacobian(robot, num_actuated_joints))

data.qpos[:] = np.zeros(model.nq)

viewer = sim.viewer

print(viewer.cam)
print(dir(viewer.cam))

# sim.reset_joint_states(robot, positions=[8.84305270e-05, 7.11378917e-02, -1.68059886e-04, -9.71690439e-01,
#                                          1.68308810e-05, 3.71467111e-01, 5.62890805e-05])

print(sim.print_xml())

# TODO: the angles are reversed when setting qpos0
# TODO: the robots
if robot_name == 'franka':
    positions = np.array([0.0277854, -0.97229678, -0.028778385, -2.427800237, -0.086976557, 1.442695354, -0.711514286,
                          0., 0.])
    sim.reset_joint_states(robot, joint_ids=joint_ids, positions=positions)
elif robot_name == 'iiwa14':
    sim.reset_joint_state(robot, joint_id=3, position=-np.pi/2)
elif robot_name == 'pendulum':
    sim.reset_joint_state(robot, joint_id=1, position=np.pi/8)

# perform step
for t in count():
    # print("nbody", model.nbody)
    # print("njnt", model.njnt)
    # print("nq", model.nq)
    # print("nv", model.nv)
    # print("na", model.na)
    # print("nu", model.nu)
    # print("qpos", mjc_sim.data.qpos)  # nqx1
    # print("body_dofnum: ", model.body_dofnum)
    # print("body_mass: ", model.body_mass)
    # print("body_subtreemass", model.body_subtreemass)
    # if (t % 200) == 0:
    #     print("Resetting position")
    #     model.body_pos[1] = range(3)
    #     # data.qpos[:3] = range(3)
    #     print(model.body_pos)
    #     print(mjc_sim.data.body_xpos[1])

    # if t % 200 == 0:
    #     # print(mjc_sim.data.subtree_com)
    #     pos = np.zeros(3)
    #     jacp = np.zeros(3 * model.nv)
    #     jacr = np.zeros(3 * model.nv)
    #     mujoco.functions.mj_jac(model, data, jacp, jacr, pos, 4)
    #     print(jacp)
    #     print(jacr)
    #     # model.body_pos[1] = range(3)
    #     # sim.reset_base_position(sphere, [2, -1, 3])

    # position = np.array([0.5, r * np.cos(w * t + np.pi / 2), (1. - r) + r * np.sin(w * t + np.pi / 2)])
    # sim.reset_base_position(sphere, position)
    # data.qpos[:] = np.zeros(model.nq)

    # print joint positions
    # print(sim.get_joint_positions(robot))
    # sim.set_joint_positions(robot, joint_ids, 0 * np.ones(num_actuated_joints))
    # sim.reset_joint_states(robot, joint_ids=joint_ids, positions=positions)
    # sim.set_joint_positions(robot, joint_ids, positions)
    # sim.set_joint_positions(robot, joint_ids, [0., 0., 0., np.pi/2, 0., 0., 0.])

    sim.set_joint_positions(robot, joint_ids, np.zeros(num_actuated_joints), kps=50, kds=1)
    # sim.set_joint_positions(robot, joint_ids, np.pi/2 * np.ones(num_actuated_joints), kps=100, kds=10)
    # sim.set_joint_positions(robot, joint_ids=1, positions=np.pi/2, kps=100, kds=10)
    # sim.set_joint_velocities(robot, joint_ids, np.zeros(num_actuated_joints))
    # sim.set_joint_velocities(robot, joint_ids, velocities=5 * np.ones(num_actuated_joints))
    # sim.set_joint_torques(robot, joint_ids, torques=np.zeros(num_actuated_joints))
    # sim.set_joint_torques(robot, joint_ids, torques=5 * np.ones(num_actuated_joints))

    # if t == 500:
    #     cylinder = sim.create_primitive_object(sim.GEOM_CYLINDER, position=(0, 2, 2), mass=1)

    # if t == 2000:
    #     sim.remove_body(cylinder)

    # qpos = data.qpos
    # print("time: ", data.time)
    # # print(qpos)
    # data.qpos[:] = np.zeros(len(qpos))
    # print(data.xfrc_applied.shape)
    # print(data.mocap_quat)

    # print(mjc_sim.data.contact)
    sim.step(sleep_time=sim.dt)
