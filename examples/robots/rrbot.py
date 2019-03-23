#!/usr/bin/env python
"""Load the RRBot robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import RRBot

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# load robot
robot = RRBot(sim)
# robot.add_joint_slider()

print("Robot: {}".format(robot))
print("Total number of joints: {}".format(robot.num_joints))
print("Joint names: {}".format(robot.get_joint_names(range(robot.num_joints))))
print("Link names: {}".format(robot.get_link_names(range(robot.num_joints))))

print("Number of DoFs: {}".format(robot.num_dofs))
print("Robot actuated joint ids: {}".format(robot.joints))
print("Actuated joint names: {}".format(robot.get_joint_names()))
print("Actuated link names: {}".format(robot.get_link_names()))
print("Current joint positions: {}".format(robot.get_joint_positions()))

print("Number of end-effectors: {}".format(robot.num_end_effectors))
print("End-effector names: {}".format(robot.get_link_names(robot.end_effectors)))

print("Robot base position: {}".format(robot.get_base_position()))
robot.change_transparency()
visuals = robot.sim.get_visual_shape_data(robot.id)
visuals = {visual[1]: visual[3] for visual in visuals}

# robot.draw_link_coms()
robot.draw_link_frames()
# robot.draw_bounding_boxes()

for i in robot.joints:
    print("Link {}".format(i))
    state = robot.sim.get_link_state(robot.id, i)
    info = robot.sim.get_joint_info(robot.id, i)

    print("\t CoM world position: {}".format(state[0]))
    print("\t Local inertial frame position: {}".format(state[2]))
    print("\t World link frame position: {}".format(state[4]))

    print("\t CoM world orientation: {}".format(state[1]))
    print("\t Local inertial frame orientation: {}".format(state[3]))
    print("\t World link frame orientation: {}".format(state[5]))

    print("\t Joint axis: {}".format(info[-4]))
    if i in visuals:
        print("\t Dimensions: {}".format(visuals[i]))

for _ in count():
    world.step(sleep_dt=1./240)

raw_input('press enter')

print("Inertia matrix: {}".format(np.array(sim.calculate_mass_matrix(robot.id, [0., 0., 0., 0., 0., 0.]))))
linkId = 2
com_frame = robot.get_link_states(linkId)[2]
q = robot.get_joint_positions()
print(com_frame)
# com_frame = [0.,0.,0.]
print("Jacobian matrix: {}".format(sim.calculate_jacobian(robot.id, linkId, com_frame)))

Jlin = robot.get_jacobian(linkId + 1)[:3]
print("Jacobian matrix: {}".format(Jlin))

robot.draw_velocity_manipulability_ellipsoid(linkId + 1, Jlin)

Jlin = robot.get_jacobian(linkId)[:3]
robot.draw_velocity_manipulability_ellipsoid(linkId, Jlin, color=(1, 0, 0, 0.7))

cnt = 0
for i in count():
    if i%240 == 0:
        if cnt < 3:
            Jlin = robot.get_jacobian(linkId + 1)[:3]
            robot.draw_velocity_manipulability_ellipsoid(linkId + 1, Jlin)
        cnt += 1
    world.step(sleep_dt=1./240)
    # robot.set_joint_torques()

raw_input('press enter')

print(robot.get_link_names())
force = np.array([1., 0., 0.])
pos = np.array([0., 0., 0.])
sim.apply_external_force(robot.id, 1, force, pos, flags=p.LINK_FRAME) # link_frame = 1

slider = sim.add_user_debug_parameter('force', -1000., 1000., 0)

dq, ddq = [0., 0.], [0., 0.]
J = sim.calculate_jacobian(robot.id, 1, [0., 0., 0.])
print(np.array(J[0]))

a = robot.get_joint_positions()
# print(robot.getJacobianMatrix(1, np.array([0.,0.]))) # TODO: need to convert numpy array to list

linkId = 2
com_frame = robot.get_link_states(linkId)[2]
xdes = np.array(robot.get_link_world_positions(linkId))
K = 100*np.identity(3)
D = 2*np.sqrt(K)  # critically damped
D = 3*D  # manually increase damping

# run simulator
for i in range(10000):
    joint_states = p.get_joint_states(robot.id, robot.joints)
    # print("joint state: ", joint_states)
    q = [joint_state[0] for joint_state in joint_states]
    dq = [joint_state[1] for joint_state in joint_states]

    # q = robot.get_joint_positions().tolist()
    # dq = robot.get_joint_velocities().tolist()
    x = np.array(robot.get_link_world_positions(linkId))
    dx = np.array(robot.get_link_world_linear_velocities(linkId))
    tau = robot.calculate_inverse_dynamics(ddq, dq, q)  # Coriolis, centrifugal and gravity compensation
    Jlin = np.array(sim.calculate_jacobian(robot.id, linkId, com_frame)[0])
    F = K.dot(xdes - x) - D.dot(dx)  # compute cartesian forces
    # print("force: {}".format(F))
    tau += Jlin.T.dot(F)  # cartesian PD with gravity compensation
    # tau += Jlin.T.dot(- D.dot(dx))  # active compliance

    # tau = Jlin.T.dot(F)

    # compute manipulability measure :math:`w = sqrt(det(JJ^T))`
    Jlin = Jlin[[0, 2], :]
    w = np.sqrt(np.linalg.det(Jlin.dot(Jlin.T)))
    # print("manipulability: {}".format(w))

    # Impedance/Torque control
    sim.set_joint_motor_control(robot.id, robot.joints, sim.TORQUE_CONTROL, forces=tau)

    force = sim.read_user_debug_parameter(slider)
    if force > 0:
        force = np.array([0., 0., 1.])
    else:
        force = np.array([0., 0., 0.])
    sim.apply_external_force(robot.id, linkId, force, pos, flags=p.LINK_FRAME)  # p.LINK_FRAME = 1

    # robot.update_joint_slider()
    world.step(sleep_dt=1./240)
