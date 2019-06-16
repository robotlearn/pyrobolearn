#!/usr/bin/env python
"""In this file, we use and compare several IK libraries using the Kuka robot.

Namely, we compare: pybullet, PyKDL, trac_ik, and rbdl.

Set the `solver_flag` to a number between 0 and 4 (see lines [53,60]) to select which solver to select.
0 = pybullet + calculate_inverse_kinematics()
1 = pybullet + damped-least-squares IK using Jacobian (provided by pybullet)
2 = PyKDL
3 = trac_ik
4 = rbdl + damped-least-squares IK using Jacobian (provided by rbdl)
"""

import os
import numpy as np
from itertools import count

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA


# import PyKDL
try:
    import PyKDL as kdl
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `PyKDL`: '
                                'sudo apt-get install ros-<distribution>-python-orocos-kdl'
                                'or install it manually from `https://github.com/orocos/orocos_kinematics_dynamics`')

# import kdl_parser_py
try:
    import kdl_parser_py.urdf as KDLParser
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `kdl_parser_py`: '
                                'sudo apt-get install ros-<distribution>-kdl-parser-py')

# import track_ik_python
try:
    from trac_ik_python.trac_ik import IK as TracIK
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `trac_ik_python`: '
                                'sudo apt-get install ros-<distribution>-trac-ik-python')

# import rbdl
try:
    import rbdl
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `rbdl` manually from `https://bitbucket.org/rbdl/rbdl`')


# TO BE SET BY THE USER
# select IK solver, by setting the flag:
# 0 = pybullet + calculate_inverse_kinematics()
# 1 = pybullet + damped-least-squares IK using Jacobian (provided by pybullet)
# 2 = PyKDL
# 3 = trac_ik
# 4 = rbdl + damped-least-squares IK using Jacobian (provided by rbdl)
solver_flag = 1  # 1 and 4 gives pretty good results


# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
robot = KukaIIWA(sim)

# define useful variables for IK
dt = 1./240
link_id = robot.get_end_effector_ids(end_effector=0)
joint_ids = robot.joints  # actuated joint
base_name = robot.base_name
end_effector_name = robot.get_link_names(link_id)
urdf = os.path.dirname(os.path.abspath(__file__)) + '/../../../pyrobolearn/robots/urdfs/kuka/kuka_iiwa/iiwa14.urdf'
damping = 0.01  # for damped-least-squares IK
wrt_link_id = -1  # robot.get_link_ids('iiwa_link_1')
chain_name = robot.get_link_names(joint_ids)

# desired position
xd = np.array([0.5, 0., 0.5])
world.load_visual_sphere(xd, radius=0.05, color=(1, 0, 0, 0.5))

# joint_ids = joint_ids[2:]

# print information about the robot
print("")
print("Robot: {}".format(robot))
print("Number of DoFs: {}".format(robot.num_dofs))
print("Joint ids: {}".format(robot.joints))
print("Q Indices: {}".format(robot.get_q_indices()))
print("Actuated joint ids: {}".format(robot.joints))
print("Link names: {}".format(robot.get_link_names(robot.joints)))
print("End-effector names: {}".format(robot.get_link_names(robot.get_end_effector_ids())))
print("Floating base? {}".format(robot.has_floating_base()))
print("Total mass = {} kg".format(robot.mass))
print("")
print("Base name for IK: {}".format(base_name))
print("Link name for IK: {}".format(end_effector_name))
print("Chain: {}".format(chain_name))
print("")


robot.change_transparency()
robot.draw_link_frames([-1, 0])
robot.draw_bounding_boxes(joint_ids[0])
# robot.draw_link_coms([-1,0])

qIdx = robot.get_q_indices(joint_ids)

print(qIdx)
print(joint_ids)


#####################
# IK using pybullet #
#####################

# OPTION 1: using `calculate_inverse_kinematics`###
if solver_flag == 0:
    for _ in count():
        # # get current position in the task/operational space
        # x = robot.get_link_positions(link_id, wrt_link_id)
        x = robot.get_link_world_positions(link_id)
        # print("(xd - x) = {}".format(xd - x))

        # perform full IK
        q = robot.calculate_inverse_kinematics(link_id, position=xd)

        # set the joint positions
        robot.set_joint_positions(q[qIdx], joint_ids)

        # step in simulation
        world.step(sleep_dt=dt)

# OPTION 2: using Jacobian and manual damped-least-squares IK ###
elif solver_flag == 1:
    kp = 50    # 5 if velocity control, 50 if position control
    kd = 0  # 2*np.sqrt(kp)

    for _ in count():
        # get current position in the task/operational space
        # x = robot.get_link_positions(link_id, wrt_link_id)
        # dx = robot.get_link_linear_velocities(link_id, wrt_link_id)
        x = robot.get_link_world_positions(link_id)
        dx = robot.get_link_world_linear_velocities(link_id)
        print("(xd - x) = {}".format(xd - x))

        # Get joint configuration
        q = robot.get_joint_positions()

        # Get linear jacobian
        if robot.has_floating_base():
            J = robot.get_linear_jacobian(link_id, q=q)[:, qIdx + 6]
        else:
            J = robot.get_linear_jacobian(link_id, q=q)[:, qIdx]

        # Pseudo-inverse
        # Jp = robot.get_pinv_jacobian(J)
        # Jp = J.T.dot(np.linalg.inv(J.dot(J.T) + damping*np.identity(3)))   # this also works
        Jp = robot.get_damped_least_squares_inverse(J, damping)

        # evaluate damped-least-squares IK
        dq = Jp.dot(kp * (xd - x) - kd * dx)

        # set joint velocities
        # robot.set_joint_velocities(dq)

        # set joint positions
        q = q[qIdx] + dq * dt
        robot.set_joint_positions(q, joint_ids=joint_ids)

        # step in simulation
        world.step(sleep_dt=dt)


##################
# IK using PyKDL #
##################
elif solver_flag == 2:
    print("Using PyKDL:")
    model = KDLParser.treeFromFile(urdf)
    if model[0]:
        model = model[1]
    else:
        raise ValueError("Error during the parsing")

    # define the kinematic chain
    chain = model.getChain(base_name, end_effector_name)
    print("Number of joints in the chain: {}".format(chain.getNrOfJoints()))

    # define the FK solver
    FK = kdl.ChainFkSolverPos_recursive(chain)

    # define the IK Solver
    # IKV = kdl.ChainIkSolverVel_pinv(chain)
    # IK = kdl.ChainIkSolverPos_NR(chain, FK, IKV)
    IK = kdl.ChainIkSolverPos_LMA(chain)  # ,_maxiter=ik_max_iter, _eps_joints=ik_tol)

    # desired final cartesian position
    Fd = kdl.Frame(kdl.Vector(xd[0], xd[1], xd[2]))

    for _ in count():
        # get current position in the task/operational space
        # x = robot.get_link_positions(link_id, wrt_link_id)
        x = robot.get_link_world_positions(link_id)
        print("(xd - x) = {}".format(xd - x))

        # buffer to put the solution
        q_solved = kdl.JntArray(chain.getNrOfJoints())

        # initial joint positions
        q = robot.get_joint_positions(joint_ids)
        q_init = kdl.JntArray(chain.getNrOfJoints())
        for i, j in enumerate(q):
            q_init[i] = j

        # Solve IK
        IK.CartToJnt(q_init, Fd, q_solved)
        q_solved = np.array([q_solved[i] for i in range(q_solved.rows())])

        # set joint positions
        robot.set_joint_positions(q_solved.tolist(), joint_ids)

        # step in simulation
        world.step(sleep_dt=dt)


#####################
# IK using track_ik #
#####################
elif solver_flag == 3:
    # Documentation: https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_python/
    # read urdf
    urdf_string = open(urdf, 'r').read()

    # create IK solver
    ik_solver = TracIK(base_link=base_name, tip_link=end_effector_name, urdf_string=urdf_string, solve_type='Distance')

    # define upper and lower limits (optional)
    # lb, ub = -np.ones(6)*100, np.ones(6)*100
    # ik_solver.set_joint_limits(lb, ub)

    for _ in count():
        # get current position in the task/operational space
        # x = robot.get_link_positions(link_id, wrt_link_id)
        x = robot.get_link_world_positions(link_id)
        print("(xd - x) = {}".format(xd - x))

        # get current joint configuration and orientation (the orientation has to be specified when using trac_ik)
        q = robot.get_joint_positions(joint_ids)
        quat = robot.get_link_world_orientations(link_id)

        # get solution
        q = ik_solver.get_ik(q, x=xd[0], y=xd[1], z=xd[2], rx=quat.x, ry=quat.y, rz=quat.z, rw=quat.w)

        # set joint positions
        if q is not None:
            q = np.array(q)
            robot.set_joint_positions(q, joint_ids)
        else:
            print('got None')

        # step in the simulation
        world.step(sleep_dt=dt)


######################################################
# IK using RBDL and own damped-least-squares inverse #
######################################################
elif solver_flag == 4:
    # load model using rbdl
    model = rbdl.loadModel(urdf, verbose=False, floating_base=False)
    rbdl_link_id = model.GetBodyId('iiwa_link_7')  # end_effector_name)
    rbdl_id = [model.GetBodyId(name) - 1 for name in chain_name]
    print("RBDL Link ID: {}".format(rbdl_id))
    print("RBDL Q size: {}".format(model.q_size))
    print("RBDL number of DoFs: {}".format(model.dof_count))

    J = np.zeros((3, model.dof_count))

    def position_pd(x, xd, v, vd=np.zeros(3), ad=np.zeros(3), kp=100, kd=None):
        # if damping is not specified, make it critically damped
        if kd is None:
            kd = 2.0 * np.sqrt(kp)

        # return PD error
        return kp * (xd - x) + kd * (vd - v) + ad

    def damped_least_squares_ik(q, x, xd, v, damping, dt, kp=1, kd=None):
        err = position_pd(x, xd, v, kp=kp, kd=kd)
        body_point = np.zeros(3)
        rbdl.CalcPointJacobian(model, q, rbdl_link_id, body_point, J, update_kinematics=True)
        J_dagger = J.T.dot(np.linalg.inv(J.dot(J.T) + damping*np.identity(3)))
        dq = J_dagger.dot(err)
        q = q + dq * dt
        return q

    for _ in count():
        # get current position in the task/operational space
        # x = robot.get_link_positions(link_id, wrt_link_id)
        x = robot.get_link_world_positions(link_id)
        dx = robot.get_link_world_linear_velocities(link_id)
        print("(xd - x) = {}".format(xd - x))

        # get joint configuration
        q = robot.get_joint_positions()

        # get solution by performing damped-least-squares IK
        q = damped_least_squares_ik(q, x, xd, dx, damping=0., dt=dt, kp=50, kd=0)

        # set the joint positions
        robot.set_joint_positions(q[rbdl_id], joint_ids)

        # step in simulation
        world.step(sleep_dt=dt)
