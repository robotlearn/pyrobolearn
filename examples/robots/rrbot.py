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
# robot.addJointSlider()

print("Robot: {}".format(robot))
print("Total number of joints: {}".format(robot.getNumberOfJoints()))
print("Joint names: {}".format(robot.getJointNames(range(robot.getNumberOfJoints()))))
print("Link names: {}".format(robot.getLinkNames(range(robot.getNumberOfJoints()))))

print("Number of DoFs: {}".format(robot.getNumberOfDoFs()))
print("Robot actuated joint ids: {}".format(robot.joints))
print("Actuated joint names: {}".format(robot.getJointNames()))
print("Actuated link names: {}".format(robot.getLinkNames()))
print("Current joint positions: {}".format(robot.getJointPositions()))

print("Number of end-effectors: {}".format(robot.getNumberOfEndEffectors()))
print("End-effector names: {}".format(robot.getEndEffectorNames()))

print("Robot base position: {}".format(robot.getBasePosition()))
robot.changeTransparency()
visuals = robot.sim.getVisualShapeData(robot.id)
visuals = {visual[1]: visual[3] for visual in visuals}

# robot.drawLinkCoMs()
robot.drawLinkFrames()
# robot.drawBoundingBoxes()

for i in robot.joints:
    print("Link {}".format(i))
    state = robot.sim.getLinkState(robot.id, i)
    info = robot.sim.getJointInfo(robot.id, i)

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

print("Inertia matrix: {}".format(np.array(sim.calculateMassMatrix(robot.id, [0.,0.,0.,0.,0.,0.]))))
linkId = 2
com_frame = robot.getLinkStates(linkId)[2]
q = robot.getJointPositions()
print(com_frame)
# com_frame = [0.,0.,0.]
print("Jacobian matrix: {}".format(np.vstack((sim.calculateJacobian(robot.id, linkId, com_frame, q.tolist(), [0.,0.], [0.,0.])))))

Jlin = robot.calculateJacobian(linkId+1, localPosition=(0.,0.,0.))[:3]
print("Jacobian matrix: {}".format(Jlin))

robot.drawVelocityManipulabilityEllipsoid(linkId+1, Jlin)

Jlin = robot.calculateJacobian(linkId)[:3]
robot.drawVelocityManipulabilityEllipsoid(linkId, Jlin, color=(1,0,0,0.7))

cnt = 0
for i in count():
    if i%240 == 0:
        if cnt < 3:
            Jlin = robot.calculateJacobian(linkId + 1, localPosition=(0., 0., 0.))[:3]
            robot.drawVelocityManipulabilityEllipsoid(linkId + 1, Jlin)
        cnt += 1
    world.step(sleep_dt=1./240)
    # robot.setJointTorques()

raw_input('press enter')

print(robot.getLinkNames())
force = np.array([1., 0., 0.])
pos = np.array([0., 0., 0.])
sim.applyExternalForce(robot.id, 1, force, pos, flags=p.LINK_FRAME) # link_frame = 1

slider = sim.addUserDebugParameter('force', -1000., 1000., 0)

dq, ddq = [0., 0.], [0., 0.]
J = sim.calculateJacobian(robot.id, 1, [0.,0.,0.], [0.,0.], dq, ddq)
print(np.array(J[0]))

a = robot.getJointPositions()
# print(robot.getJacobianMatrix(1, np.array([0.,0.]))) # TODO: need to convert numpy array to list

linkId = 2
com_frame = robot.getLinkStates(linkId)[2]
xdes = np.array(robot.getLinkWorldPositions(linkId))
K = 100*np.identity(3)
D = 2*np.sqrt(K)  # critically damped
D = 3*D  # manually increase damping

# run simulator
for i in range(10000):
    joint_states = p.getJointStates(robot.id, robot.joints)
    # print("joint state: ", joint_states)
    q = [joint_state[0] for joint_state in joint_states]
    dq = [joint_state[1] for joint_state in joint_states]

    # q = robot.getJointPositions().tolist()
    # dq = robot.getJointVelocities().tolist()
    x = np.array(robot.getLinkWorldPositions(linkId))
    dx = np.array(robot.getLinkWorldLinearVelocities(linkId))
    tau = robot.calculateID(q, dq, ddq)  # Coriolis, centrifugal and gravity compensation
    Jlin = np.array(sim.calculateJacobian(robot.id, linkId, com_frame, q, [0.,0.], ddq)[0])
    F = K.dot(xdes - x) - D.dot(dx) # compute cartesian forces
    # print("force: {}".format(F))
    tau += Jlin.T.dot(F) # cartesian PD with gravity compensation
    # tau += Jlin.T.dot(- D.dot(dx))  # active compliance

    # tau = Jlin.T.dot(F)

    # compute manipulability measure :math:`w = sqrt(det(JJ^T))`
    Jlin = Jlin[[0, 2], :]
    w = np.sqrt(np.linalg.det(Jlin.dot(Jlin.T)))
    # print("manipulability: {}".format(w))

    # Impedance/Torque control
    sim.setJointMotorControlArray(robot.id, robot.joint_indices, sim.TORQUE_CONTROL, forces=tau)

    force = sim.readUserDebugParameter(slider)
    if force > 0:
        force = np.array([0., 0., 1.])
    else:
        force = np.array([0., 0., 0.])
    sim.applyExternalForce(robot.id, linkId, force, pos, flags=p.LINK_FRAME)  # p.LINK_FRAME = 1

    # robot.updateJointSlider()
    world.step(sleep_dt=1./240)
