#!/usr/bin/env python
"""Provide the Cartpole robotic platform.
"""

import os
import pybullet_data
import sympy
import sympy.physics.mechanics as mechanics

from robot import Robot
from pyrobolearn.utils.orientation import getSymbolicMatrixFromAxisAngle


class CartPole(Robot):
    r"""CartPole robot

    In its original formulation, the Cart-pole robot is a cart mounted by an inverted pendulum.
    The number of links for the pendulum can be specified during runtime.

    References:
        [1] "Reinforcement Learning: an Introduction", Barto and Sutton, 1998
        [2] Cartpole bullet environment:
            github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py
        [3] "PyDy Tutorial: Human Standing": https://github.com/pydy/pydy-tutorial-human-standing
        [4] "Dynamics with Python balancing the five link pendulum": http://www.moorepants.info/blog/npendulum.html
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 scaling=1.,
                 useFixedBase=True,
                 urdf_path=os.path.join(pybullet_data.getDataPath(), "cartpole.urdf"),
                 num_links=1,
                 inverted_pole=False,
                 pole_mass=1):   # pole_mass=10
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(CartPole, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'cartpole'

        # create dynamically other links if necessary
        # Refs:
        # 1. https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12345
        # 2. https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/urdfEditor.py
        # if num_links > 1:
        #     # get info about the link (dimensions, mass, etc).
        #     jointInfo = self.getJointInfo(self.joints[-1])
        #     linkInfo = self.getLinkStates(self.joints[-1])
        #     dynamicInfo = self.sim.g
        #
        #     dims =
        #
        #     info =
        #     jointType = info[2]
        #     jointAxis, parent = info[-4]
        #     parentLinkIndex = info
        #
        #     # create visual, collision shapes, and the body
        #     collision_shape = self.sim.createCollisionShape(self.sim.GEOM_BOX, halfExtents=dimensions)
        #     visual_shape = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=dimensions, rgbaColor=color)
        #
        #     for i in range(num_links - 1):
        #         # create new link and attached it to the previous link
        #         linkId = self.sim.createMultiBody(baseMass=mass,
        #                                           baseCollisionShapeIndex=collision_shape,
        #                                           baseVisualShapeIndex=visual_shape,
        #                                           basePosition=position,
        #                                           baseOrientation=orientation,
        #                                           linkParentIndices=[self.joints[-1]],
        #                                           linkJointTypes=[self.sim.JOINT_REVOLUTE])

        # create dynamically the cartpole because currently we can not add a link with a revolute joint in PyBullet;
        # we have to build the whole multibody system
        # The values are from the cartpole URDF: https://github.com/bulletphysics/bullet3/blob/master/data/cartpole.urdf

        # remove body
        self.sim.removeBody(self.id)

        # create slider
        dims = (15, 0.025, 0.025)
        color = (0, 0.8, 0.8, 1)
        mass = 0
        position = (0, 0, 0)
        orientation = (0, 0, 0, 1)
        collisionShape = self.sim.createCollisionShape(self.sim.GEOM_BOX, halfExtents=dims)
        visualShape = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=dims, rgbaColor=color)

        # create cart and pole
        cartDims = (0.25, 0.25, 0.1)
        cartCollisionShape = self.sim.createCollisionShape(self.sim.GEOM_BOX, halfExtents=cartDims)
        cartVisualShape = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=cartDims, rgbaColor=(0,0,0.8,1))
        poleDims = (0.025, 0.025, 0.5)
        poleCollisionShape = self.sim.createCollisionShape(self.sim.GEOM_BOX, halfExtents=poleDims)
        poleVisualShape = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=poleDims, rgbaColor=(1,1,1,1))
        radius = 0.05
        sphereCollisionShape = self.sim.createCollisionShape(self.sim.GEOM_SPHERE, radius=radius)
        sphereVisualShape = self.sim.createVisualShape(self.sim.GEOM_SPHERE, radius=radius, rgbaColor=(1,0,0,1))

        linkMasses = [1]
        linkCollisionShapeIds = [cartCollisionShape]
        linkVisualShapeIds = [cartVisualShape]

        linkPositions = [[0, 0, 0]]
        linkOrientations = [[0, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1]]

        parentIndices = [0]

        jointTypes = [self.sim.JOINT_PRISMATIC]
        jointAxis = [[1, 0, 0]]

        # for each new link
        if num_links > 0:
            linkMasses += [0.001, pole_mass] * num_links
            linkCollisionShapeIds += [sphereCollisionShape, poleCollisionShape] * num_links
            linkVisualShapeIds += [sphereVisualShape, poleVisualShape] * num_links
            if inverted_pole:
                linkPositions += [[0, 0, 0], [0, 0, -0.5]]
                linkPositions += [[0, 0, -0.5], [0, 0, -0.5]] * (num_links - 1)
            else:
                linkPositions += [[0, 0, 0], [0, 0, 0.5]]
                linkPositions += [[0, 0, 0.5], [0, 0, 0.5]] * (num_links - 1)
            linkOrientations += [[0, 0, 0, 1]] * 2 * num_links
            linkInertialFramePositions += [[0, 0, 0]] * 2 * num_links
            linkInertialFrameOrientations += [[0, 0, 0, 1]] * 2 * num_links
            parentIndices += range(1, 1 + 2 * num_links)
            jointTypes += [self.sim.JOINT_REVOLUTE, self.sim.JOINT_FIXED] * num_links
            jointAxis += [[0, 1, 0], [0, 1, 0]] * num_links

        # create the whole body
        self.id = self.sim.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionShape,
                                           baseVisualShapeIndex=visualShape, basePosition=position,
                                           baseOrientation=orientation,
                                           baseInertialFramePosition=[0, 0, 0],
                                           baseInertialFrameOrientation=[0, 0, 0, 1],
                                           linkMasses=linkMasses,
                                           linkCollisionShapeIndices=linkCollisionShapeIds,
                                           linkVisualShapeIndices=linkVisualShapeIds,
                                           linkPositions=linkPositions, linkOrientations=linkOrientations,
                                           linkInertialFramePositions=linkInertialFramePositions,
                                           linkInertialFrameOrientations=linkInertialFrameOrientations,
                                           linkParentIndices=parentIndices, linkJointTypes=jointTypes,
                                           linkJointAxis=jointAxis)

        # useful variables
        self.joints = []  # non-fixed joint/link indices in the simulator
        self.joint_names = {}  # joint name to id in the simulator
        self.link_names = {}  # link name to id in the simulator
        for joint in range(self.getNumberOfJoints()):
            # Get joint info
            jnt = self.sim.getJointInfo(self.id, joint)
            self.joint_names[jnt[1]] = jnt[0]
            self.link_names[jnt[12]] = jnt[0]
            # remember actuated joints
            if jnt[2] != self.sim.JOINT_FIXED:
                self.joints.append(jnt[0])

        # disable the joints for the pole links
        # self.disableMotor(self.joints[1:])
        self.disableMotor(parentIndices[1::2])

    def getSymbolicEquationsOfMotion(self, verbose=False):
        """
        This returns the symbolic equation of motions of the robot (using the URDF). Internally, this used the
        `sympy.mechanics` module.
        """
        # gravity and time
        g, t = sympy.symbols('g t')

        # create the world inertial frame of reference and its origin
        worldFrame = mechanics.ReferenceFrame('Fw')
        worldOrigin = mechanics.Point('Pw')
        worldOrigin.set_vel(worldFrame, mechanics.Vector(0))

        # create the base frame (its position, orientation and velocities) + generalized coordinates and speeds
        baseId = -1

        # Check if the robot has a fixed base and create the generalized coordinates and speeds based on that,
        # as well the base position, orientation and velocities
        if robot.hasFixedBase():
            # generalized coordinates q(t) and speeds dq(t)
            q = mechanics.dynamicsymbols('q:{}'.format(len(self.joints)))
            dq = mechanics.dynamicsymbols('dq:{}'.format(len(self.joints)))
            pos, orn = robot.getBasePositionAndOrientation(convert_to_numpy_quaternion=False)
            linVel, angVel = [0,0,0], [0,0,0]   # 0 because fixed base
            jointId = 0
        else:
            # generalized coordinates q(t) and speeds dq(t)
            q = mechanics.dynamicsymbols('q:{}'.format(7 + len(self.joints)))
            dq = mechanics.dynamicsymbols('dq:{}'.format(6 + len(self.joints)))
            pos, orn = q[:3], q[3:7]
            linVel, angVel = dq[:3], dq[3:6]
            jointId = 7

        # set the position, orientation and velocities of the base
        baseFrame = worldFrame.orientnew('Fb', 'Quaternion', [orn[3], orn[0], orn[1], orn[2]])
        baseFrame.set_ang_vel(worldFrame, angVel[0] * worldFrame.x + angVel[1] * worldFrame.y + angVel[2] * worldFrame.z)
        baseOrigin = worldOrigin.locatenew('Pb', pos[0] * worldFrame.x + pos[1] * worldFrame.y + pos[2] * worldFrame.z)
        baseOrigin.set_vel(worldFrame, linVel[0] * worldFrame.x + linVel[1] * worldFrame.y + linVel[2] * worldFrame.z)

        # inputs u(t) (applied torques)
        u = mechanics.dynamicsymbols('u:{}'.format(len(self.joints)))
        jointIdU = 0

        # kinematics differential equations
        kd_eqs = [q[i].diff(t) - dq[i] for i in range(len(self.joints))]

        # define useful lists/dicts for later
        bodies, loads = [], []
        frames = {baseId: (baseFrame, baseOrigin)}
        # frames = {baseId: (worldFrame, worldOrigin)}

        # go through each joint/link (each link is associated to a joint)
        for linkId in range(self.getNumberOfLinks()):

            # get useful information about joint/link kinematics and dynamics from simulator
            info = self.sim.getDynamicsInfo(self.id, linkId)
            mass, localInertiaDiagonal = info[0], np.array(info[2])
            info = self.sim.getLinkState(self.id, linkId)
            localInertialFramePosition, localInertialFrameOrientation = info[2], info[3]
            # worldLinkFramePosition, worldLinkFrameOrientation = info[4], info[5]
            info = self.sim.getJointInfo(self.id, linkId)
            jointName, jointType = info[1:3]
            # jointDamping, jointFriction = info[6:8]
            linkName, jointAxisInLocalFrame, parentFramePosition, parentFrameOrientation, parentIdx = info[-5:]
            xl, yl, zl = jointAxisInLocalFrame

            # get previous references
            parentFrame, parentPoint = frames[parentIdx]

            # create a reference frame with its origin for each joint
            # set frame orientation
            if jointType == self.sim.JOINT_REVOLUTE:
                R = np.array(self.sim.getMatrixFromQuaternion(parentFrameOrientation)).reshape(3,3)
                R1 = getSymbolicMatrixFromAxisAngle(jointAxisInLocalFrame, q[jointId])
                R = R1.dot(R)
                frame = parentFrame.orientnew('F' + str(linkId), 'DCM', sympy.Matrix(R))
            else:
                x, y, z, w = parentFrameOrientation  # orientation of the joint in parent CoM inertial frame
                frame = parentFrame.orientnew('F' + str(linkId), 'Quaternion', [w, x, y, z])

            # set frame angular velocity
            angVel = 0
            if jointType == self.sim.JOINT_REVOLUTE:
                angVel = dq[jointId] * (xl * frame.x + yl * frame.y + zl * frame.z)
            frame.set_ang_vel(parentFrame, angVel)

            # create origin of the reference frame
            # set origin position
            x, y, z = parentFramePosition  # position of the joint in parent CoM inertial frame
            pos = x * parentFrame.x + y * parentFrame.y + z * parentFrame.z
            if jointType == self.sim.JOINT_PRISMATIC:
                pos += q[jointId] * (xl * frame.x + yl * frame.y + zl * frame.z)
            origin = parentPoint.locatenew('P' + str(linkId), pos)

            # set origin velocity
            if jointType == self.sim.JOINT_PRISMATIC:
                vel = dq[jointId] * (xl * frame.x + yl * frame.y + zl * frame.z)
                origin.set_vel(worldFrame, vel.express(worldFrame))
            else:
                origin.v2pt_theory(parentPoint, worldFrame, parentFrame)

            # define CoM frame and position (and velocities) wrt the local link frame
            x, y, z, w = localInertialFrameOrientation
            com_frame = frame.orientnew('Fc' + str(linkId), 'Quaternion', [w, x, y, z])
            com_frame.set_ang_vel(frame, mechanics.Vector(0))
            x, y, z = localInertialFramePosition
            com = origin.locatenew('C' + str(linkId), x * frame.x + y * frame.y + z * frame.z)
            com.v2pt_theory(origin, worldFrame, frame)

            # define com particle
            # com_particle = mechanics.Particle('Pa' + str(linkId), com, mass)
            # bodies.append(com_particle)

            # save
            # frames[linkId] = (frame, origin)
            # frames[linkId] = (frame, origin, com_frame, com)
            frames[linkId] = (com_frame, com)

            # define mass and inertia
            ixx, iyy, izz = localInertiaDiagonal
            inertia = mechanics.inertia(com_frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0)
            inertia = (inertia, com)

            # define rigid body associated to frame
            body = mechanics.RigidBody(linkName, com, frame, mass, inertia)
            bodies.append(body)

            # define dynamical forces/torques acting on the body
            # gravity force applied on the CoM
            force = (com, - mass * g * worldFrame.z)
            loads.append(force)

            # if prismatic joint, compute force
            if jointType == self.sim.JOINT_PRISMATIC:
                force = (origin, u[jointIdU] * (xl * frame.x + yl * frame.y + zl * frame.z))
                # force = (com, u[jointIdU] * (x * frame.x + y * frame.y + z * frame.z) - mass * g * worldFrame.z)
                loads.append(force)

            # if revolute joint, compute torque
            if jointType == self.sim.JOINT_REVOLUTE:
                v = (xl * frame.x + yl * frame.y + zl * frame.z)
                # torqueOnPrevBody = (parentFrame, - u[jointIdU] * v)
                torqueOnPrevBody = (parentFrame, - u[jointIdU] * v)
                torqueOnCurrBody = (frame, u[jointIdU] * v)
                loads.append(torqueOnPrevBody)
                loads.append(torqueOnCurrBody)

            # if joint is not fixed increment the current joint id
            if jointType != self.sim.JOINT_FIXED:
                jointId += 1
                jointIdU += 1

            if verbose:
                print("\nLink name with type: {} - {}".format(linkName, self.getJointTypes(jointId=linkId)))
                print("------------------------------------------------------")
                print("Position of joint frame wrt parent frame: {}".format(origin.pos_from(parentPoint)))
                print("Orientation of joint frame wrt parent frame: {}".format(frame.dcm(parentFrame)))
                print("Linear velocity of joint frame wrt parent frame: {}".format(origin.vel(worldFrame).express(parentFrame)))
                print("Angular velocity of joint frame wrt parent frame: {}".format(frame.ang_vel_in(parentFrame)))
                print("------------------------------------------------------")
                print("Position of joint frame wrt world frame: {}".format(origin.pos_from(worldOrigin)))
                print("Orientation of joint frame wrt world frame: {}".format(frame.dcm(worldFrame).simplify()))
                print("Linear velocity of joint frame wrt world frame: {}".format(origin.vel(worldFrame)))
                print("Angular velocity of joint frame wrt parent frame: {}".format(frame.ang_vel_in(worldFrame)))
                print("------------------------------------------------------")
                # print("Local position of CoM wrt joint frame: {}".format(com.pos_from(origin)))
                # print("Local linear velocity of CoM wrt joint frame: {}".format(com.vel(worldFrame).express(frame)))
                # print("Local angular velocity of CoM wrt joint frame: {}".format(com_frame.ang_vel_in(frame)))
                # print("------------------------------------------------------")
                if jointType == self.sim.JOINT_PRISMATIC:
                    print("Input value (force): {}".format(loads[-1]))
                elif jointType == self.sim.JOINT_REVOLUTE:
                    print("Input value (torque on previous and current bodies): {} and {}".format(loads[-2], loads[-1]))
                print("")

        if verbose:
            print("Summary:")
            print("Generalized coordinates: {}".format(q))
            print("Generalized speeds: {}".format(dq))
            print("Inputs: {}".format(u))
            print("Kinematic differential equations: {}".format(kd_eqs))
            print("Bodies: {}".format(bodies))
            print("Loads: {}".format(loads))
            print("")

        # TODO: 1. account for external forces applied on different rigid-bodies (e.g. contact forces)
        # TODO: 2. account for constraints (e.g. holonomic, non-holonomic, etc.)

        # Get the Equation of Motion (EoM) using Kane's method
        kane = mechanics.KanesMethod(worldFrame, q_ind=q, u_ind=dq, kd_eqs=kd_eqs)
        kane.kanes_equations(bodies=bodies, loads=loads)

        # get mass matrix and force vector (after simplifying) such that :math:`M(x,t) \dot{x} = f(x,t)`
        M = sympy.trigsimp(kane.mass_matrix_full)
        f = sympy.trigsimp(kane.forcing_full)
        # mechanics.find_dynamicsymbols(M)
        # mechanics.find_dynamicsymbols(f)

        # save useful info for future use (by other methods)
        constants = [g]
        constant_values = [9.81]
        parameters = (dict(zip(constants, constant_values)))
        self.symbols = {'q': q, 'dq': dq, 'kane': kane, 'parameters': parameters}

        # linearize
        # parameters = dict(zip(constants, constant_values))
        # M_, A_, B_, u_ = kane.linearize()
        # A_ = A_.subs(parameters)
        # B_ = B_.subs(parameters)
        # M_ = kane.mass_matrix_full.subs(parameters)

        # self.symbols = {'A_': A_, 'B_': B_, 'M_': M_}
        # return M_, A_, B_, u_

        return M, f

    def linearizeEquationOfMotion(self, point=None, verbose=False):
        """
        Linearize the equation of motions around the given point (=state). That is, instead of having
        :math:`\dot{x} = f(x,u)` where :math:`f` is in general a non-linear function, it linearizes it around
        a certain point.

        .. math:: \dot{x} = A x + B u

        where :math:`x` is the state vector, :math:`u` is the control input vector, and :math:`A` and :math:`B` are
        the matrices.
        """
        if self.symbols is None:
            self.getSymbolicEquationsOfMotion()

        if point is None:  # take current state
            point = list(self.getJointPositions()) + list(self.getJointVelocities())
        point = dict(zip(self.symbols['q'] + self.symbols['dq'], point))

        kane, parameters = self.symbols['kane'], self.symbols['parameters']

        # linearizer = self.symbols['kane'].to_linearizer()
        # A, B = linearizer.linearize(op_point=[point, parameters], A_and_B=True)

        M_, A_, B_, u_ = kane.linearize()
        fA = A_.subs(parameters).subs(point)
        fB = B_.subs(parameters).subs(point)
        M = kane.mass_matrix_full.subs(parameters).subs(point)

        # compute A and B
        Minv = M.inv()
        A = np.array(Minv * fA).astype(np.float64)
        B = np.array(Minv * fB).astype(np.float64)

        if verbose:
            print("M_ = {}".format(M_))
            print("A_ = {}".format(A_))
            print("B_ = {}".format(B_))
            print("u_ = {}".format(u_))
            print("fA = {}".format(fA))
            print("fB = {}".format(fB))
            print("M = {}".format(M))
            print("inv(M) = {}".format(Minv))

        return A, B


import control
from scipy.linalg import solve_continuous_are


class LQR(object):
    r"""Linear Quadratic Regulator

    Type: Model-based (optimal control)

    LQR assumes that the dynamics are described by a set of linear differential equations, and a quadratic cost.
    That is, the dynamics can written as :math:`\dot{x} = A x + B u`, where :math:`x` is the state vector, and
    :math:`u` is the control vector, and the cost is given by:

    .. math:: J = x(T)^T F(T) x(T) + \int_0^T (x(t)^T Q x(t) + u(t)^T R u(t) + 2 x(t)^T N u(t)) dt

    where :math:`Q` and :math:`R` represents weight matrices which allows to specify the relative importance
    of each state/control variable. These are normally set by the user.

    The goal is to find the feedback control law :math:`u` that minimizes the above cost :math:`J`. Solving it
    gives us :math:`u = -K x`, where :math:`K = R^{-1} (B^T S + N^T)` with :math:`S` is found by solving the
    continuous time Riccati differential equation :math:`S A + A^T S - (S B + N) R^{-1} (B^T S + N^T) + Q = 0`.

    Thus, LQR requires thus the model/dynamics of the system to be given (i.e. :math:`A` and :math:`B`).
    If the dynamical system is described by a set of nonlinear differential equations, we first have to linearize
    them around fixed points.

    Time complexity: O(M^3) where M is the size of the state vector
    Note: A desired state xd can also be given to the system: u = -K (x - xd)   (P control)

    See also:
        - `ilqr.py`: iterative LQR
        - `lqg.py`: LQG = LQR + LQE
        - `ilqg.py`: iterative LQG
    """

    def __init__(self, A, B, Q=None, R=None, N=None):
        if not self.isControllable(A, B):
            raise ValueError("The system is not controllable")
        self.A = A
        self.B = B
        if Q is None: Q = np.identity(A.shape[1])
        self.Q = Q
        if R is None: R = np.identity(B.shape[1])
        self.R = R
        self.N = N
        self.K = None

    @staticmethod
    def isControllable(A, B):
        return np.linalg.matrix_rank(control.ctrb(A, B)) == A.shape[0]

    def getRiccatiSolution(self):
        S = solve_continuous_are(self.A, self.B, self.Q, self.R, s=self.N)
        return S

    def getGainK(self):
        #S = self.getRiccatiSolution()
        #S1 = self.B.T.dot(S)
        #if self.N is not None: S1 += self.N.T
        #K = np.linalg.inv(self.R).dot(S1)

        if self.N is None:
            K, S, E = control.lqr(self.A, self.B, self.Q, self.R)
        else:
            K, S, E = control.lqr(self.A, self.B, self.Q, self.R, self.N)
        return K

    def compute(self, x, xd=None):
        """Return the u."""

        if self.K is None:
            self.K = self.getGainK()

        if xd is None:
            return self.K.dot(x)
        else:
            return self.K.dot(xd - x)


# Test
if __name__ == "__main__":
    import numpy as np
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import World

    # Create simulator
    sim = BulletSim()

    # create world
    world = World(sim)

    # create robot
    num_links = 1
    robot = CartPole(sim, num_links=num_links)

    # print information about the robot
    robot.printRobotInfo()

    robot.getSymbolicEquationsOfMotion()

    eq_point = np.zeros((num_links + 1) * 2)  # state = [q, dq]
    A, B = robot.linearizeEquationOfMotion(eq_point)

    # LQR controller
    lqr = LQR(A, B)
    K = lqr.getGainK()

    for i in count():
        # control
        x = np.concatenate((robot.getJointPositions(), robot.getJointVelocities()))
        u = K.dot(eq_point - x)
        robot.setJointTorques(u[0], 0)

        print("U[0] = {}".format(u[0]))

        # step in simulation
        world.step(sleep_dt=1./240)
