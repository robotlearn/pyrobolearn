#!/usr/bin/env python
"""Provide the Cartpole robotic platform.
"""

import os
import numpy as np
import pybullet_data
import sympy
import sympy.physics.mechanics as mechanics

from pyrobolearn.robots.robot import Robot
from pyrobolearn.utils.transformation import get_symbolic_matrix_from_axis_angle, get_matrix_from_quaternion

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


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
                 position=(0, 0, 0),
                 orientation=(0, 0, 0, 1),
                 scale=1.,
                 fixed_base=True,
                 urdf=os.path.join(pybullet_data.getDataPath(), "cartpole.urdf"),
                 num_links=1,
                 inverted_pole=False,
                 pole_mass=1):   # pole_mass=10
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(CartPole, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'cartpole'

        # create dynamically other links if necessary
        # Refs:
        # 1. https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12345
        # 2. https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/urdfEditor.py
        # if num_links > 1:
        #     # get info about the link (dimensions, mass, etc).
        #     jointInfo = self.get_joint_info(self.joints[-1])
        #     linkInfo = self.getLinkStates(self.joints[-1])
        #     dynamicInfo = self.sim.g
        #
        #     dims =
        #
        #     info =
        #     jointType = info[2]
        #     joint_axis, parent = info[-4]
        #     parentLinkIndex = info
        #
        #     # create visual, collision shapes, and the body
        #     collision_shape = self.sim.create_collision_shape(self.sim.GEOM_BOX, half_extents=dimensions)
        #     visual_shape = self.sim.create_visual_shape(self.sim.GEOM_BOX, half_extents=dimensions, rgba_color=color)
        #
        #     for i in range(num_links - 1):
        #         # create new link and attached it to the previous link
        #         linkId = self.sim.create_body(baseMass=mass,
        #                                           base_collision_shapeIndex=collision_shape,
        #                                           base_visual_shapeIndex=visual_shape,
        #                                           position=position,
        #                                           baseOrientation=orientation,
        #                                           linkParentIndices=[self.joints[-1]],
        #                                           linkJointTypes=[self.sim.JOINT_REVOLUTE])

        # create dynamically the cartpole because currently we can not add a link with a revolute joint in PyBullet;
        # we have to build the whole multibody system
        # The values are from the cartpole URDF: https://github.com/bulletphysics/bullet3/blob/master/data/cartpole.urdf

        # remove body
        self.sim.remove_body(self.id)

        # create slider
        dims = (15, 0.025, 0.025)
        color = (0, 0.8, 0.8, 1)
        mass = 0
        position = (0, 0, 0)
        orientation = (0, 0, 0, 1)
        collision_shape = self.sim.create_collision_shape(self.sim.GEOM_BOX, half_extents=dims)
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_BOX, half_extents=dims, rgba_color=color)

        # create cart and pole
        cart_dims = (0.25, 0.25, 0.1)
        cart_collision_shape = self.sim.create_collision_shape(self.sim.GEOM_BOX, half_extents=cart_dims)
        cart_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_BOX, half_extents=cart_dims,
                                                         rgba_color=(0, 0, 0.8, 1))
        pole_dims = (0.025, 0.025, 0.5)
        pole_collision_shape = self.sim.create_collision_shape(self.sim.GEOM_BOX, half_extents=pole_dims)
        pole_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_BOX, half_extents=pole_dims,
                                                         rgba_color=(1, 1, 1, 1))
        radius = 0.05
        sphere_collision_shape = self.sim.create_collision_shape(self.sim.GEOM_SPHERE, radius=radius)
        sphere_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius,
                                                           rgba_color=(1, 0, 0, 1))

        link_masses = [1]
        link_collision_shape_ids = [cart_collision_shape]
        link_visual_shape_ids = [cart_visual_shape]

        link_positions = [[0, 0, 0]]
        link_orientations = [[0, 0, 0, 1]]
        link_inertial_frame_positions = [[0, 0, 0]]
        link_inertial_frame_orientations = [[0, 0, 0, 1]]

        parent_indices = [0]

        joint_types = [self.sim.JOINT_PRISMATIC]
        joint_axis = [[1, 0, 0]]

        # for each new link
        if num_links > 0:
            link_masses += [0.001, pole_mass] * num_links
            link_collision_shape_ids += [sphere_collision_shape, pole_collision_shape] * num_links
            link_visual_shape_ids += [sphere_visual_shape, pole_visual_shape] * num_links
            if inverted_pole:
                link_positions += [[0, 0, 0], [0, 0, -0.5]]
                link_positions += [[0, 0, -0.5], [0, 0, -0.5]] * (num_links - 1)
            else:
                link_positions += [[0, 0, 0], [0, 0, 0.5]]
                link_positions += [[0, 0, 0.5], [0, 0, 0.5]] * (num_links - 1)
            link_orientations += [[0, 0, 0, 1]] * 2 * num_links
            link_inertial_frame_positions += [[0, 0, 0]] * 2 * num_links
            link_inertial_frame_orientations += [[0, 0, 0, 1]] * 2 * num_links
            parent_indices += range(1, 1 + 2 * num_links)
            joint_types += [self.sim.JOINT_REVOLUTE, self.sim.JOINT_FIXED] * num_links
            joint_axis += [[0, 1, 0], [0, 1, 0]] * num_links

        # create the whole body
        self.id = self.sim.create_body(mass=mass, collision_shape_id=collision_shape, visual_shape_id=visual_shape,
                                       position=position, orientation=orientation, baseInertialFramePosition=[0, 0, 0],
                                       baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=link_masses,
                                       linkCollisionShapeIndices=link_collision_shape_ids,
                                       linkVisualShapeIndices=link_visual_shape_ids,
                                       linkPositions=link_positions, linkOrientations=link_orientations,
                                       linkInertialFramePositions=link_inertial_frame_positions,
                                       linkInertialFrameOrientations=link_inertial_frame_orientations,
                                       linkParentIndices=parent_indices, linkJointTypes=joint_types,
                                       linkJointAxis=joint_axis)

        # useful variables
        self.joints = []  # non-fixed joint/link indices in the simulator
        self.joint_names = {}  # joint name to id in the simulator
        self.link_names = {}  # link name to id in the simulator
        for joint in range(self.num_joints):
            # Get joint info
            jnt = self.sim.get_joint_info(self.id, joint)
            self.joint_names[jnt[1]] = jnt[0]
            self.link_names[jnt[12]] = jnt[0]
            # remember actuated joints
            if jnt[2] != self.sim.JOINT_FIXED:
                self.joints.append(jnt[0])

        # disable the joints for the pole links
        # self.disable_motor(self.joints[1:])
        self.disable_motor(parent_indices[1::2])

    def get_symbolic_equations_of_motion(self, verbose=False):
        """
        This returns the symbolic equation of motions of the robot (using the URDF). Internally, this used the
        `sympy.mechanics` module.
        """
        # gravity and time
        g, t = sympy.symbols('g t')

        # create the world inertial frame of reference and its origin
        world_frame = mechanics.ReferenceFrame('Fw')
        world_origin = mechanics.Point('Pw')
        world_origin.set_vel(world_frame, mechanics.Vector(0))

        # create the base frame (its position, orientation and velocities) + generalized coordinates and speeds
        base_id = -1

        # Check if the robot has a fixed base and create the generalized coordinates and speeds based on that,
        # as well the base position, orientation and velocities
        if self.has_fixed_base():
            # generalized coordinates q(t) and speeds dq(t)
            q = mechanics.dynamicsymbols('q:{}'.format(len(self.joints)))
            dq = mechanics.dynamicsymbols('dq:{}'.format(len(self.joints)))
            pos, orn = self.get_base_pose()
            lin_vel, ang_vel = [0,0,0], [0,0,0]   # 0 because fixed base
            joint_id = 0
        else:
            # generalized coordinates q(t) and speeds dq(t)
            q = mechanics.dynamicsymbols('q:{}'.format(7 + len(self.joints)))
            dq = mechanics.dynamicsymbols('dq:{}'.format(6 + len(self.joints)))
            pos, orn = q[:3], q[3:7]
            lin_vel, ang_vel = dq[:3], dq[3:6]
            joint_id = 7

        # set the position, orientation and velocities of the base
        base_frame = world_frame.orientnew('Fb', 'Quaternion', [orn[3], orn[0], orn[1], orn[2]])
        base_frame.set_ang_vel(world_frame, ang_vel[0] * world_frame.x + ang_vel[1] * world_frame.y + ang_vel[2] *
                               world_frame.z)
        base_origin = world_origin.locatenew('Pb', pos[0] * world_frame.x + pos[1] * world_frame.y + pos[2] *
                                             world_frame.z)
        base_origin.set_vel(world_frame, lin_vel[0] * world_frame.x + lin_vel[1] * world_frame.y + lin_vel[2] *
                            world_frame.z)

        # inputs u(t) (applied torques)
        u = mechanics.dynamicsymbols('u:{}'.format(len(self.joints)))
        joint_id_u = 0

        # kinematics differential equations
        kd_eqs = [q[i].diff(t) - dq[i] for i in range(len(self.joints))]

        # define useful lists/dicts for later
        bodies, loads = [], []
        frames = {base_id: (base_frame, base_origin)}
        # frames = {base_id: (worldFrame, worldOrigin)}

        # go through each joint/link (each link is associated to a joint)
        for link_id in range(self.num_links):

            # get useful information about joint/link kinematics and dynamics from simulator
            info = self.sim.get_dynamics_info(self.id, link_id)
            mass, local_inertia_diagonal = info[0], info[2]
            info = self.sim.get_link_state(self.id, link_id)
            local_inertial_frame_position, local_inertial_frame_orientation = info[2], info[3]
            # worldLinkFramePosition, worldLinkFrameOrientation = info[4], info[5]
            info = self.sim.get_joint_info(self.id, link_id)
            joint_name, joint_type = info[1:3]
            # jointDamping, jointFriction = info[6:8]
            link_name, joint_axis_in_local_frame, parent_frame_position, parent_frame_orientation, \
                parent_idx = info[-5:]
            xl, yl, zl = joint_axis_in_local_frame

            # get previous references
            parent_frame, parent_point = frames[parent_idx]

            # create a reference frame with its origin for each joint
            # set frame orientation
            if joint_type == self.sim.JOINT_REVOLUTE:
                R = get_matrix_from_quaternion(parent_frame_orientation)
                R1 = get_symbolic_matrix_from_axis_angle(joint_axis_in_local_frame, q[joint_id])
                R = R1.dot(R)
                frame = parent_frame.orientnew('F' + str(link_id), 'DCM', sympy.Matrix(R))
            else:
                x, y, z, w = parent_frame_orientation  # orientation of the joint in parent CoM inertial frame
                frame = parent_frame.orientnew('F' + str(link_id), 'Quaternion', [w, x, y, z])

            # set frame angular velocity
            ang_vel = 0
            if joint_type == self.sim.JOINT_REVOLUTE:
                ang_vel = dq[joint_id] * (xl * frame.x + yl * frame.y + zl * frame.z)
            frame.set_ang_vel(parent_frame, ang_vel)

            # create origin of the reference frame
            # set origin position
            x, y, z = parent_frame_position  # position of the joint in parent CoM inertial frame
            pos = x * parent_frame.x + y * parent_frame.y + z * parent_frame.z
            if joint_type == self.sim.JOINT_PRISMATIC:
                pos += q[joint_id] * (xl * frame.x + yl * frame.y + zl * frame.z)
            origin = parent_point.locatenew('P' + str(link_id), pos)

            # set origin velocity
            if joint_type == self.sim.JOINT_PRISMATIC:
                vel = dq[joint_id] * (xl * frame.x + yl * frame.y + zl * frame.z)
                origin.set_vel(world_frame, vel.express(world_frame))
            else:
                origin.v2pt_theory(parent_point, world_frame, parent_frame)

            # define CoM frame and position (and velocities) wrt the local link frame
            x, y, z, w = local_inertial_frame_orientation
            com_frame = frame.orientnew('Fc' + str(link_id), 'Quaternion', [w, x, y, z])
            com_frame.set_ang_vel(frame, mechanics.Vector(0))
            x, y, z = local_inertial_frame_position
            com = origin.locatenew('C' + str(link_id), x * frame.x + y * frame.y + z * frame.z)
            com.v2pt_theory(origin, world_frame, frame)

            # define com particle
            # com_particle = mechanics.Particle('Pa' + str(linkId), com, mass)
            # bodies.append(com_particle)

            # save
            # frames[linkId] = (frame, origin)
            # frames[linkId] = (frame, origin, com_frame, com)
            frames[link_id] = (com_frame, com)

            # define mass and inertia
            ixx, iyy, izz = local_inertia_diagonal
            inertia = mechanics.inertia(com_frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0)
            inertia = (inertia, com)

            # define rigid body associated to frame
            body = mechanics.RigidBody(link_name, com, frame, mass, inertia)
            bodies.append(body)

            # define dynamical forces/torques acting on the body
            # gravity force applied on the CoM
            force = (com, - mass * g * world_frame.z)
            loads.append(force)

            # if prismatic joint, compute force
            if joint_type == self.sim.JOINT_PRISMATIC:
                force = (origin, u[joint_id_u] * (xl * frame.x + yl * frame.y + zl * frame.z))
                # force = (com, u[jointIdU] * (x * frame.x + y * frame.y + z * frame.z) - mass * g * worldFrame.z)
                loads.append(force)

            # if revolute joint, compute torque
            if joint_type == self.sim.JOINT_REVOLUTE:
                v = (xl * frame.x + yl * frame.y + zl * frame.z)
                # torqueOnPrevBody = (parentFrame, - u[jointIdU] * v)
                torque_on_prev_body = (parent_frame, - u[joint_id_u] * v)
                torque_on_curr_body = (frame, u[joint_id_u] * v)
                loads.append(torque_on_prev_body)
                loads.append(torque_on_curr_body)

            # if joint is not fixed increment the current joint id
            if joint_type != self.sim.JOINT_FIXED:
                joint_id += 1
                joint_id_u += 1

            if verbose:
                print("\nLink name with type: {} - {}".format(link_name, self.get_joint_types(joint_ids=link_id)))
                print("------------------------------------------------------")
                print("Position of joint frame wrt parent frame: {}".format(origin.pos_from(parent_point)))
                print("Orientation of joint frame wrt parent frame: {}".format(frame.dcm(parent_frame)))
                print("Linear velocity of joint frame wrt parent frame: {}".format(origin.vel(world_frame).express(parent_frame)))
                print("Angular velocity of joint frame wrt parent frame: {}".format(frame.ang_vel_in(parent_frame)))
                print("------------------------------------------------------")
                print("Position of joint frame wrt world frame: {}".format(origin.pos_from(world_origin)))
                print("Orientation of joint frame wrt world frame: {}".format(frame.dcm(world_frame).simplify()))
                print("Linear velocity of joint frame wrt world frame: {}".format(origin.vel(world_frame)))
                print("Angular velocity of joint frame wrt parent frame: {}".format(frame.ang_vel_in(world_frame)))
                print("------------------------------------------------------")
                # print("Local position of CoM wrt joint frame: {}".format(com.pos_from(origin)))
                # print("Local linear velocity of CoM wrt joint frame: {}".format(com.vel(worldFrame).express(frame)))
                # print("Local angular velocity of CoM wrt joint frame: {}".format(com_frame.ang_vel_in(frame)))
                # print("------------------------------------------------------")
                if joint_type == self.sim.JOINT_PRISMATIC:
                    print("Input value (force): {}".format(loads[-1]))
                elif joint_type == self.sim.JOINT_REVOLUTE:
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
        kane = mechanics.KanesMethod(world_frame, q_ind=q, u_ind=dq, kd_eqs=kd_eqs)
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

    def linearize_equations_of_motion(self, point=None, verbose=False):
        r"""
        Linearize the equation of motions around the given point (=state). That is, instead of having
        :math:`\dot{x} = f(x,u)` where :math:`f` is in general a non-linear function, it linearizes it around
        a certain point.

        .. math:: \dot{x} = A x + B u

        where :math:`x` is the state vector, :math:`u` is the control input vector, and :math:`A` and :math:`B` are
        the matrices.
        """
        if self.symbols is None:
            self.get_symbolic_equations_of_motion()

        if point is None:  # take current state
            point = list(self.get_joint_positions()) + list(self.get_joint_velocities())
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
        if not self.is_controllable(A, B):
            raise ValueError("The system is not controllable")
        self.A = A
        self.B = B
        if Q is None: 
            Q = np.identity(A.shape[1])
        self.Q = Q
        if R is None: 
            R = np.identity(B.shape[1])
        self.R = R
        self.N = N
        self.K = None

    @staticmethod
    def is_controllable(A, B):
        return np.linalg.matrix_rank(control.ctrb(A, B)) == A.shape[0]

    def get_riccati_solution(self):
        S = solve_continuous_are(self.A, self.B, self.Q, self.R, s=self.N)
        return S

    def get_gain_k(self):
        # S = self.get_riccati_solution()
        # S1 = self.B.T.dot(S)
        # if self.N is not None: S1 += self.N.T
        # K = np.linalg.inv(self.R).dot(S1)

        if self.N is None:
            K, S, E = control.lqr(self.A, self.B, self.Q, self.R)
        else:
            K, S, E = control.lqr(self.A, self.B, self.Q, self.R, self.N)
        return K

    def compute(self, x, xd=None):
        """Return the u."""

        if self.K is None:
            self.K = self.get_gain_k()

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
    robot.print_info()

    robot.get_symbolic_equations_of_motion()

    eq_point = np.zeros((num_links + 1) * 2)  # state = [q, dq]
    A, B = robot.linearize_equations_of_motion(eq_point)

    # LQR controller
    lqr = LQR(A, B)
    K = lqr.get_gain_k()

    for i in count():
        # control
        x = np.concatenate((robot.get_joint_positions(), robot.get_joint_velocities()))
        u = K.dot(eq_point - x)
        robot.set_joint_torques(u[0], 0)

        print("U[0] = {}".format(u[0]))

        # step in simulation
        world.step(sleep_dt=1./240)
