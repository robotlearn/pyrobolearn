#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the DART (Dynamic Animation and Robotics Toolkit) Simulator API.

This is the main interface that communicates with the DART simulator [1, 2]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
``dartpy``.

The signature of each method defined here are inspired by [1,2] but in accordance with the PEP8 style guide [3].
Parts of the documentation for the methods have been copied-pasted from [1] for completeness purposes.

Note that there are 2 python wrappers for DART [1]: `dartpy` (which uses `pybind11` to wrap C++ code) [1], and
`pydart` (which uses `SWIG` to wrap the C++ code) [2]. Currently, it seems that:
- `dartpy` seems to be pretty new (and doesn't have a lot of examples). It can be found on the official DART github
repo. Note that you can only use it in Python 3 (>=3.4).
- `pydart` has some examples but seems to not be under active development anymore (the last commit was on May 25, 2018).
It can be used with Python2.7 and Python3.5.
- Both have a very poor Python documentation.

We selected to use ``dartpy`` instead of ``pydart`` as this last one is no more under active development, and
``dartpy`` is the official release.

- Supported Python versions: Python 3.*
- Python wrappers: pybind11 [4]

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

Dependencies in PRL: None

References:
    - [1] DART: Dynamic Animation and Robotics Toolkit
        - paper: http://joss.theoj.org/papers/10.21105/joss.00500
        - webpage: https://dartsim.github.io/
        - github: https://github.com/dartsim/dart/
        - dartpy: http://dartsim.github.io/install_dartpy_on_ubuntu.html
    - [2] PyDART
        - source code: https://pydart2.readthedocs.io/en/latest/
        - documentation: https://pydart2.readthedocs.io/en/latest/
    - [3] PEP8: https://www.python.org/dev/peps/pep-0008/
    - [4] Pybind11: https://pybind11.readthedocs.io/en/stable/
        - Cython, pybind11, cffi – which tool should you choose?:
          http://blog.behnel.de/posts/cython-pybind11-cffi-which-tool-to-choose.html
"""

# import standard libraries
import os
import time
import numpy as np
from collections import OrderedDict

# import dartpy
try:
    import dartpy as dart
except ImportError as e:
    raise ImportError(e.__str__() + "\n: HINT: you can install `dartpy` by following the instructions given at: "
                                    "https://dartsim.github.io/install_dartpy_on_ubuntu.html")

# import PRL simulator
from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.utils.parsers.robots import URDFParser, MuJoCoParser, SDFParser, SkelParser
from pyrobolearn.utils.transformation import get_quaternion_from_matrix, get_rpy_from_quaternion

# check Python version
import sys
if sys.version_info[0] < 3:
    raise RuntimeError("You must use Python 3 with the Dart simulator.")

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["DART", "Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Dart(Simulator):
    r"""DART simulator

    This is a wrapper around the ``dartpy`` API [1] (which is itself a Python wrapper around DART [1]).

    With DART, we first have to create a world, then spawn a skeleton (a `Skeleton` is a structure that consists of
    `BodyNode` which are connected by `Joint`).

    Warnings: by default, in DART, the world frame axis are defined with x pointing forward, y pointing upward, and
    z pointing on the right. To be consistent with the other simulators, we change this to be x pointing forward,
    y pointing on the left, and z pointing upward.

    Notes:
        - In the documentation, Isometry refers to a homogeneous transformation matrix.

    Examples:
        sim = Dart()

    References:
        - [1] Dart:
            - webpage: https://dartsim.github.io/
            - github: https://github.com/dartsim/dart/
            - dartpy: http://dartsim.github.io/install_dartpy_on_ubuntu.html
        - [2] PEP8: https://www.python.org/dev/peps/pep-0008/
    """

    def __init__(self, render=True, num_instances=1, middleware=None, **kwargs):
        """
        Initialize the Dart simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server.
            num_instances (int): number of simulator instances.
            middleware (MiddleWare, None): middleware instance.
            **kwargs (dict): optional arguments (this is not used here).
        """
        super(Dart, self).__init__(render=render, num_instances=num_instances, middleware=middleware, **kwargs)

        # dart = {'collision': ['BulletCollisionDetector', 'BulletCollisionGroup', 'CollisionDetector',
        #                       'CollisionGroup', 'CollisionOption', 'CollisionResult', 'Contact',
        #                       'DARTCollisionDetector', 'DARTCollisionGroup', 'DistanceOption', 'DistanceResult',
        #                       'FCLCollisionDetector', 'FCLCollisionGroup', 'OdeCollisionDetector',
        #                       'OdeCollisionGroup', 'RayHit', 'RaycastOption', 'RaycastResult'],
        #         'common': ['Composite', 'Observer', 'Subject', 'Uri'],
        #         'constraint': ['BallJointConstraint', 'BoxedLcpConstraintSolver', 'BoxedLcpSolver', 'ConstraintBase',
        #                        'ConstraintSolver', 'DantzigBoxedLcpSolver', 'JointConstraint',
        #                        'JointCoulombFrictionConstraint', 'JointLimitConstraint', 'PgsBoxedLcpSolver',
        #                        'PgsBoxedLcpSolverOption', 'WeldJointConstraint'],
        #         'dynamics': ['ArrowShape', 'ArrowShapeProperties', 'BallJoint', 'BallJointProperties', 'BodyNode',
        #                      'BodyNodeAspectProperties', 'BodyNodeProperties', 'BoxShape', 'CapsuleShape', 'Chain',
        #                      'ChainCriteria', 'CollisionAspect',
        #                      'CompositeJoiner_EmbedProperties_EulerJoint_EulerJointUniqueProperties_GenericJoint_R3Space',
        #                      'CompositeJoiner_EmbedProperties_PlanarJoint_PlanarJointUniqueProperties_GenericJoint_R3Space',
        #                      'CompositeJoiner_EmbedProperties_PrismaticJoint_PrismaticJointUniqueProperties_GenericJoint_R1Space',
        #                      'CompositeJoiner_EmbedProperties_RevoluteJoint_RevoluteJointUniqueProperties_GenericJoint_R1Space',
        #                      'CompositeJoiner_EmbedProperties_ScrewJoint_ScrewJointUniqueProperties_GenericJoint_R1Space',
        #                      'CompositeJoiner_EmbedProperties_TranslationalJoint2D_TranslationalJoint2DUniqueProperties_GenericJoint_R2Space',
        #                      'CompositeJoiner_EmbedProperties_UniversalJoint_UniversalJointUniqueProperties_GenericJoint_R2Space',
        #                      'CompositeJoiner_EmbedStateAndProperties_GenericJoint_R1GenericJointStateGenericJointUniqueProperties_Joint',
        #                      'CompositeJoiner_EmbedStateAndProperties_GenericJoint_R2GenericJointStateGenericJointUniqueProperties_Joint',
        #                      'CompositeJoiner_EmbedStateAndProperties_GenericJoint_R3GenericJointStateGenericJointUniqueProperties_Joint',
        #                      'CompositeJoiner_EmbedStateAndProperties_GenericJoint_SE3GenericJointStateGenericJointUniqueProperties_Joint',
        #                      'CompositeJoiner_EmbedStateAndProperties_GenericJoint_SO3GenericJointStateGenericJointUniqueProperties_Joint',
        #                      'ConeShape', 'CylinderShape', 'DegreeOfFreedom', 'Detachable', 'DynamicsAspect',
        #                      'EllipsoidShape',
        #                      'EmbedPropertiesOnTopOf_EulerJoint_EulerJointUniqueProperties_GenericJoint_R3Space',
        #                      'EmbedPropertiesOnTopOf_PlanarJoint_PlanarJointUniqueProperties_GenericJoint_R3Space',
        #                      'EmbedPropertiesOnTopOf_PrismaticJoint_PrismaticJointUniqueProperties_GenericJoint_R1Space',
        #                      'EmbedPropertiesOnTopOf_RevoluteJoint_RevoluteJointUniqueProperties_GenericJoint_R1Space',
        #                      'EmbedPropertiesOnTopOf_ScrewJoint_ScrewJointUniqueProperties_GenericJoint_R1Space',
        #                      'EmbedPropertiesOnTopOf_TranslationalJoint2D_TranslationalJoint2DUniqueProperties_GenericJoint_R2Space',
        #                      'EmbedPropertiesOnTopOf_UniversalJoint_UniversalJointUniqueProperties_GenericJoint_R2Space',
        #                      'EmbedProperties_EulerJoint_EulerJointUniqueProperties',
        #                      'EmbedProperties_Joint_JointProperties',
        #                      'EmbedProperties_PlanarJoint_PlanarJointUniqueProperties',
        #                      'EmbedProperties_PrismaticJoint_PrismaticJointUniqueProperties',
        #                      'EmbedProperties_RevoluteJoint_RevoluteJointUniqueProperties',
        #                      'EmbedProperties_ScrewJoint_ScrewJointUniqueProperties',
        #                      'EmbedProperties_TranslationalJoint2D_TranslationalJoint2DUniqueProperties',
        #                      'EmbedProperties_UniversalJoint_UniversalJointUniqueProperties',
        #                      'EmbedStateAndPropertiesOnTopOf_GenericJoint_R1_GenericJointState_GenericJointUniqueProperties_Joint',
        #                      'EmbedStateAndPropertiesOnTopOf_GenericJoint_R2_GenericJointState_GenericJointUniqueProperties_Joint',
        #                      'EmbedStateAndPropertiesOnTopOf_GenericJoint_R3_GenericJointState_GenericJointUniqueProperties_Joint',
        #                      'EmbedStateAndPropertiesOnTopOf_GenericJoint_SE3_GenericJointState_GenericJointUniqueProperties_Joint',
        #                      'EmbedStateAndPropertiesOnTopOf_GenericJoint_SO3_GenericJointState_GenericJointUniqueProperties_Joint',
        #                      'EmbedStateAndProperties_GenericJoint_R1GenericJointState_GenericJointUniqueProperties',
        #                      'EmbedStateAndProperties_GenericJoint_R2GenericJointState_GenericJointUniqueProperties',
        #                      'EmbedStateAndProperties_GenericJoint_R3GenericJointState_GenericJointUniqueProperties',
        #                      'EmbedStateAndProperties_GenericJoint_SE3GenericJointState_GenericJointUniqueProperties',
        #                      'EmbedStateAndProperties_GenericJoint_SO3GenericJointState_GenericJointUniqueProperties',
        #                      'Entity', 'EulerJoint', 'EulerJointProperties', 'EulerJointUniqueProperties', 'Frame',
        #                      'FreeJoint', 'FreeJointProperties', 'GenericJointProperties_R1',
        #                      'GenericJointProperties_R2', 'GenericJointProperties_R3', 'GenericJointProperties_SE3',
        #                      'GenericJointProperties_SO3', 'GenericJointUniqueProperties_R1',
        #                      'GenericJointUniqueProperties_R2', 'GenericJointUniqueProperties_R3',
        #                      'GenericJointUniqueProperties_SE3', 'GenericJointUniqueProperties_SO3',
        #                      'GenericJoint_R1', 'GenericJoint_R2', 'GenericJoint_R3', 'GenericJoint_SE3',
        #                      'GenericJoint_SO3', 'InverseKinematics', 'InverseKinematicsErrorMethod', 'JacobianNode',
        #                      'Joint', 'JointProperties', 'LineSegmentShape', 'Linkage', 'LinkageCriteria',
        #                      'MeshShape', 'MetaSkeleton', 'MultiSphereConvexHullShape', 'Node', 'PlanarJoint',
        #                      'PlanarJointProperties', 'PlanarJointUniqueProperties', 'PlaneShape', 'PrismaticJoint',
        #                      'PrismaticJointProperties', 'PrismaticJointUniqueProperties', 'ReferentialSkeleton',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_EulerJoint_EulerJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_Joint_JointProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_PlanarJoint_PlanarJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_PrismaticJoint_PrismaticJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_RevoluteJoint_RevoluteJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_ScrewJoint_ScrewJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_TranslationalJoint2D_TranslationalJoint2DUniqueProperties',
        #                      'RequiresAspect_EmbeddedPropertiesAspect_UniversalJoint_UniversalJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_R1_GenericJointState_GenericJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_R2_GenericJointState_GenericJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_R3_GenericJointState_GenericJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_SE3_GenericJointState_GenericJointUniqueProperties',
        #                      'RequiresAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_SO3_GenericJointState_GenericJointUniqueProperties',
        #                      'RevoluteJoint', 'RevoluteJointProperties', 'RevoluteJointUniqueProperties',
        #                      'ScrewJoint', 'ScrewJointProperties', 'ScrewJointUniqueProperties', 'Shape',
        #                      'ShapeFrame', 'ShapeNode', 'SimpleFrame', 'Skeleton', 'SoftMeshShape',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_EulerJoint_EulerJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_Joint_JointProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_PlanarJoint_PlanarJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_PrismaticJoint_PrismaticJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_RevoluteJoint_RevoluteJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_ScrewJoint_ScrewJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_TranslationalJoint2D_TranslationalJoint2DUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedPropertiesAspect_UniversalJoint_UniversalJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_R1_GenericJointState_GenericJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_R2_GenericJointState_GenericJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_R3_GenericJointState_GenericJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_SE3_GenericJointState_GenericJointUniqueProperties',
        #                      'SpecializedForAspect_EmbeddedStateAndPropertiesAspect_GenericJoint_SO3_GenericJointState_GenericJointUniqueProperties',
        #                      'SphereShape', 'TemplatedJacobianBodyNode', 'TranslationalJoint', 'TranslationalJoint2D',
        #                      'TranslationalJoint2DProperties', 'TranslationalJoint2DUniqueProperties',
        #                      'TranslationalJointProperties', 'UniversalJoint', 'UniversalJointProperties',
        #                      'UniversalJointUniqueProperties', 'VisualAspect', 'WeldJoint', 'ZeroDofJoint',
        #                      'ZeroDofJointProperties'],
        #         'gui': {'osg': ['BodyNodeDnD', 'DragAndDrop', 'GUIActionAdapter', 'GUIEventAdapter',
        #                         'GUIEventHandler', 'GridVisual', 'ImGuiHandler', 'ImGuiViewer', 'ImGuiWidget',
        #                         'InteractiveFrame', 'InteractiveFrameDnD', 'InteractiveTool', 'RealTimeWorldNode',
        #                         'ShadowMap', 'ShadowTechnique', 'SimpleFrameDnD', 'SimpleFrameShapeDnD', 'Viewer',
        #                         'ViewerAttachment', 'WorldNode']},
        #         'math': ['AngleAxis', 'Isometry3', 'Quaternion', 'Random', 'eulerXYXToMatrix', 'eulerXYZToMatrix',
        #                  'eulerXZXToMatrix', 'eulerXZYToMatrix', 'eulerYXYToMatrix', 'eulerYXZToMatrix',
        #                  'eulerYZXToMatrix', 'eulerYZYToMatrix', 'eulerZXYToMatrix', 'eulerZXZToMatrix',
        #                  'eulerZYXToMatrix', 'eulerZYZToMatrix', 'expAngular', 'expMap', 'expMapJac', 'expMapRot',
        #                  'expToQuat', 'matrixToEulerXYX', 'matrixToEulerXYZ', 'matrixToEulerXZY', 'matrixToEulerYXZ',
        #                  'matrixToEulerYZX', 'matrixToEulerZXY', 'matrixToEulerZYX', 'quatToExp', 'verifyRotation',
        #                  'verifyTransform'],
        #         'optimizer': ['Function', 'GradientDescentSolver', 'GradientDescentSolverProperties',
        #                       'GradientDescentSolverUniqueProperties', 'ModularFunction', 'MultiFunction',
        #                       'NloptSolver', 'NullFunction', 'Problem', 'Solver', 'SolverProperties'],
        #         'simulation': ['World'],
        #         'utils': ['DartLoader', 'SkelParser']]

        # Skeleton = ['checkIndexingConsistency', 'clearConstraintImpulses', 'clearExternalForces', 'clearIK',
        #             'clearInternalForces', 'clone', 'cloneMetaSkeleton', 'computeForwardDynamics',
        #             'computeForwardKinematics', 'computeImpulseForwardDynamics', 'computeInverseDynamics',
        #             'computeKineticEnergy', 'computeLagrangian', 'computePotentialEnergy',
        #             'createBallJointAndBodyNodePair', 'createEulerJointAndBodyNodePair',
        #             'createFreeJointAndBodyNodePair', 'createPlanarJointAndBodyNodePair',
        #             'createPrismaticJointAndBodyNodePair', 'createRevoluteJointAndBodyNodePair',
        #             'createScrewJointAndBodyNodePair', 'createTranslationalJoint2DAndBodyNodePair',
        #             'createTranslationalJointAndBodyNodePair', 'createUniversalJointAndBodyNodePair',
        #             'createWeldJointAndBodyNodePair', 'dirtyArticulatedInertia', 'dirtySupportPolygon',
        #             'disableAdjacentBodyCheck', 'disableSelfCollisionCheck', 'enableAdjacentBodyCheck',
        #             'enableSelfCollisionCheck', 'getAcceleration', 'getAccelerationLowerLimit',
        #             'getAccelerationLowerLimits', 'getAccelerationUpperLimit', 'getAccelerationUpperLimits',
        #             'getAccelerations', 'getAdjacentBodyCheck', 'getAngularJacobian', 'getAngularJacobianDeriv',
        #             'getAugMassMatrix', 'getBodyNode', 'getBodyNodes', 'getCOM', 'getCOMJacobian',
        #             'getCOMJacobianSpatialDeriv', 'getCOMLinearAcceleration', 'getCOMLinearJacobian',
        #             'getCOMLinearJacobianDeriv', 'getCOMLinearVelocity', 'getCOMSpatialAcceleration',
        #             'getCOMSpatialVelocity', 'getCommand', 'getCommands', 'getConfiguration', 'getConstraintForces',
        #             'getCoriolisAndGravityForces', 'getCoriolisForces', 'getDof', 'getDofs', 'getExternalForces',
        #             'getForce', 'getForceLowerLimit', 'getForceLowerLimits', 'getForceUpperLimit',
        #             'getForceUpperLimits', 'getForces', 'getGravity', 'getGravityForces', 'getIK', 'getIndexOf',
        #             'getInvMassMatrix', 'getJacobian', 'getJacobianClassicDeriv', 'getJacobianSpatialDeriv',
        #             'getJoint', 'getJointConstraintImpulses', 'getJoints', 'getLinearJacobian',
        #             'getLinearJacobianDeriv', 'getLockableReference', 'getMass', 'getMassMatrix', 'getName',
        #             'getNumBodyNodes', 'getNumDofs', 'getNumEndEffectors', 'getNumJoints', 'getNumMarkers',
        #             'getNumRigidBodyNodes', 'getNumShapeNodes', 'getNumSoftBodyNodes', 'getNumTrees', 'getPosition',
        #             'getPositionDifferences', 'getPositionLowerLimit', 'getPositionLowerLimits',
        #             'getPositionUpperLimit', 'getPositionUpperLimits', 'getPositions', 'getProperties', 'getPtr',
        #             'getRootBodyNode', 'getRootJoint', 'getSelfCollisionCheck', 'getSkeleton', 'getState',
        #             'getSupportVersion', 'getTimeStep', 'getTreeBodyNodes', 'getVelocities', 'getVelocity',
        #             'getVelocityChanges', 'getVelocityDifferences', 'getVelocityLowerLimit',
        #             'getVelocityLowerLimits', 'getVelocityUpperLimit', 'getVelocityUpperLimits', 'getWorldJacobian',
        #             'hasBodyNode', 'hasJoint', 'integratePositions', 'integrateVelocities',
        #             'isEnabledAdjacentBodyCheck', 'isEnabledSelfCollisionCheck', 'isImpulseApplied', 'isMobile',
        #             'mUnionIndex', 'mUnionRootSkeleton', 'mUnionSize', 'resetAccelerations', 'resetCommands',
        #             'resetGeneralizedForces', 'resetPositions', 'resetUnion', 'resetVelocities', 'setAcceleration',
        #             'setAccelerationLowerLimit', 'setAccelerationLowerLimits', 'setAccelerationUpperLimit',
        #             'setAccelerationUpperLimits', 'setAccelerations', 'setAdjacentBodyCheck', 'setAspectProperties',
        #             'setCommand', 'setCommands', 'setConfiguration', 'setForce', 'setForceLowerLimit',
        #             'setForceLowerLimits', 'setForceUpperLimit', 'setForceUpperLimits', 'setForces', 'setGravity',
        #             'setImpulseApplied', 'setJointConstraintImpulses', 'setMobile', 'setName', 'setPosition',
        #             'setPositionLowerLimit', 'setPositionLowerLimits', 'setPositionUpperLimit',
        #             'setPositionUpperLimits', 'setPositions', 'setProperties', 'setSelfCollisionCheck', 'setState',
        #             'setTimeStep', 'setVelocities', 'setVelocity', 'setVelocityLowerLimit', 'setVelocityLowerLimits',
        #             'setVelocityUpperLimit', 'setVelocityUpperLimits', 'updateBiasImpulse', 'updateVelocityChange']

        # create world
        self.sim = dart.simulation.World()
        self.world = self.sim

        # create urdf parser
        self._urdf_parser = dart.utils.DartLoader()

        # set time step
        self.dt = self.get_time_step()

        # main camera in the simulator
        self._camera = None
        self.viewer = None
        self._frame_cnt = 0      # frame counter
        self._frame_ticks = 15   # number of ticks to sleep before next acquisition

        # set the gravity
        self.set_gravity()

        # if we need to render
        if render:
            self.render()

        # keep track of visual and collision shapes
        self.visual_shapes = {}  # {visual_id: Visual}
        self.collision_shapes = {}  # {collision_id: Collision}
        self._bodies = OrderedDict()  # {body_id: Body}
        self.textures = {}  # {texture_id: Texture}
        self._constraints = OrderedDict()  # {constraint_id: Constraint}

        # create counters
        self._visual_cnt = 0
        self._collision_cnt = 0
        self._body_cnt = 0  # 0 is for the world
        self._texture_cnt = 0
        self._constraint_cnt = 0

        self._floor_id = None

    ##############
    # Properties #
    ##############

    @property
    def version(self):
        """Return the version of the simulator."""
        return "6.10.0"  # TODO: get it from the code

    @property
    def timestep(self):
        """Return the simulator time step."""
        return self.dt

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the simulator. This can be overridden in the child class."""
        return self.__class__(render=self._render, **self.kwargs)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the simulator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass.
        """
        # if the object has already been copied return the reference to the copied object
        if self in memo:
            return memo[self]

        # create a new copy of the simulator
        sim = self.__class__(render=self._render, **self.kwargs)

        memo[self] = sim
        return sim

    ##################
    # Static methods #
    ##################

    @staticmethod
    def supports_acceleration():
        """Return True if the simulator provides acceleration (dynamic) information (such as joint accelerations, link
        Cartesian accelerations, etc). If not, the `Robot` class will have to implement these using finite
        difference."""
        return True

    ###########
    # Methods #
    ###########

    ###########
    # Private #
    ###########

    @staticmethod
    def _convert_wxyz_to_xyzw(q):
        """Convert a quaternion in the (w,x,y,z) format to (x,y,z,w)."""
        return np.roll(q, shift=-1)

    @staticmethod
    def _convert_xyzw_to_wxyz(q):
        """Convert a quaternion in the (x,y,z,w) format to (w,x,y,z)."""
        return np.roll(q, shift=1)

    @staticmethod
    def _get_matrix_from_transform(transform):
        """Return the homogeneous matrix from the given transform.

        Args:
            transform (dartpy.math.Isometry3): transform.

        Returns:
            np.array[float[4,4]]: homogeneous matrix.
        """
        rot = transform.rotation()
        pos = transform.translation()
        return np.vstack((np.hstack((rot, pos.reshape(-1, 1))),
                          np.array([0., 0., 0., 1.])))

    @staticmethod
    def _get_pose_from_transform(transform):
        """Return the pose from the transform.

        Args:
            transform (dartpy.math.Isometry3): transform.

        Returns:
            np.array[float[3]]: position vector.
            np.array[float[4]]: quaternion (expressed as [x,y,z,w]).
        """
        quat = Dart._convert_wxyz_to_xyzw(transform.quaternion().wxyz())
        pos = transform.translation()
        return pos, quat

    @staticmethod
    def _get_quat_from_transform(transform):
        """Return the quaternion from the given transform.

        Args:
            transform (dartpy.math.Isometry3): transform.

        Returns:
            np.array[float[4]]: quaternion (expressed as [x,y,z,w]).
        """
        return Dart._convert_wxyz_to_xyzw(transform.quaternion().wxyz())

    @staticmethod
    def _get_pos_from_transform(transform):
        """Return the position from the given transform.

        Args:
            transform (dartpy.math.Isometry3): transform.

        Returns:
            np.array[float[3]]: position vector.
        """
        return transform.translation()

    ##############
    # Simulators #
    ##############

    def reset(self, *args, **kwargs):
        """Reset the simulator."""
        self.world.reset()

    def close(self):
        """Close the simulator."""
        del self.world

    def seed(self, seed=None):
        """Set the given seed in the simulator."""
        pass

    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified time.

        Args:
            sleep_time (float): time to sleep after performing one step in the simulation.
        """
        self.world.step()
        if self._render:
            if self._frame_cnt % self._frame_ticks == 0:
                self.viewer.frame()
                self._frame_cnt = 0
            self._frame_cnt += 1

    def is_rendering(self):
        """Return True if the simulator is in the render mode."""
        return self._render

    def reset_scene_camera(self, camera=None):
        """
        Reinitialize/Reset the scene view camera to the previous one.

        Args:
            camera (object): scene view camera. This is let to the user to decide what to do.
        """
        pass

    def render(self, enable=True):
        """Render the simulation.

        Args:
            enable (bool): If True, it will render the simulator by enabling the GUI.
        """
        self._render = enable
        if self._render:
            if self.viewer is None:
                self.viewer = dart.gui.osg.Viewer()
                gui_node = dart.gui.osg.WorldNode(self.world)
                self.viewer.addWorldNode(gui_node)
                self.viewer.setUpViewInWindow(0, 0, 1280, 960)
                self.viewer.setCameraHomePosition([8., -8., 4.], [0, 0, -0.25], [0, 0, 0.5])
                self._frame_cnt = 0
                # self.viewer.run()  # TODO: call self.viewer.frame() instead (need to update the python wrapper)
        else:
            # TODO: close the viewer before setting it to None
            del self.viewer
            self.viewer = None

    def hide(self):
        """Hide the GUI."""
        self.render(False)

    def get_time_step(self):
        """Get the time step in the simulator.

        Returns:
            float: time step in the simulator
        """
        return self.world.getTimeStep()

    def set_time_step(self, time_step):
        """Set the time step in the simulator.

        Args:
            time_step (float): Each time you call 'step' the time step will proceed with 'time_step'.
        """
        self.world.setTimeStep(time_step)

    def set_real_time(self, enable=True):
        """Enable real time in the simulator.

        Args:
            enable (bool): If True, it will enable the real-time simulation. If False, it will disable it.
        """
        # create real-time world node (dart.gui.osg.RealTimeWorldNode)
        pass  # TODO: use threads and create class that inherits from dart.gui.osg.RealTimeWorldNode

    def pause(self):
        """Pause the simulator if in real-time."""
        pass

    def unpause(self):
        """Unpause the simulator if in real-time."""
        pass

    def get_physics_properties(self):
        """Get the physics engine parameters."""
        pass

    def set_physics_properties(self, *args, **kwargs):
        """Set the physics engine parameters."""
        pass

    def start_logging(self, *args, **kwargs):
        """Start the logging."""
        pass

    def stop_logging(self, logger_id):
        """Stop the logging."""
        pass

    def get_gravity(self):
        """Return the gravity set in the simulator."""
        return self.world.getGravity()

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.

        Args:
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
        """
        self.world.setGravity(gravity)

    def save(self, filename=None, *args, **kwargs):
        """Save the state of the simulator.

        Args:
            filename (None, str): path to file to store the state of the simulator. If None, it will save it in
                memory instead of the disk.

        Returns:
            int: unique state id. This id can be used to load the state.
        """
        pass

    def load(self, state, *args, **kwargs):
        """Load / Restore the simulator to a previous state.

        Args:
            state (int, str): unique state id, or path to the file containing the state.
        """
        pass

    def load_plugin(self, plugin_path, name, *args, **kwargs):
        """Load a certain plugin in the simulator.

        Args:
            plugin_path (str): path, location on disk where to find the plugin
            name (str): postfix name of the plugin that is appended to each API

        Returns:
             int: unique plugin id. If this id is negative, the plugin is not loaded. Once a plugin is loaded, you can
                send commands to the plugin using `execute_plugin_commands`
        """
        pass

    def execute_plugin_commands(self, plugin_id, *args, **kwargs):
        """Execute the commands on the specified plugin.

        Args:
            plugin_id (int): unique plugin id.
            *args (list): list of argument values to be interpreted by the plugin. One can be a string, while the
                others must be integers or float.
        """
        pass

    def unload_plugin(self, plugin_id, *args, **kwargs):
        """Unload the specified plugin from the simulator.

        Args:
            plugin_id (int): unique plugin id.
        """
        pass

    ######################################
    # loading URDFs, SDFs, MJCFs, meshes #
    ######################################

    def load_urdf(self, filename, position=None, orientation=(0., 0., 0., 1.), use_fixed_base=0, scale=1.0, *args,
                  **kwargs):
        """Load a URDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the URDF file on the file system of the physics server.
            position (np.array[float[3]]): create the base of the object at the specified position in world space
              coordinates [x,y,z].
            orientation (np.array[float[4]]): create the base of the object at the specified orientation as world
              space quaternion [x,y,z,w].
            use_fixed_base (bool): force the base of the loaded object to be static
            scale (float): scale factor to the URDF model.

        Returns:
            int (non-negative): unique id associated to the load model.
        """
        skeleton = self._urdf_parser.parseSkeleton(filename)
        self.world.addSkeleton(skeleton)
        rpy = get_rpy_from_quaternion(orientation) if orientation is not None else None

        # set orientation
        if rpy is not None:
            for i in range(3):
                skeleton.setPosition(index=i, position=rpy[i])

        # set position
        if position is not None:
            for i in range(3):
                skeleton.setPosition(index=i+3, position=position[i])

        return self.world.getNumSkeletons() - 1

    def load_sdf(self, filename, scaling=1., *args, **kwargs):
        """Load a SDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the SDF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        pass

    def load_mjcf(self, filename, scaling=1., *args, **kwargs):
        """Load a Mujoco file in the simulator.

        Args:
            filename (str): a relative or absolute path to the MJCF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        mujoco_parser = MuJoCoParser(filename=filename)
        urdf_generator = URDFParser()
        skel_generator = SkelParser()

        # generate the URDF/Skel

        raise NotImplementedError("This method is not available yet with the DART simulator.")

    def load_mesh(self, filename, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=None,
                  with_collision=True, flags=None, *args, **kwargs):
        """Load a mesh into the simulator.

        Args:
            filename (str): path to file for the mesh. Currently, only Wavefront .obj. It will create convex hulls
                for each object (marked as 'o') in the .obj file.
            position (float[3]): position of the mesh in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the mesh using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass of the mesh (in kg). If mass = 0, it won't move even if there is a collision.
            scale (float[3]): scale the mesh in the (x,y,z) directions
            color (int[4], None): color of the mesh for red, green, blue, and alpha, each in range [0,1].
            with_collision (bool): If True, it will also create the collision mesh, and not only a visual mesh.
            flags (int, None): if flag = `sim.GEOM_FORCE_CONCAVE_TRIMESH` (=1), this will create a concave static
                triangle mesh. This should not be used with dynamic/moving objects, only for static (mass=0) terrain.

        Returns:
            int: unique id of the mesh in the world
        """
        # compute the mesh inertia

        # create URDF with that file

        self.world.addSkeleton(filename)

    @staticmethod
    def get_available_sdfs(fullpath=False):
        """Return the list of available SDFs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the SDFs. If False, it will just return the
                name of the SDF files (without the extension).
        """
        return []

    @staticmethod
    def get_available_urdfs(fullpath=False):
        """Return the list of available URDFs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the URDFs. If False, it will just return the
                name of the URDF files (without the extension).
        """
        return []

    @staticmethod
    def get_available_mjcfs(fullpath=False):
        """Return the list of available MJCFs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the MJCFs. If False, it will just return the
            name of the MJCF files (without the extension).
        """
        return []

    @staticmethod
    def get_available_objs(fullpath=False):
        """Return the list of available OBJs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the OBJs. If False, it will just return the
                name of the OBJ files (without the extension).
        """
        return []

    ##########
    # Bodies #
    ##########

    def create_primitive_object(self, shape_type, position, mass, orientation=(0., 0., 0., 1.), radius=0.5,
                                half_extents=(.5, .5, .5), height=1., filename=None, mesh_scale=(1., 1., 1.),
                                plane_normal=(0., 0., 1.), rgba_color=None, specular_color=None, frame_position=None,
                                frame_orientation=None, vertices=None, indices=None, uvs=None, normals=None, flags=-1):
        """Create a primitive object in the simulator. This is basically the combination of `create_visual_shape`,
        `create_collision_shape`, and `create_body`.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5)
            position (np.array[float[3]]): Cartesian world position of the base
            mass (float): mass of the base, in kg (if using SI units)
            orientation (np.array[float[4]]): Orientation of base as quaternion [x,y,z,w]
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.array[float[3]], list/tuple of 3 floats): only for GEOM_BOX.
            height (float): only for GEOM_CAPSULE, GEOM_CYLINDER (height = length).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.array[float[3]], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.array[float[3]], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            rgba_color (list/tuple of 4 floats): color components for red, green, blue and alpha, each in range [0..1].
            specular_color (list/tuple of 3 floats): specular reflection color, red, green, blue components in range
                [0..1]
            frame_position (np.array[float[3]]): translational offset of the visual and collision shape with respect
              to the link frame.
            frame_orientation (np.array[float[4]]): rotational offset (quaternion x,y,z,w) of the visual and collision
              shape with respect to the link frame.
            vertices (list[np.array[float[3]]]): Instead of creating a mesh from obj file, you can provide vertices,
              indices, uvs and normals
            indices (list[int]): triangle indices, should be a multiple of 3.
            uvs (list of np.array[2]): uv texture coordinates for vertices. Use `changeVisualShape` to choose the
              texture image. The number of uvs should be equal to number of vertices.
            normals (list[np.array[float[3]]]): vertex normals, number should be equal to number of vertices.
            flags (int): unused / to be decided

        Returns:
            int: non-negative unique id for primitive object, or -1 for failure
        """
        # increment body counter
        self._body_cnt += 1

        # create new skeleton
        skeleton = dart.dynamics.Skeleton(name="body_" + str(self._body_cnt))

        # create shape based on given primitive
        if shape_type == self.GEOM_BOX:
            shape = dart.dynamics.BoxShape(size=2*np.asarray(half_extents))
        elif shape_type == self.GEOM_SPHERE:
            shape = dart.dynamics.SphereShape(radius=radius)
        elif shape_type == self.GEOM_CAPSULE:
            shape = dart.dynamics.CapsuleShape(radius=radius, height=height)
        elif shape_type == self.GEOM_CYLINDER:
            shape = dart.dynamics.CylinderShape(radius=radius, height=height)
        elif shape_type == self.GEOM_ELLIPSOID:
            shape = dart.dynamics.EllipsoidShape(diameters=np.asarray(half_extents))
        elif shape_type == self.GEOM_CONE:
            shape = dart.dynamics.ConeShape(radius=radius, height=height)
        # elif shape_type == self.GEOM_ARROW:
        #     shape = dart.dynamics.ArrowShape(tail=, head=)
        # elif shape_type == self.GEOM_MESH:
        #     shape = dart.dynamics.MeshShape(scale=, mesh=, uri=)
        else:
            raise NotImplementedError("Primitive object not defined for the given shape type.")

        # create body node and free joint
        # if self._floor_id is not None:
        #     floor = self.world.getSkeleton(index=self._floor_id).getBodyNode(index=0)
        #     joint, body = skeleton.createFreeJointAndBodyNodePair()  # parent=floor)
        # else:
        #     joint, body = skeleton.createFreeJointAndBodyNodePair()
        if mass == 0:
            joint, body = skeleton.createWeldJointAndBodyNodePair()
        else:
            joint, body = skeleton.createFreeJointAndBodyNodePair()

        # set the dynamics properties
        if mass != 0:
            body.setMass(mass=mass)
            # body.setMomentOfInertia(shape.computeInertia(mass=1))
        body.setCollidable(True)

        # set the shape to the body node
        shape_node = body.createShapeNode(shape)

        # create visual aspect
        visual_aspect = shape_node.createVisualAspect()
        rgba_color = (1, 1, 1, 1) if rgba_color is None else rgba_color
        visual_aspect.setColor(rgba_color)

        # create collision aspect (to enable collisions)
        collision_aspect = shape_node.createCollisionAspect()

        # create dynamics aspect (for friction and restitution)
        dynamics_aspect = shape_node.createDynamicsAspect()

        # set the position and orientation
        orientation = get_rpy_from_quaternion(orientation) if orientation is not None else None
        if orientation is not None:
            for i in range(3):
                joint.setPosition(i, orientation[i])
        for i in range(3):
            joint.setPosition(i+3, position[i])

        # # increment body counter and remember the body
        # self._body_cnt += 1
        # self._bodies[self._body_cnt] = skeleton
        # return self._body_cnt

        # add skeleton to the world
        self.world.addSkeleton(skeleton)

        # return skeleton index in the world
        return self.world.getNumSkeletons() - 1

    def load_floor(self, dimension=20):
        """Load a floor in the simulator.

        Args:
            dimension (float): dimension of the floor.

        Returns:
            int: non-negative unique id for the floor, or -1 for failure.
        """
        # ground = self._urdf_parser.parseSkeleton("dart://sample/urdf/KR5/ground.urdf")
        # self.world.addSkeleton(ground)
        # return self.world.getNumSkeletons() - 1
        dim = dimension/2.
        self._floor_id = self.create_primitive_object(shape_type=self.GEOM_BOX, position=(0, 0, 0), mass=0,
                                                      half_extents=(dim, dim, 0.01))
        return self._floor_id

    def create_body(self, visual_shape_id=-1, collision_shape_id=-1, mass=0., position=(0., 0., 0.),
                    orientation=(0., 0., 0., 1.), *args, **kwargs):
        """Create a body in the simulator.

        Args:
            visual_shape_id (int): unique id from createVisualShape or -1. You can reuse the visual shape (instancing)
            collision_shape_id (int): unique id from createCollisionShape or -1. You can re-use the collision shape
                for multiple multibodies (instancing)
            mass (float): mass of the base, in kg (if using SI units)
            position (np.array[float[3]]): Cartesian world position of the base
            orientation (np.array[float[4]]): Orientation of base as quaternion [x,y,z,w]

        Returns:
            int: non-negative unique id or -1 for failure.
        """
        # TODO

        # create skeleton
        skeleton = dart.dynamics.Skeleton()

        # add skeleton to the world
        self.world.addSkeleton(skeleton)

        return self.world.getNumSkeletons() - 1

    def remove_body(self, body_id):
        """Remove a particular body in the simulator.

        Args:
            body_id (int): unique body id.
        """
        self.world.removeSkeleton(body_id)

    def num_bodies(self):
        """Return the number of bodies present in the simulator.

        Returns:
            int: number of bodies
        """
        return self.world.getNumSkeletons()

    def get_body_info(self, body_id):
        """Get the specified body information.

        Specifically, it returns the base name extracted from the URDF, SDF, MJCF, or other file.

        Args:
            body_id (int): unique body id.

        Returns:
            str: base name
        """
        pass

    def get_body_id(self, index):
        """Get the body id associated to the index which is between 0 and `num_bodies()`.

        Args:
            index (int): index between [0, `num_bodies()`]

        Returns:
            int: unique body id.
        """
        return index

    ###############
    # constraints #
    ###############

    def create_constraint(self, parent_body_id, parent_link_id, child_body_id, child_link_id, joint_type,
                          joint_axis, parent_frame_position, child_frame_position,
                          parent_frame_orientation=(0., 0., 0., 1.), child_frame_orientation=(0., 0., 0., 1.),
                          *args, **kwargs):
        """
        Create a constraint.

        Args:
            parent_body_id (int): parent body unique id
            parent_link_id (int): parent link index (or -1 for the base)
            child_body_id (int): child body unique id, or -1 for no body (specify a non-dynamic child frame in world
                coordinates)
            child_link_id (int): child link index, or -1 for the base
            joint_type (int): joint type: JOINT_PRISMATIC (=1), JOINT_FIXED (=4), JOINT_POINT2POINT (=5),
                JOINT_GEAR (=6)
            joint_axis (np.array[float[3]]): joint axis, in child link frame
            parent_frame_position (np.array[float[3]]): position of the joint frame relative to parent CoM frame.
            child_frame_position (np.array[float[3]]): position of the joint frame relative to a given child CoM frame
                (or world origin if no child specified)
            parent_frame_orientation (np.array[float[4]]): the orientation of the joint frame relative to parent CoM
                coordinate frame
            child_frame_orientation (np.array[float[4]]): the orientation of the joint frame relative to the child CoM
                coordinate frame (or world origin frame if no child specified)

        Returns:
            int: constraint unique id.
        """
        # Check dart.constraint.*
        # ['BallJointConstraint', 'BoxedLcpConstraintSolver', 'BoxedLcpSolver', 'ConstraintBase', 'ConstraintSolver',
        # 'DantzigBoxedLcpSolver', 'JointConstraint', 'JointCoulombFrictionConstraint', 'JointLimitConstraint',
        # 'PgsBoxedLcpSolver', 'PgsBoxedLcpSolverOption', 'WeldJointConstraint']

        parent = self.world.getSkeleton(parent_body_id)
        parent_node = parent.getBodyNode(parent_link_id + 1)

        if child_body_id != -1:  # if child is a skeleton
            child = self.world.getSkeleton(child_body_id)
            child_node = child.getBodyNode(child_link_id + 1)
        else:  # if child is the world
            pass

        # create constraint based on the type
        if joint_type == Simulator.JOINT_FIXED:
            constraint = dart.constraint.WeldJointConstraint(parent_node, child_node)

        self._constraint_cnt += 1
        self._constraints[self._constraint_cnt] = constraint

        # add constraint
        self.world.getConstraintSolver().addConstraint(constraint)

        return self._constraint_cnt

    def remove_constraint(self, constraint_id):
        """
        Remove the specified constraint.

        Args:
            constraint_id (int): constraint unique id.
        """
        constraint = self._constraints.pop(constraint_id)
        self.world.getConstraintSolver().removeConstraint(constraint)

    def change_constraint(self, constraint_id, *args, **kwargs):
        """
        Change the parameters of an existing constraint.

        Args:
            constraint_id (int): constraint unique id.
        """
        constraint = self._constraints.pop(constraint_id)
        pass

    def num_constraints(self):
        """
        Get the number of constraints created.

        Returns:
            int: number of constraints created.
        """
        return len(self._constraints)

    def get_constraint_id(self, index):
        """
        Get the constraint unique id associated with the index which is between 0 and `num_constraints()`.

        Args:
            index (int): index between [0, `num_constraints()`]

        Returns:
            int: constraint unique id.
        """
        return list(self._constraints.keys())[index]

    def get_constraint_info(self, constraint_id):
        """
        Get information about the given constaint id.

        Args:
            constraint_id (int): constraint unique id.

        Returns:
            dict, list: info
        """
        pass

    def get_constraint_state(self, constraint_id):
        """
        Get the state of the given constraint.

        Args:
            constraint_id (int): constraint unique id.

        Returns:
            dict, list: state
        """
        pass

    ###########
    # objects #
    ###########

    def get_mass(self, body_id):
        """
        Return the total mass of the robot (=sum of all mass links).

        Args:
            body_id (int): unique object id, as returned from `load_urdf`.

        Returns:
            float: total mass of the robot [kg]
        """
        skeleton = self.world.getSkeleton(body_id)
        return skeleton.getMass()

    def get_base_mass(self, body_id):
        """Return the base mass of the robot.

        Args:
            body_id (int): unique object id.
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()  # this is the same as getBodyNode(0)
        return base.getMass()

    def get_base_name(self, body_id):
        """
        Return the base name.

        Args:
            body_id (int): unique object id.

        Returns:
            str: base name
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()  # this is the same as getBodyNode(0)
        return base.getName()

    def get_center_of_mass_position(self, body_id, link_ids=None):
        """
        Return the center of mass position.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: center of mass position in the Cartesian world coordinates
        """
        skeleton = self.world.getSkeleton(body_id)

        if link_ids is None:
            return skeleton.getCOM()

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getCOM()

        com = 0
        for link_id in link_ids:
            body = skeleton.getBodyNode(link_id + 1)
            com += body.getCOM * body.getMass()
        com /= skeleton.getMass()

        return com

    def get_center_of_mass_velocity(self, body_id, link_ids=None):
        """
        Return the center of mass linear velocity.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: center of mass linear velocity.
        """
        skeleton = self.world.getSkeleton(body_id)

        if link_ids is None:
            return skeleton.getCOMLinearVelocity()

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getCOMLinearVelocity()

        vel = 0
        for link_id in link_ids:
            body = skeleton.getBodyNode(link_id + 1)
            vel += body.getCOMLinearVelocity() * body.getMass()
        vel /= skeleton.getMass()
        return vel

    def get_base_pose(self, body_id):
        """
        Get the current position and orientation of the base (or root link) of the body in Cartesian world coordinates.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position
            np.array[float[4]]: base orientation (quaternion [x,y,z,w])
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()
        transform = base.getWorldTransform()
        # position = self._get_pos_from_transform(transform)
        position = base.getCOM()
        orientation = self._get_quat_from_transform(transform)
        return position, orientation

    def get_base_position(self, body_id):
        """
        Return the base position of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position.
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()
        # return self._get_pos_from_transform(base.getWorldTransform())
        return base.getCOM()

    def get_base_orientation(self, body_id):
        """
        Get the base orientation of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[4]]: base orientation in the form of a quaternion (x,y,z,w)
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()
        transform = base.getWorldTransform()
        return self._get_quat_from_transform(transform)

    def reset_base_pose(self, body_id, position, orientation):
        """
        Reset the base position and orientation of the specified object id.

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        skeleton = self.world.getSkeleton(body_id)
        rpy = get_rpy_from_quaternion(orientation)
        for i in range(6):
            if i < 3:
                skeleton.setPosition(index=i, position=rpy[i])
            else:
                skeleton.setPosition(index=i, position=position[i - 3])

    def reset_base_position(self, body_id, position):
        """
        Reset the base position of the specified body/object id while preserving its orientation.

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
        """
        skeleton = self.world.getSkeleton(body_id)
        for i in range(3):
            skeleton.setPosition(index=i+3, position=position[i])

    def reset_base_orientation(self, body_id, orientation):
        """
        Reset the base orientation of the specified body/object id while preserving its position.

        Args:
            body_id (int): unique object id.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        skeleton = self.world.getSkeleton(body_id)
        rpy = get_rpy_from_quaternion(orientation)
        for i in range(3):
            skeleton.setPosition(index=i, position=rpy[i])

    def get_base_velocity(self, body_id):
        """
        Return the base linear and angular velocities.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: linear velocity of the base in Cartesian world space coordinates
            np.array[float[3]]: angular velocity of the base in Cartesian world space coordinates
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()
        lin_vel = base.getLinearVelocity()
        ang_vel = base.getAngularVelocity()
        return lin_vel.reshape(-1), ang_vel.reshape(-1)

    def get_base_linear_velocity(self, body_id):
        """
        Return the linear velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: linear velocity of the base in Cartesian world space coordinates
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()
        lin_vel = base.getLinearVelocity()
        return lin_vel.reshape(-1)

    def get_base_angular_velocity(self, body_id):
        """
        Return the angular velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: angular velocity of the base in Cartesian world space coordinates
        """
        base = self.world.getSkeleton(body_id).getRootBodyNode()
        ang_vel = base.getAngularVelocity()
        return ang_vel.reshape(-1)

    def reset_base_velocity(self, body_id, linear_velocity=None, angular_velocity=None):
        """
        Reset the base velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base.
            angular_velocity (np.array[float[3]]): new angular velocity of the base.
        """
        skeleton = self.world.getSkeleton(body_id)
        for i in range(3):
            skeleton.setVelocity(index=i + 3, velocity=linear_velocity[i])

    def reset_base_linear_velocity(self, body_id, linear_velocity):
        """
        Reset the base linear velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base
        """
        skeleton = self.world.getSkeleton(body_id)
        for i in range(3):
            skeleton.setVelocity(index=i+3, velocity=linear_velocity[i])

    def reset_base_angular_velocity(self, body_id, angular_velocity):
        """
        Reset the base angular velocity.

        Args:
            body_id (int): unique object id.
            angular_velocity (np.array[float[3]]): new angular velocity of the base
        """
        skeleton = self.world.getSkeleton(body_id)
        for i in range(3):
            skeleton.setVelocity(index=i, velocity=angular_velocity[i])

    def apply_external_force(self, body_id, link_id=-1, force=(0., 0., 0.), position=(0., 0., 0.), frame=1):
        """
        Apply the specified external force on the specified position on the body / link.

        Args:
            body_id (int): unique body id.
            link_id (int): unique link id. If -1, it will be the base.
            force (np.array[float[3]]): external force to be applied.
            position (np.array[float[3]]): position on the link where the force is applied. See `flags` for coordinate
                systems. If None, it is the center of mass of the body (or the link if specified).
            frame (int): if frame = 1, then the force / position is described in the link frame. If frame = 2, they
                are described in the world frame.
        """
        link = self.world.getSkeleton(body_id).getBodyNode(link_id + 1)
        force = np.asarray(force).reshape(-1, 1)  # (3,1)
        offset = np.asarray(position).reshape(-1, 1)  # (3,1)
        is_local = (frame == Simulator.LINK_FRAME)
        link.setExtForce(force=force, offset=offset, isForceLocal=is_local, isOffsetLocal=is_local)

    def apply_external_torque(self, body_id, link_id=-1, torque=(0., 0., 0.), frame=1):
        """
        Apply an external torque on a body, or a link of the body. Note that after each simulation step, the external
        torques are cleared to 0.

        Args:
            body_id (int): unique body id.
            link_id (int): link id to apply the torque, if -1 it will apply the torque on the base
            torque (float[3]): Cartesian torques to be applied on the body
            frame (int): Specify the coordinate system of force/position: either `pybullet.WORLD_FRAME` (=2) for
                Cartesian world coordinates or `pybullet.LINK_FRAME` (=1) for local link coordinates.
        """
        link = self.world.getSkeleton(body_id).getBodyNode(link_id + 1)
        torque = np.asarray(torque).reshape(-1, 1)  # (3,1)
        is_local = (frame == Simulator.LINK_FRAME)
        link.setExtForce(torque=torque, isLocal=is_local)

    ###################
    # transformations #
    ###################

    #############################
    # robots (joints and links) #
    #############################

    def num_joints(self, body_id):
        """
        Return the total number of joints of the specified body. This is the same as calling `num_links`.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of joints with the associated body id.
        """
        skeleton = self.world.getSkeleton(body_id)
        return skeleton.getNumJoints()  # TODO: check if we have to add: -1

    def num_actuated_joints(self, body_id):
        """
        Return the total number of actuated joints associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of actuated joints of the specified body.
        """
        skeleton = self.world.getSkeleton(body_id)
        return skeleton.getNumDofs()  # TODO: check with fixed and floating base

    def num_links(self, body_id):
        """
        Return the total number of links of the specified body. This is the same as calling `num_joints`.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of links with the associated body id.
        """
        # skeleton = self.world.getSkeleton(body_id)
        # return skeleton.getNumBodyNodes()
        return self.num_joints(body_id)

    def get_joint_info(self, body_id, joint_id):
        """
        Return information about the given joint about the specified body.

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint id is included in [0..`num_joints(body_id)`].

        Returns:
            [0] int:        the same joint id as the input parameter
            [1] str:        name of the joint (as specified in the URDF/SDF/etc file)
            [2] int:        type of the joint which implie the number of position and velocity variables.
                            The types include JOINT_REVOLUTE (=0), JOINT_PRISMATIC (=1), JOINT_SPHERICAL (=2),
                            JOINT_PLANAR (=3), and JOINT_FIXED (=4).
            [3] int:        q index - the first position index in the positional state variables for this body
            [4] int:        dq index - the first velocity index in the velocity state variables for this body
            [5] int:        flags (reserved)
            [6] float:      the joint damping value (as specified in the URDF file)
            [7] float:      the joint friction value (as specified in the URDF file)
            [8] float:      the positional lower limit for slider and revolute joints
            [9] float:      the positional upper limit for slider and revolute joints
            [10] float:     maximum force specified in URDF. Note that this value is not automatically used.
                            You can use maxForce in 'setJointMotorControl2'.
            [11] float:     maximum velocity specified in URDF. Note that this value is not used in actual
                            motor control commands at the moment.
            [12] str:       name of the link (as specified in the URDF/SDF/etc file)
            [13] np.array[float[3]]:  joint axis in local frame (ignored for JOINT_FIXED)
            [14] np.array[float[3]]:  joint position in parent frame
            [15] np.array[float[4]]:  joint orientation in parent frame
            [16] int:       parent link index, -1 for base
        """
        skeleton = self.world.getSkeleton(body_id)
        joint = skeleton.getJoint(joint_id + 1)

        name = joint.getName()

        if isinstance(joint, dart.dynamics.FreeJoint):
            joint_type = Simulator.JOINT_FREE
        elif isinstance(joint, dart.dynamics.WeldJoint):
            joint_type = Simulator.JOINT_FIXED
        elif isinstance(joint, dart.dynamics.RevoluteJoint):
            joint_type = Simulator.JOINT_REVOLUTE
        elif isinstance(joint, dart.dynamics.BallJoint):
            joint_type = Simulator.JOINT_SPHERICAL
        elif isinstance(joint, dart.dynamics.PrismaticJoint):
            joint_type = Simulator.JOINT_PRISMATIC
        else:
            joint_type = -1

        q_idx = joint.getIndexInTree(joint_id)  # or joint.getIndexInSkeleton(joint_id)
        dq_idx = 0
        flag = -1
        damping = joint.getDampingCoefficient()
        friction = joint.getCoulombFriction()
        pos_limits = (joint.getPositionLowerLimit, joint.getPositionUpperLimit)
        force_limits = (joint.getForceLowerLimit, joint.getForceUpperLimit)
        vel_limits = (joint.getVelocityLowerLimit, joint.getVelocityUpperLimit)
        link_name = joint.getChildBodyNode().getName()
        axis = joint.getAxis() if hasattr(joint, 'getAxis') else np.zeros(3)

        transform = joint.getTransformFromParentBodyNode()
        pos, quat = self._get_pose_from_transform(transform)

        return joint_id, name, joint_type, q_idx, dq_idx, flag, damping, friction, pos_limits, force_limits, \
               vel_limits, link_name, axis, pos, quat

    def get_joint_state(self, body_id, joint_id):
        """
        Get the joint state.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint index in range [0..num_joints(body_id)]

        Returns:
            float: The position value of this joint.
            float: The velocity value of this joint.
            np.array[float[6]]: These are the joint reaction forces, if a torque sensor is enabled for this joint it is
                [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0, 0, 0, 0, 0, 0].
            float: This is the motor torque applied during the last stepSimulation. Note that this only applies in
                VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the applied joint motor torque
                is exactly what you provide, so there is no need to report it separately.
        """
        joint = self.world.getSkeleton(body_id).getJoint(joint_id + 1)
        body = joint.getChildBodyNode()
        position = joint.getPosition(0)  # joints can have less or more than 1 DoF (like WeldJoint, FreeJoint)
        velocity = joint.getVelocity(0)
        reaction_forces = body.getExternalForceGlobal()
        torque = joint.getForce(0)
        return position, velocity, reaction_forces, torque

    def get_joint_states(self, body_id, joint_ids):
        """
        Get the joint state of the specified joints.

        Args:
            body_id (int): unique body id.
            joint_ids (list[int]): list of joint ids.

        Returns:
            list:
                float: The position value of this joint.
                float: The velocity value of this joint.
                np.array[float[6]]: These are the joint reaction forces, if a torque sensor is enabled for this joint
                    it is [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0, 0, 0, 0, 0, 0].
                float: This is the motor torque applied during the last `step`. Note that this only applies in
                    VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the applied joint motor
                    torque is exactly what you provide, so there is no need to report it separately.
        """
        if isinstance(joint_ids, int):
            return self.get_joint_state(body_id, joint_ids)
        return [self.get_joint_state(body_id, joint_id) for joint_id in joint_ids]

    def reset_joint_state(self, body_id, joint_id, position, velocity=None):
        """
        Reset the state of the joint. It is best only to do this at the start, while not running the simulation:
        `reset_joint_state` overrides all physics simulation.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint index in range [0..num_joints(body_id)]
            position (float): the joint position (angle in radians [rad] or position [m])
            velocity (float): the joint velocity (angular [rad/s] or linear velocity [m/s])
        """
        joint = self.world.getSkeleton(body_id).getJoint(joint_id + 1)
        # joint.resetPosition(0)
        # joint.resetVelocity(0)
        joint.setPosition(0, position)
        if velocity is not None:
            joint.setVelocity(0, velocity)

    def enable_joint_force_torque_sensor(self, body_id, joint_ids, enable=True):
        """
        You can enable or disable a joint force/torque sensor in each joint.

        Args:
            body_id (int): body unique id.
            joint_ids (int, int[N]): joint index in range [0..num_joints(body_id)], or list of joint ids.
            enable (bool): True to enable, False to disable the force/torque sensor
        """
        pass

    def set_joint_motor_control(self, body_id, joint_ids, control_mode=Simulator.POSITION_CONTROL, positions=None,
                                velocities=None, forces=None, kp=None, kd=None, max_velocity=None):
        r"""
        Set the joint motor control.

        In position control:
        .. math:: error = Kp (x_{des} - x) + Kd (\dot{x}_{des} - \dot{x})

        In velocity control:
        .. math:: error = \dot{x}_{des} - \dot{x}

        Note that the maximum forces and velocities are not automatically used for the different control schemes.

        Args:
            body_id (int): body unique id.
            joint_ids (int): joint/link id, or list of joint ids.
            control_mode (int): POSITION_CONTROL (=2) (which is in fact CONTROL_MODE_POSITION_VELOCITY_PD),
                VELOCITY_CONTROL (=0), TORQUE_CONTROL (=1) and PD_CONTROL (=3).
            positions (float, np.array[float[N]]): target joint position(s) (used in POSITION_CONTROL).
            velocities (float, np.array[float[N]]): target joint velocity(ies). In VELOCITY_CONTROL and
                POSITION_CONTROL, the target velocity(ies) is(are) the desired velocity of the joint. Note that the
                target velocity(ies) is(are) not the maximum joint velocity(ies). In PD_CONTROL and
                POSITION_CONTROL/CONTROL_MODE_POSITION_VELOCITY_PD, the final target velocities are computed using:
                `kp*(erp*(desiredPosition-currentPosition)/dt)+currentVelocity+kd*(m_desiredVelocity - currentVelocity)`
            forces (float, list[float]): in POSITION_CONTROL and VELOCITY_CONTROL, these are the maximum motor
                forces used to reach the target values. In TORQUE_CONTROL these are the forces / torques to be applied
                each simulation step.
            kp (float, list[float]): position (stiffness) gain(s) (used in POSITION_CONTROL).
            kd (float, list[float]): velocity (damping) gain(s) (used in POSITION_CONTROL).
            max_velocity (float): in POSITION_CONTROL this limits the velocity to a maximum.
        """
        pass

    def get_link_state(self, body_id, link_id, compute_velocity=False, compute_forward_kinematics=False):
        """
        Get the state of the associated link.

        Args:
            body_id (int): body unique id.
            link_id (int): link index.
            compute_velocity (bool): If True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            np.array[float[3]]: Cartesian position of CoM
            np.array[float[4]]: Cartesian orientation of CoM, in quaternion [x,y,z,w]
            np.array[float[3]]: local position offset of inertial frame (center of mass) expressed in the URDF link
                frame
            np.array[float[4]]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed in URDF
                link frame
            np.array[float[3]]: world position of the URDF link frame
            np.array[float[4]]: world orientation of the URDF link frame
            np.array[float[3]]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
            np.array[float[3]]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        body = self.world.getSkeleton(body_id).getBodyNode(link_id + 1)
        transform = body.getWorldTransform()
        com_position = body.getCOM().reshape(-1)
        com_orientation = get_quaternion_from_matrix(transform[:-1, :-1])

        local_position = body.getLocalCOM().reshape(-1)
        local_orientation = np.array([0., 0., 0., 1.])

        world_position = transform[:-1, 3]
        world_orientation = np.array(com_orientation)

        results = [com_position, com_orientation, local_position, local_orientation, world_position, world_orientation]

        if compute_velocity:
            linear_velocity = body.getLinearVelocity().reshape(-1)
            angular_velocity = body.getAngularVelocity().reshape(-1)
            results.append(linear_velocity)
            results.append(angular_velocity)

        return results

    def get_link_states(self, body_id, link_ids, compute_velocity=False, compute_forward_kinematics=False):
        """
        Get the state of the associated links.

        Args:
            body_id (int): body unique id.
            link_ids (list[int]): list of link index.
            compute_velocity (bool): If True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            list:
                np.array[float[3]]: Cartesian position of CoM
                np.array[float[4]]: Cartesian orientation of CoM, in quaternion [x,y,z,w]
                np.array[float[3]]: local position offset of inertial frame (center of mass) expressed in the URDF
                    link frame
                np.array[float[4]]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed in
                    URDF link frame
                np.array[float[3]]: world position of the URDF link frame
                np.array[float[4]]: world orientation of the URDF link frame
                np.array[float[3]]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
                np.array[float[3]]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        if isinstance(link_ids, int):
            return self.get_link_state(body_id, link_ids, compute_velocity, compute_forward_kinematics)
        return [self.get_link_state(body_id, link_id, compute_velocity, compute_forward_kinematics)
                for link_id in link_ids]

    def get_link_names(self, body_id, link_ids):
        """
        Return the name of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link id, or list of link ids.

        Returns:
            if 1 link:
                str: link name
            if multiple links:
                str[N]: link names
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getName()

        return [skeleton.getBodyNode(link + 1).getName() for link in link_ids]

    def get_link_masses(self, body_id, link_ids):
        """
        Return the mass of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link id, or list of link ids.

        Returns:
            if 1 link:
                float: mass of the given link
            else:
                float[N]: mass of each link
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getMass()

        return [skeleton.getBodyNode(link + 1).getMass() for link in link_ids]

    def get_link_frames(self, body_id, link_ids):
        r"""
        Return the link world frame position(s) and orientation(s).

        Args:
            body_id (int): body id.
            link_ids (int, int[N]): link id, or list of desired link ids.

        Returns:
            if 1 link:
                np.array[float[3]]: the link frame position in the world space
                np.array[float[4]]: Cartesian orientation of the link frame [x,y,z,w]
            if multiple links:
                np.array[float[N,3]]: link frame position of each link in world space
                np.array[float[N,4]]: orientation of each link frame [x,y,z,w]
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            transform = skeleton.getBodyNode(link_ids + 1).getWorldTransform()
            return transform[:-1, 3], get_quaternion_from_matrix(transform[:-1, :-1])

        positions, orientations = [], []
        for link_id in link_ids:
            transform = skeleton.getBodyNode(link_ids + 1).getWorldTransform()
            positions.append(transform[:-1, 3])
            orientations.append(get_quaternion_from_matrix(transform[:-1, :-1]))

        return np.array(positions), np.array(orientations)

    def get_link_world_positions(self, body_id, link_ids):
        """
        Return the CoM position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: the link CoM position in the world space
            if multiple links:
                np.array[float[N,3]]: CoM position of each link in world space
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getCOM().reshape(-1)

        return [skeleton.getBodyNode(link + 1).getCOM().reshape(-1) for link in link_ids]

    def get_link_positions(self, body_id, link_ids):
        pass

    def get_link_world_orientations(self, body_id, link_ids):
        """
        Return the CoM orientation (in the Cartesian world space) of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): list of link indices.

        Returns:
            if 1 link:
                np.array[float[4]]: Cartesian orientation of the link CoM (x,y,z,w)
            if multiple links:
                np.array[float[N,4]]: CoM orientation of each link (x,y,z,w)
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            return get_quaternion_from_matrix(skeleton.getBodyNode(link_ids + 1).getWorldTransform()[:-1, :-1])

        return [get_quaternion_from_matrix(skeleton.getBodyNode(link + 1).getWorldTransform()[:-1, :-1])
                for link in link_ids]

    def get_link_orientations(self, body_id, link_ids):
        pass

    def get_link_world_linear_velocities(self, body_id, link_ids):
        """
        Return the linear velocity of the link(s) expressed in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: linear velocity of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,3]]: linear velocity of each link
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getLinearVelocity().reshape(-1)

        return [skeleton.getBodyNode(link + 1).getLinearVelocity.reshape(-1) for link in link_ids]

    def get_link_world_angular_velocities(self, body_id, link_ids):
        """
        Return the angular velocity of the link(s) in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: angular velocity of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,3]]: angular velocity of each link
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            return skeleton.getBodyNode(link_ids + 1).getAngularVelocity().reshape(-1)

        return [skeleton.getBodyNode(link + 1).getAngularVelocity.reshape(-1) for link in link_ids]

    def get_link_world_velocities(self, body_id, link_ids):
        """
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        link(s).

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): list of link indices.

        Returns:
            if 1 link:
                np.array[float[6]]: linear and angular velocity of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,6]]: linear and angular velocity of each link
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(link_ids, int):
            link = skeleton.getBodyNode(link_ids + 1)
            lin_vel = link.getLinearVelocity().reshape(-1)
            ang_vel = link.getAngularVelocity().reshape(-1)
            return np.concatenate((lin_vel, ang_vel))

        velocities = []
        for link_id in link_ids:
            link = skeleton.getBodyNode(link_id + 1)
            lin_vel = link.getLinearVelocity().reshape(-1)
            ang_vel = link.getAngularVelocity().reshape(-1)
            velocities.append(np.concatenate((lin_vel, ang_vel)))
        return velocities

    def get_link_velocities(self, body_id, link_ids):
        pass

    def get_q_indices(self, body_id, joint_ids):
        """
        Get the corresponding q index of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                int: q index
            if multiple joints:
                np.array[int[N]]: q indices
        """
        pass

    def get_actuated_joint_ids(self, body_id):
        """
        Get the actuated joint ids associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            list[int]: actuated joint ids.
        """
        skeleton = self.world.getSkeleton(body_id)

        joint_ids = []
        for joint_id in range(1, skeleton.getNumJoints()):
            num_dofs = skeleton.getJoint(joint_id).getNumDofs()
            if num_dofs > 0:
                joint_ids.append(joint_id - 1)
        return joint_ids

    def get_joint_names(self, body_id, joint_ids):
        """
        Return the name of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                str: name of the joint
            if multiple joints:
                str[N]: name of each joint
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            return skeleton.getJoint(joint_ids + 1).getName()

        return [skeleton.getJoint(joint + 1).getName() for joint in joint_ids]

    def get_joint_type_ids(self, body_id, joint_ids):
        """
        Get the joint type ids.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                int: joint type id.
            if multiple joints: list of above
        """
        pass

    def get_joint_type_names(self, body_id, joint_ids):  # TODO: make sure it is the same as other simulators
        """
        Get joint type names.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                str: joint type name.
            if multiple joints: list of above
        """
        # bullet: ['revolute', 'prismatic', 'spherical', 'planar', 'fixed', 'point2point', 'gear']
        # dart: ['ball', 'free', 'euler', 'weld' (=fixed), 'revolute', 'universal', 'prismatic']
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            return skeleton.getJoint(joint_ids + 1).getType()

        return [skeleton.getJoint(joint + 1).getType() for joint in joint_ids]

    def get_joint_dampings(self, body_id, joint_ids):
        """
        Get the damping coefficient of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: damping coefficient of the given joint
            if multiple joints:
                np.array[float[N]]: damping coefficient for each specified joint
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            return skeleton.getJoint(joint_ids + 1).getDampingCoefficient(0)

        return [skeleton.getJoint(joint + 1).getDampingCoefficient(0) for joint in joint_ids]

    def get_joint_frictions(self, body_id, joint_ids):
        """
        Get the friction coefficient of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: friction coefficient of the given joint
            if multiple joints:
                np.array[float[N]]: friction coefficient for each specified joint
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            return skeleton.getJoint(joint_ids + 1).getCoulombFriction(0)

        return [skeleton.getJoint(joint + 1).getCoulombFriction(0) for joint in joint_ids]

    def get_joint_limits(self, body_id, joint_ids):
        """
        Get the joint limits of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                np.array[float[2]]: lower and upper limit
            if multiple joints:
                np.array[float[N,2]]: lower and upper limit for each specified joint
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            joint = skeleton.getJoint(joint_ids + 1)
            return np.array([joint.getPositionLowerLimit(0), joint.getPositionUpperLimit(0)])

        limits = []
        for joint_id in joint_ids:
            joint = skeleton.getJoint(joint_id + 1)
            limits.append([joint.getPositionLowerLimit(0), joint.getPositionUpperLimit(0)])
        return np.array(limits)

    def get_joint_max_forces(self, body_id, joint_ids):
        """
        Get the maximum force that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: maximum force [N]
            if multiple joints:
                np.array[float[N]]: maximum force for each specified joint [N]
        """
        # TODO
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            joint = skeleton.getJoint(joint_ids + 1)
            return np.max(np.abs([joint.getForceLowerLimit(0), joint.getForceUpperLimit(0)]))

        forces = []
        for joint_id in joint_ids:
            joint = skeleton.getJoint(joint_id + 1)
            forces.append(np.max(np.abs([joint.getForceLowerLimit(0), joint.getForceUpperLimit(0)])))
        return np.array(forces)

    def get_joint_max_velocities(self, body_id, joint_ids):
        """
        Get the maximum velocity that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: maximum velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: maximum velocities for each specified joint [rad/s]
        """
        # TODO
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            joint = skeleton.getJoint(joint_ids + 1)
            return np.max(np.abs([joint.getVelocityLowerLimit(0), joint.getVelocityUpperLimit(0)]))

        velocities = []
        for joint_id in joint_ids:
            joint = skeleton.getJoint(joint_id + 1)
            velocities.append(np.max(np.abs([joint.getVelocityLowerLimit(0), joint.getVelocityUpperLimit(0)])))
        return np.array(velocities)

    def get_joint_axes(self, body_id, joint_ids):
        """
        Get the joint axis about the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                np.array[float[3]]: joint axis
            if multiple joint:
                np.array[float[N,3]]: list of joint axis
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            joint = skeleton.getJoint(joint_ids + 1)
            if hasattr(joint, 'getAxis'):
                return joint.getAxis()
            return np.zeros(3)  # TODO: should we return None instead?

        axes = []
        for joint_id in joint_ids:
            joint = skeleton.getJoint(joint_id + 1)
            axis = np.zeros(3) if not hasattr(joint, 'getAxis') else joint.getAxis()
            axes.append(axis)
        return np.array(axes)

    def set_joint_positions(self, body_id, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            positions (float, np.array[float[N]]): desired position, or list of desired positions [rad]
            velocities (None, float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.array[float[N]]): position gain(s)
            kds (None, float, np.array[float[N]]): velocity gain(s)
            forces (None, float, np.array[float[N]]): maximum motor force(s)/torque(s) used to reach the target values.
        """
        skeleton = self.world.getSkeleton(body_id)

        # TODO: use position control

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).setPosition(0, positions)

        for joint_id, q in zip(joint_ids, positions):
            skeleton.getJoint(joint_id + 1).setPosition(0, q)

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).getPosition(0)

        return np.array([skeleton.getJoint(joint_id + 1).getPosition(0) for joint_id in joint_ids])

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques
        """
        skeleton = self.world.getSkeleton(body_id)

        # TODO: use velocity control

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).setVelocity(0, velocities)

        for joint_id, dq in zip(joint_ids, velocities):
            skeleton.getJoint(joint_id + 1).setVelocity(0, dq)

    def get_joint_velocities(self, body_id, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: joint velocities [rad/s]
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).getVelocity(0)

        return np.array([skeleton.getJoint(joint_id + 1).getVelocity(0) for joint_id in joint_ids])

    def set_joint_accelerations(self, body_id, joint_ids, accelerations, q=None, dq=None):
        """
        Set the acceleration of the given joint(s) (using force control). This is achieved by performing inverse
        dynamic which given the joint accelerations compute the joint torques to be applied.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            accelerations (float, np.array[float[N]]): desired joint acceleration, or list of desired joint
                accelerations [rad/s^2]
        """
        pass

    def get_joint_accelerations(self, body_id, joint_ids, q=None, dq=None):
        """
        Get the acceleration at the given joint(s). This is carried out by first getting the joint torques, then
        performing forward dynamics to get the joint accelerations from the joint torques.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            q (list[int], None): all the joint positions. If None, it will compute it.
            dq (list[int], None): all the joint velocities. If None, it will compute it.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.array[float[N]]: joint accelerations [rad/s^2]
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).getAcceleration(0)

        return np.array([skeleton.getJoint(joint_id + 1).getAcceleration(0) for joint_id in joint_ids])

    def set_joint_torques(self, body_id, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            torques (float, list[float]): desired torque(s) to apply to the joint(s) [N].
        """
        skeleton = self.world.getSkeleton(body_id)

        # TODO: use velocity control

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).setForce(0, torques)

        for joint_id, dq in zip(joint_ids, torques):
            skeleton.getJoint(joint_id + 1).setForce(0, dq)

    def get_joint_torques(self, body_id, joint_ids):
        """
        Get the applied torque(s) on the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[float[N]]: torques associated to the given joints [Nm]
        """
        skeleton = self.world.getSkeleton(body_id)

        if isinstance(joint_ids, int):
            # Some joints have more or less than 1 DoF (like Free, Weld=Fixed)
            return skeleton.getJoint(joint_ids + 1).getForce(0)

        return np.array([skeleton.getJoint(joint_id + 1).getForce(0) for joint_id in joint_ids])

    def get_joint_reaction_forces(self, body_id, joint_ids):
        """Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            body_id (int): unique body id.
            joint_ids (int, int[N]): joint id, or list of joint ids

        Returns:
            if 1 joint:
                np.array[float[6]]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.array[float[N,6]]: joint reaction forces [N, Nm]
        """
        if isinstance(joint_ids, int):
            joint = self.world.getSkeleton(body_id).getJoint(joint_ids + 1)
            body = joint.getChildBodyNode()
            return body.getExternalForceGlobal()
        forces = []
        for joint_id in joint_ids:
            joint = self.world.getSkeleton(body_id).getJoint(joint_id + 1)
            body = joint.getChildBodyNode()
            forces.append(body.getExternalForceGlobal())
        return np.array(forces)

    def get_joint_powers(self, body_id, joint_ids):
        """Return the applied power at the given joint(s). Power = torque * velocity.

        Args:
            body_id (int): unique body id.
            joint_ids (int, int[N]): joint id, or list of joint ids

        Returns:
            if 1 joint:
                float: joint power [W]
            if multiple joints:
                np.array[float[N]]: power at each joint [W]
        """
        torque = self.get_joint_torques(body_id, joint_ids)
        velocity = self.get_joint_velocities(body_id, joint_ids)
        return torque * velocity

    #################
    # Visualization #
    #################

    def create_visual_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), length=1., filename=None,
                            mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1, rgba_color=None,
                            specular_color=None, visual_frame_position=None, vertices=None, indices=None, uvs=None,
                            normals=None, visual_frame_orientation=None):
        """
        Create a visual shape in the simulator.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5)
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.array[float[3]], list/tuple of 3 floats): only for GEOM_BOX.
            length (float): only for GEOM_CAPSULE, GEOM_CYLINDER (length = height).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.array[float[3]], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.array[float[3]], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            flags (int): unused / to be decided
            rgba_color (list/tuple of 4 floats): color components for red, green, blue and alpha, each in range [0..1].
            specular_color (list/tuple of 3 floats): specular reflection color, red, green, blue components in range
                [0..1]
            visual_frame_position (np.array[float[3]]): translational offset of the visual shape with respect to the
                link frame
            vertices (list of np.array[float[3]]): Instead of creating a mesh from obj file, you can provide vertices,
                indices, uvs and normals
            indices (list[int]): triangle indices, should be a multiple of 3.
            uvs (list of np.array[float[2]]): uv texture coordinates for vertices. Use changeVisualShape to choose the
                texture image. The number of uvs should be equal to number of vertices
            normals (list of np.array[float[3]]): vertex normals, number should be equal to number of vertices.
            visual_frame_orientation (np.array[float[4]]): rotational offset (quaternion x,y,z,w) of the visual shape
                with respect to the link frame

        Returns:
            int: The return value is a non-negative int unique id for the visual shape or -1 if the call failed.
        """
        pass

    def get_visual_shape_data(self, object_id, flags=-1):
        """
        Get the visual shape data associated with the given object id. It will output a list of visual shape data.

        Args:
            object_id (int): object unique id.
            flags (int, None): VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) will also provide `texture_unique_id`.

        Returns:
            list:
                int: object unique id.
                int: link index or -1 for the base
                int: visual geometry type (TBD)
                np.array[float[3]]: dimensions (size, local scale) of the geometry
                str: path to the triangle mesh, if any. Typically relative to the URDF, SDF or MJCF file location, but
                    could be absolute
                np.array[float[3]]: position of local visual frame, relative to link/joint frame
                np.array[float[4]]: orientation of local visual frame relative to link/joint frame
                list of 4 floats: URDF color (if any specified) in Red / Green / Blue / Alpha
                int: texture unique id of the shape or -1 if None. This field only exists if using
                    VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) flag.
        """
        pass

    def change_visual_shape(self, object_id, link_id, shape_id=None, texture_id=None, rgba_color=None,
                            specular_color=None):
        """
        Allows to change the texture of a shape, the RGBA color and other properties.

        Args:
            object_id (int): unique object id.
            link_id (int): link id.
            shape_id (int): shape id.
            texture_id (int): texture id.
            rgba_color (float[4]): RGBA color. Each is in the range [0..1]. Alpha has to be 0 (invisible) or 1
                (visible) at the moment.
            specular_color (int[3]): specular color components, RED, GREEN and BLUE, can be from 0 to large number
                (>100).
        """
        pass

    def load_texture(self, filename):
        """
        Load a texture from file and return a non-negative texture unique id if the loading succeeds.
        This unique id can be used with changeVisualShape.

        Args:
            filename (str): path to the file.

        Returns:
            int: texture unique id. If non-negative, the texture was loaded successfully.
        """
        pass

    def compute_view_matrix(self, eye_position, target_position, up_vector):
        """Compute the view matrix.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            eye_position (np.array[float[3]]): eye position in Cartesian world coordinates
            target_position (np.array[float[3]]): position of the target (focus) point in Cartesian world coordinates
            up_vector (np.array[float[3]]): up vector of the camera in Cartesian world coordinates

        Returns:
            np.array[float[4,4]]: the view matrix
        """
        pass

    def compute_view_matrix_from_ypr(self, target_position, distance, yaw, pitch, roll, up_axis_index=2):
        """Compute the view matrix from the yaw, pitch, and roll angles.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            target_position (np.array[float[3]]): target focus point in Cartesian world coordinates
            distance (float): distance from eye to focus point
            yaw (float): yaw angle in radians left/right around up-axis
            pitch (float): pitch in radians up/down.
            roll (float): roll in radians around forward vector
            up_axis_index (int): either 1 for Y or 2 for Z axis up.

        Returns:
            np.array[float[4,4]]: the view matrix
        """
        pass

    def compute_projection_matrix(self, left, right, bottom, top, near, far):
        """Compute the orthographic projection matrix.

        The projection matrix is the 4x4 matrix that maps from the camera/eye coordinates to clipped coordinates.
        It is applied after the view matrix.

        There are 2 projection matrices:
        * orthographic projection
        * perspective projection

        For the perspective projection, see `computeProjectionMatrixFOV(self)`.

        Args:
            left (float): left screen (canvas) coordinate
            right (float): right screen (canvas) coordinate
            bottom (float): bottom screen (canvas) coordinate
            top (float): top screen (canvas) coordinate
            near (float): near plane distance
            far (float): far plane distance

        Returns:
            np.array[float[4,4]]: the perspective projection matrix
        """
        pass

    def compute_projection_matrix_fov(self, fov, aspect, near, far):
        """Compute the perspective projection matrix using the field of view (FOV).

        Args:
            fov (float): field of view
            aspect (float): aspect ratio
            near (float): near plane distance
            far (float): far plane distance

        Returns:
            np.array[float[4,4]]: the perspective projection matrix
        """
        pass

    def get_camera_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                         light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                         light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_camera_image` API will return a RGB image, a depth buffer and a segmentation mask buffer with body
        unique ids of visible objects for each pixel.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer
            flags (int): flags

        Returns:
            int: width image resolution in pixels (horizontal)
            int: height image resolution in pixels (vertical)
            np.array[int[width, height, 4]]: RBGA pixels (each pixel is in the range [0..255] for each channel).
            np.array[float[width, heigth]]: Depth buffer.
            np.array[int[width, height]]: Segmentation mask buffer. For each pixels the visible object unique id.
        """
        pass

    def get_rgba_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                       light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                       light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_rgba_image` API will return a RGBA image.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.array[int[width, height, 4]]: RBGA pixels (each pixel is in the range [0..255] for each channel).
        """
        pass

    def get_depth_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                        light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                        light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_depth_image` API will return a depth buffer.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.array[float[width, heigth]]: Depth buffer.
        """
        pass

    def get_segmentation_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                               light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                               light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_segmentation_image` API will return a segmentation mask buffer with body unique ids of visible objects
        for each pixel.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer
            flags (int): flags

        Returns:
            np.array[int[width, height]]: Segmentation mask buffer. For each pixels the visible object unique id.
        """
        pass

    ##############
    # Collisions #
    ##############

    def create_collision_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), height=1., filename=None,
                               mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1,
                               collision_frame_position=None, collision_frame_orientation=None):
        """
        Create collision shape in the simulator.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5)
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.array[float[3]], list/tuple of 3 floats): only for GEOM_BOX.
            height (float): only for GEOM_CAPSULE, GEOM_CYLINDER (length = height).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.array[float[3]], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.array[float[3]], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            flags (int): unused / to be decided
            collision_frame_position (np.array[float[3]]): translational offset of the collision shape with respect to
                the link frame
            collision_frame_orientation (np.array[float[4]]): rotational offset (quaternion x,y,z,w) of the collision
                shape with respect to the link frame

        Returns:
            int: The return value is a non-negative int unique id for the collision shape or -1 if the call failed.
        """
        pass

    def get_collision_shape_data(self, object_id, link_id=-1):
        """
        Get the collision shape data associated with the specified object id and link id.

        Args:
            object_id (int): object unique id.
            link_id (int): link index or -1 for the base.

        Returns:
            int: object unique id.
            int: link id.
            int: geometry type; GEOM_BOX (=3), GEOM_SPHERE (=2), GEOM_CAPSULE (=7), GEOM_MESH (=5), GEOM_PLANE (=6)
            np.array[float[3]]: depends on geometry type:
                for GEOM_BOX: extents,
                for GEOM_SPHERE: dimensions[0] = radius,
                for GEOM_CAPSULE and GEOM_CYLINDER: dimensions[0] = height (length), dimensions[1] = radius.
                For GEOM_MESH: dimensions is the scaling factor.
            str: Only for GEOM_MESH: file name (and path) of the collision mesh asset.
            np.array[float[3]]: Local position of the collision frame with respect to the center of mass/inertial frame
            np.array[float[4]]: Local orientation of the collision frame with respect to the inertial frame
        """
        pass

    def get_overlapping_objects(self, aabb_min, aabb_max):
        """
        This query will return all the unique ids of objects that have Axis Aligned Bounding Box (AABB) overlap with
        a given axis aligned bounding box. Note that the query is conservative and may return additional objects that
        don't have actual AABB overlap. This happens because the acceleration structures have some heuristic that
        enlarges the AABBs a bit (extra margin and extruded along the velocity vector).

        Args:
            aabb_min (np.array[float[3]]): minimum coordinates of the aabb
            aabb_max (np.array[float[3]]): maximum coordinates of the aabb

        Returns:
            list[int]: list of object unique ids.
        """
        pass

    def get_aabb(self, body_id, link_id=-1):
        """
        You can query the axis aligned bounding box (in world space) given an object unique id, and optionally a link
        index. (when you don't pass the link index, or use -1, you get the AABB of the base).

        Args:
            body_id (int): object unique id as returned by creation methods
            link_id (int): link index in range [0..`getNumJoints(..)]

        Returns:
            np.array[float[3]]: minimum coordinates of the axis aligned bounding box
            np.array[float[3]]: maximum coordinates of the axis aligned bounding box
        """
        pass

    def get_contact_points(self, body1, body2=None, link1_id=None, link2_id=None):
        """
        Returns the contact points computed during the most recent call to `step`.

        Args:
            body1 (int): only report contact points that involve body A
            body2 (int, None): only report contact points that involve body B. Important: you need to have a valid
                body A if you provide body B
            link1_id (int, None): only report contact points that involve link index of body A
            link2_id (int, None): only report contact points that involve link index of body B

        Returns:
            list:
                int: contact flag (reserved)
                int: body unique id of body A
                int: body unique id of body B
                int: link index of body A, -1 for base
                int: link index of body B, -1 for base
                np.array[float[3]]: contact position on A, in Cartesian world coordinates
                np.array[float[3]]: contact position on B, in Cartesian world coordinates
                np.array[float[3]]: contact normal on B, pointing towards A
                float: contact distance, positive for separation, negative for penetration
                float: normal force applied during the last `step`
                float: lateral friction force in the first lateral friction direction (see next returned value)
                np.array[float[3]]: first lateral friction direction
                float: lateral friction force in the second lateral friction direction (see next returned value)
                np.array[float[3]]: second lateral friction direction
        """
        # results: contact point, normal and penetration depth
        results = self.world.checkCollision()
        pass

    def get_closest_points(self, body1, body2, distance, link1_id=None, link2_id=None):
        """
        Computes the closest points, independent from `step`. This also lets you compute closest points of objects
        with an arbitrary separating distance. In this query there will be no normal forces reported.

        Args:
            body1 (int): only report contact points that involve body A
            body2 (int): only report contact points that involve body B. Important: you need to have a valid body A
                if you provide body B
            distance (float): If the distance between objects exceeds this maximum distance, no points may be returned.
            link1_id (int): only report contact points that involve link index of body A
            link2_id (int): only report contact points that involve link index of body B

        Returns:
            list:
                int: contact flag (reserved)
                int: body unique id of body A
                int: body unique id of body B
                int: link index of body A, -1 for base
                int: link index of body B, -1 for base
                np.array[float[3]]: contact position on A, in Cartesian world coordinates
                np.array[float[3]]: contact position on B, in Cartesian world coordinates
                np.array[float[3]]: contact normal on B, pointing towards A
                float: contact distance, positive for separation, negative for penetration
                float: normal force applied during the last `step`. Always equal to 0.
                float: lateral friction force in the first lateral friction direction (see next returned value)
                np.array[float[3]]: first lateral friction direction
                float: lateral friction force in the second lateral friction direction (see next returned value)
                np.array[float[3]]: second lateral friction direction
        """
        pass

    def ray_test(self, from_position, to_position):
        """
        Performs a single raycast to find the intersection information of the first object hit.

        Args:
            from_position (np.array[float[3]]): start of the ray in world coordinates
            to_position (np.array[float[3]]): end of the ray in world coordinates

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.array[float[3]]: hit position in Cartesian world coordinates
                np.array[float[3]]: hit normal in Cartesian world coordinates
        """
        pass

    def ray_test_batch(self, from_positions, to_positions, parent_object_id=None, parent_link_id=None):
        """Perform a batch of raycasts to find the intersection information of the first objects hit.

        This is similar to the ray_test, but allows you to provide an array of rays, for faster execution. The size of
        'rayFromPositions' needs to be equal to the size of 'rayToPositions'. You can one ray result per ray, even if
        there is no intersection: you need to use the objectUniqueId field to check if the ray has hit anything: if
        the objectUniqueId is -1, there is no hit. In that case, the 'hit fraction' is 1.

        Args:
            from_positions (np.array[float[N,3]]): list of start points for each ray, in world coordinates
            to_positions (np.array[float[N,3]]): list of end points for each ray in world coordinates
            parent_object_id (int): ray from/to is in local space of a parent object
            parent_link_id (int): ray from/to is in local space of a parent object

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.array[float[3]]: hit position in Cartesian world coordinates
                np.array[float[3]]: hit normal in Cartesian world coordinates
        """
        pass

    def set_collision_filter_group_mask(self, body_id, link_id, filter_group, filter_mask):
        """
        Enable/disable collision detection between groups of objects. Each body is part of a group. It collides with
        other bodies if their group matches the mask, and vise versa. The following check is performed using the group
        and mask of the two bodies involved. It depends on the collision filter mode.

        Args:
            body_id (int): unique id of the body to be configured
            link_id (int): link index of the body to be configured
            filter_group (int): bitwise group of the filter
            filter_mask (int): bitwise mask of the filter
        """
        pass

    def set_collision_filter_pair(self, body1, body2, link1=-1, link2=-1, enable=True):
        """
        Enable/disable collision between two bodies/links.

        Args:
            body1 (int): unique id of body A to be filtered
            body2 (int): unique id of body B to be filtered, A==B implies self-collision
            link1 (int): link index of body A
            link2 (int): link index of body B
            enable (bool): True to enable collision, False to disable collision
        """
        pass

    ###########################
    # Kinematics and Dynamics #
    ###########################

    def get_dynamics_info(self, body_id, link_id=-1):
        """
        Get dynamic information about the mass, center of mass, friction and other properties of the base and links.

        Args:
            body_id (int): body/object unique id.
            link_id (int): link/joint index or -1 for the base.

        Returns:
            float: mass in kg
            float: lateral friction coefficient
            np.array[float[3]]: local inertia diagonal. Note that links and base are centered around the center of mass
                and aligned with the principal axes of inertia.
            np.array[float[3]]: position of inertial frame in local coordinates of the joint frame
            np.array[float[4]]: orientation of inertial frame in local coordinates of joint frame
            float: coefficient of restitution
            float: rolling friction coefficient orthogonal to contact normal
            float: spinning friction coefficient around contact normal
            float: damping of contact constraints. -1 if not available.
            float: stiffness of contact constraints. -1 if not available.
        """
        body = self.world.getSkeleton(body_id).getBodyNode(link_id + 1)
        mass = body.getMass()

        dynamics = body.getShapeNode(0).getDynamicsAspect()
        friction = dynamics.getFrictionCoeff()

        ixx, iyy, izz, ixy, ixz, iyz = 0., 0., 0., 0., 0., 0.
        body.getMomentOfInertia(ixx, iyy, izz, ixy, ixz, iyz)
        inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]).reshape(3, 3)
        local_inertia_diag = np.linalg.eigh(inertia)[0]

        position = body.getLocalCOM().reshape(-1)
        orientation = get_quaternion_from_matrix(body.getWorldTransform()[:-1, :-1])

        restitution = dynamics.getRestitutionCoeff()
        rolling_friction = -1
        spinning_friction = -1
        damping = -1
        stiffness = -1

        return mass, friction, local_inertia_diag, position, orientation, restitution, rolling_friction, \
               spinning_friction, damping, stiffness

    def change_dynamics(self, body_id, link_id=-1, mass=None, lateral_friction=None, spinning_friction=None,
                        rolling_friction=None, restitution=None, linear_damping=None, angular_damping=None,
                        contact_stiffness=None, contact_damping=None, friction_anchor=None,
                        local_inertia_diagonal=None, inertia_position=None, inertia_orientation=None,
                        joint_damping=None, joint_friction=None):
        """
        Change dynamic properties of the given body (or link) such as mass, friction and restitution coefficients, etc.

        Args:
            body_id (int): object unique id, as returned by `load_urdf`, etc.
            link_id (int): link index or -1 for the base.
            mass (float): change the mass of the link (or base for link index -1)
            lateral_friction (float): lateral (linear) contact friction
            spinning_friction (float): torsional friction around the contact normal
            rolling_friction (float): torsional friction orthogonal to contact normal
            restitution (float): bounciness of contact. Keep it a bit less than 1.
            linear_damping (float): linear damping of the link (0.04 by default)
            angular_damping (float): angular damping of the link (0.04 by default)
            contact_stiffness (float): stiffness of the contact constraints, used together with `contact_damping`
            contact_damping (float): damping of the contact constraints for this body/link. Used together with
              `contact_stiffness`. This overrides the value if it was specified in the URDF file in the contact
              section.
            friction_anchor (int): enable or disable a friction anchor: positional friction correction (disabled by
              default, unless set in the URDF contact section)
            local_inertia_diagonal (np.array[float[3]]): diagonal elements of the inertia tensor. Note that the base
              and links are centered around the center of mass and aligned with the principal axes of inertia so
              there are no off-diagonal elements in the inertia tensor.
            inertia_position (np.array[float[3]]): new inertia position with respect to the link frame.
            inertia_orientation (np.array[float[4]]): new inertia orientation (expressed as a quaternion [x,y,z,w]
              with respect to the link frame.
            joint_damping (float): joint damping coefficient applied at each joint. This coefficient is read from URDF
              joint damping field. Keep the value close to 0.
              `joint_damping_force = -damping_coefficient * joint_velocity`.
            joint_friction (float): joint friction coefficient.
        """
        skeleton = self.world.getSkeleton(body_id)
        body = skeleton.getBodyNode(link_id + 1)
        joint = skeleton.getJoint(link_id + 1)

        if mass is not None:
            body.setMass(mass)

        if lateral_friction is not None:
            dynamics = body.getShapeNode(0).getDynamicsAspect()
            dynamics.setFrictionCoeff(lateral_friction)

        if restitution is not None:
            dynamics = body.getShapeNode(0).getDynamicsAspect()
            dynamics.setRestitutionCoeff(restitution)

        if local_inertia_diagonal is not None:
            ixx, iyy, izz = local_inertia_diagonal
            body.setMomentOfInertia(Ixx=ixx, Iyy=iyy, Izz=izz)

        if joint_damping is not None:
            joint.setDampingCoefficient(joint_damping)

        if joint_friction is not None:
            joint.setCoulombFriction(joint_friction)

    def calculate_jacobian(self, body_id, link_id, local_position, q, dq=None, des_ddq=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.array[float[3]]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.
            dq (np.array[float[N]]): joint velocities of size N, where N is the number of DoFs.
            des_ddq (np.array[float[N]]): desired joint accelerations of size N.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
        """
        skeleton = self.world.getSkeleton(body_id)
        body = skeleton.getBodyNode(link_id + 1)
        # TODO: set q?
        local_position = np.asarray(local_position).reshape(-1, 1)
        return skeleton.getWorldJacobian(node=body, localOffset=local_position)

    def calculate_mass_matrix(self, body_id, q):
        r"""
        Return the mass/inertia matrix :math:`H(q)`, which is used in the rigid-body equation of motion (EoM) in joint
        space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            body_id (int): body unique id.
            q (np.array[float[N]]): joint positions of size N, where N is the total number of DoFs.

        Returns:
            np.array[float[N,N]], np.array[float[6+N,6+N]]: inertia matrix
        """
        skeleton = self.world.getSkeleton(body_id)
        # TODO: set q?
        return skeleton.getAugMassMatrix()

    def calculate_inverse_kinematics(self, body_id, link_id, position, orientation=None, lower_limits=None,
                                     upper_limits=None, joint_ranges=None, rest_poses=None, joint_dampings=None,
                                     solver=None, q_curr=None, max_iters=None, threshold=None):
        """
        Compute the FULL Inverse kinematics; it will return a position for all the actuated joints.

        "You can compute the joint angles that makes the end-effector reach a given target position in Cartesian world
        space. Internally, Bullet uses an improved version of Samuel Buss Inverse Kinematics library. At the moment
        only the Damped Least Squares method with or without Null Space control is exposed, with a single end-effector
        target. Optionally you can also specify the target orientation of the end effector. In addition, there is an
        option to use the null-space to specify joint limits and rest poses. This optional null-space support requires
        all 4 lists (lower_limits, upper_limits, joint_ranges, rest_poses), otherwise regular IK will be used." [1]

        Args:
            body_id (int): body unique id, as returned by `load_urdf`, etc.
            link_id (int): end effector link index.
            position (np.array[float[3]]): target position of the end effector (its link coordinate, not center of mass
                coordinate!). By default this is in Cartesian world space, unless you provide `q_curr` joint angles.
            orientation (np.array[float[4]]): target orientation in Cartesian world space, quaternion [x,y,w,z]. If not
                specified, pure position IK will be used.
            lower_limits (np.array[float[N]], list of N floats): lower joint limits. Optional null-space IK.
            upper_limits (np.array[float[N]], list of N floats): upper joint limits. Optional null-space IK.
            joint_ranges (np.array[float[N]], list of N floats): range of value of each joint.
            rest_poses (np.array[float[N]], list of N floats): joint rest poses. Favor an IK solution closer to a
                given rest pose.
            joint_dampings (np.array[float[N]], list of N floats): joint damping factors. Allow to tune the IK solution
                using joint damping factors.
            solver (int): p.IK_DLS (=0) or p.IK_SDLS (=1), Damped Least Squares or Selective Damped Least Squares, as
                described in the paper by Samuel Buss "Selectively Damped Least Squares for Inverse Kinematics".
            q_curr (np.array[float[N]]): list of joint positions. By default PyBullet uses the joint positions of the
                body. If provided, the target_position and targetOrientation is in local space!
            max_iters (int): maximum number of iterations. Refine the IK solution until the distance between target
                and actual end effector position is below this threshold, or the `max_iters` is reached.
            threshold (float): residual threshold. Refine the IK solution until the distance between target and actual
                end effector position is below this threshold, or the `max_iters` is reached.

        Returns:
            np.array[float[N]]: joint positions (for each actuated joint).
        """
        pass

    def calculate_inverse_dynamics(self, body_id, q, dq, des_ddq):
        r"""
        Starting from the specified joint positions :math:`q` and velocities :math:`\dot{q}`, it computes the joint
        torques :math:`\tau` required to reach the desired joint accelerations :math:`\ddot{q}_{des}`. That is,
        :math:`\tau = ID(model, q, \dot{q}, \ddot{q}_{des})`.

        Specifically, it uses the rigid-body equation of motion in joint space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Normally, a more popular form of this equation of motion (in joint space) is given by:

        .. math:: H(q) \ddot{q} + S(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F

        which is the same as the first one with :math:`C = S\dot{q} + g(q) - J^T(q) F`. However, this last formulation
        is useful to understand what happens when we set some variables to 0.
        Assuming that there are no forces acting on the system, and giving desired joint accelerations of 0, this
        method will return :math:`\tau = S(q,\dot{q}) \dot{q} + g(q)`. If in addition joint velocities are also 0,
        it will return :math:`\tau = g(q)` which can for instance be useful for gravity compensation.

        For forward dynamics, which computes the joint accelerations given the joint positions, velocities, and
        torques (that is, :math:`\ddot{q} = FD(model, q, \dot{q}, \tau)`, this can be computed using
        :math:`\ddot{q} = H^{-1} (\tau - C)` (see also `computeFullFD`). For more information about different
        control schemes (position, force, impedance control and others), or about the formulation of the equation
        of motion in task/operational space (instead of joint space), check the references [1-4].

        Args:
            body_id (int): body unique id.
            q (np.array[float[N]]): joint positions
            dq (np.array[float[N]]): joint velocities
            des_ddq (np.array[float[N]]): desired joint accelerations

        Returns:
            np.array[float[N]]: joint torques computed using the rigid-body equation of motion

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            - [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            - [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            - [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        skeleton = self.world.getSkeleton(body_id)
        c = skeleton.getCoriolisAndGravityForces().reshape(-1)  # (M,)
        H = skeleton.getAugMassMatrix()  # (M,M)
        return H.dot(des_ddq) + c

    def calculate_forward_dynamics(self, body_id, q, dq, torques):
        r"""
        Given the specified joint positions :math:`q` and velocities :math:`\dot{q}`, and joint torques :math:`\tau`,
        it computes the joint accelerations :math:`\ddot{q}`. That is, :math:`\ddot{q} = FD(model, q, \dot{q}, \tau)`.

        Specifically, it uses the rigid-body equation of motion in joint space given by (see [1]):

        .. math:: \ddot{q} = H(q)^{-1} (\tau - C(q,\dot{q}))

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Normally, a more popular form of this equation of motion (in joint space) is given by:

        .. math:: H(q) \ddot{q} + S(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F

        which is the same as the first one with :math:`C = S\dot{q} + g(q) - J^T(q) F`. However, this last formulation
        is useful to understand what happens when we set some variables to 0.
        Assuming that there are no forces acting on the system, and giving desired joint torques of 0, this
        method will return :math:`\ddot{q} = - H(q)^{-1} (S(q,\dot{q}) \dot{q} + g(q))`. If in addition
        the joint velocities are also 0, it will return :math:`\ddot{q} = - H(q)^{-1} g(q)` which are
        the accelerations due to gravity.

        For inverse dynamics, which computes the joint torques given the joint positions, velocities, and
        accelerations (that is, :math:`\tau = ID(model, q, \dot{q}, \ddot{q})`, this can be computed using
        :math:`\tau = H(q)\ddot{q} + C(q,\dot{q})`. For more information about different
        control schemes (position, force, impedance control and others), or about the formulation of the equation
        of motion in task/operational space (instead of joint space), check the references [1-4].

        Args:
            body_id (int): unique body id.
            q (np.array[float[N]]): joint positions
            dq (np.array[float[N]]): joint velocities
            torques (np.array[float[N]]): desired joint torques

        Returns:
            np.array[float[N]]: joint accelerations computed using the rigid-body equation of motion

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            - [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            - [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            - [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        skeleton = self.world.getSkeleton(body_id)
        c = skeleton.getCoriolisAndGravityForces().reshape(-1)  # (M,)
        # H_inv = skeleton.getInvMassMatrix()  # (M,M)
        H = skeleton.getAugMassMatrix()  # (M,M)
        H_inv = np.linalg.inv(H)
        return H_inv.dot((torques - c))

    #########
    # Debug #
    #########

    def add_user_debug_line(self, from_pos, to_pos, rgb_color=None, width=None, lifetime=None, parent_object_id=None,
                            parent_link_id=None, line_id=None):
        """Add a user debug line in the simulator.

        You can add a 3d line specified by a 3d starting point (from) and end point (to), a color [red,green,blue],
        a line width and a duration in seconds.

        Args:
            from_pos (np.array[float[3]]): starting point of the line in Cartesian world coordinates
            to_pos (np.array[float[3]]): end point of the line in Cartesian world coordinates
            rgb_color (np.array[float[3]]): RGB color (each channel in range [0,1])
            width (float): line width (limited by OpenGL implementation).
            lifetime (float): use 0 for permanent line, or positive time in seconds (afterwards the line with be
                removed automatically)
            parent_object_id (int): draw line in local coordinates of a parent object.
            parent_link_id (int): draw line in local coordinates of a parent link.
            line_id (int): replace an existing line item (to avoid flickering of remove/add).

        Returns:
            int: unique user debug line id.
        """
        pass

    def add_user_debug_text(self, text, position, rgb_color=None, size=None, lifetime=None, orientation=None,
                            parent_object_id=None, parent_link_id=None, text_id=None):
        """
        Add 3D text at a specific location using a color and size.

        Args:
            text (str): text.
            position (np.array[float[3]]): 3d position of the text in Cartesian world coordinates.
            rgb_color (list/tuple of 3 floats): RGB color; each component in range [0..1]
            size (float): text size
            lifetime (float): use 0 for permanent text, or positive time in seconds (afterwards the text with be
                removed automatically)
            orientation (np.array[float[4]]): By default, debug text will always face the camera, automatically
                rotation. By specifying a text orientation (quaternion), the orientation will be fixed in world space
                or local space (when parent is specified). Note that a different implementation/shader is used for
                camera facing text, with different appearance: camera facing text uses bitmap fonts, text with
                specified orientation uses TrueType font.
            parent_object_id (int): draw text in local coordinates of a parent object.
            parent_link_id (int): draw text in local coordinates of a parent link.
            text_id (int): replace an existing text item (to avoid flickering of remove/add).

        Returns:
            int: unique user debug text id.
        """
        pass

    def add_user_debug_parameter(self, name, min_range, max_range, start_value):
        """
        Add custom sliders to tune parameters.

        Args:
            name (str): name of the parameter.
            min_range (float): minimum value.
            max_range (float): maximum value.
            start_value (float): starting value.

        Returns:
            int: unique user debug parameter id.
        """
        pass

    def read_user_debug_parameter(self, parameter_id):
        """
        Read the value of the parameter / slider.

        Args:
            parameter_id: unique user debug parameter id.

        Returns:
            float: reading of the parameter.
        """
        pass

    def remove_user_debug_item(self, item_id):
        """
        Remove the specified user debug item (line, text, parameter) from the simulator.

        Args:
            item_id (int): unique id of the debug item to be removed (line, text etc)
        """
        pass

    def remove_all_user_debug_items(self):
        """
        Remove all user debug items from the simulator.
        """
        pass

    def set_debug_object_color(self, object_id, link_id, rgb_color=(1, 0, 0)):
        """
        Override the color of a specific object and link.

        Args:
            object_id (int): unique object id.
            link_id (int): link id.
            rgb_color (float[3]): RGB debug color.
        """
        pass

    def add_user_data(self, object_id, key, value):
        """
        Add user data (at the moment text strings) attached to any link of a body. You can also override a previous
        given value. You can add multiple user data to the same body/link.

        Args:
            object_id (int): unique object/link id.
            key (str): key string.
            value (str): value string.

        Returns:
            int: user data id.
        """
        pass

    def num_user_data(self, object_id):
        """
        Return the number of user data associated with the specified object/link id.

        Args:
            object_id (int): unique object/link id.

        Returns:
            int: the number of user data
        """
        pass

    def get_user_data(self, user_data_id):
        """
        Get the specified user data value.

        Args:
            user_data_id (int): unique user data id.

        Returns:
            str: value string.
        """
        pass

    def get_user_data_id(self, object_id, key):
        """
        Get the specified user data id.

        Args:
            object_id (int): unique object/link id.
            key (str): key string.

        Returns:
            int: user data id.
        """
        pass

    def get_user_data_info(self, object_id, index):
        """
        Get the user data info associated with the given object and index.

        Args:
            object_id (int): unique object id.
            index (int): index (should be between [0, self.num_user_data(object_id)]).

        Returns:
            int: user data id.
            str: key.
            int: body id.
            int: link index
            int: visual shape index.
        """
        pass

    def remove_user_data(self, user_data_id):
        """
        Remove the specified user data.

        Args:
            user_data_id (int): user data id.
        """
        pass

    def sync_user_data(self):
        """
        Synchronize the user data.
        """
        pass

    def configure_debug_visualizer(self, flag, enable):
        """Configure the debug visualizer camera.

        Configure some settings of the built-in OpenGL visualizer, such as enabling or disabling wireframe,
        shadows and GUI rendering.

        Args:
            flag (int): The feature to enable or disable, such as
                        COV_ENABLE_WIREFRAME (=3): show/hide the collision wireframe
                        COV_ENABLE_SHADOWS (=2): show/hide shadows
                        COV_ENABLE_GUI (=1): enable/disable the GUI
                        COV_ENABLE_VR_PICKING (=5): enable/disable VR picking
                        COV_ENABLE_VR_TELEPORTING (=4): enable/disable VR teleporting
                        COV_ENABLE_RENDERING (=7): enable/disable rendering
                        COV_ENABLE_TINY_RENDERER (=12): enable/disable tiny renderer
                        COV_ENABLE_VR_RENDER_CONTROLLERS (=6): render VR controllers
                        COV_ENABLE_KEYBOARD_SHORTCUTS (=9): enable/disable keyboard shortcuts
                        COV_ENABLE_MOUSE_PICKING (=10): enable/disable mouse picking
                        COV_ENABLE_Y_AXIS_UP (Z is default world up axis) (=11): enable/disable Y axis up
                        COV_ENABLE_RGB_BUFFER_PREVIEW (=13): enable/disable RGB buffer preview
                        COV_ENABLE_DEPTH_BUFFER_PREVIEW (=14): enable/disable Depth buffer preview
                        COV_ENABLE_SEGMENTATION_MARK_PREVIEW (=15): enable/disable segmentation mark preview
            enable (bool): False (disable) or True (enable)
        """
        pass

    def get_debug_visualizer(self):
        """Get information about the debug visualizer camera.

        Returns:
            float: width of the visualizer camera
            float: height of the visualizer camera
            np.array[float[4,4]]: view matrix [4,4]
            np.array[float[4,4]]: perspective projection matrix [4,4]
            np.array[float[3]]: camera up vector expressed in the Cartesian world space
            np.array[float[3]]: forward axis of the camera expressed in the Cartesian world space
            np.array[float[3]]: This is a horizontal vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            np.array[float[3]]: This is a vertical vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            float: yaw angle (in radians) of the camera, in Cartesian local space coordinates
            float: pitch angle (in radians) of the camera, in Cartesian local space coordinates
            float: distance between the camera and the camera target
            np.array[float[3]]: target of the camera, in Cartesian world space coordinates
        """
        pass

    def reset_debug_visualizer(self, distance, yaw, pitch, target_position):
        """Reset the debug visualizer camera.

        Reset the 3D OpenGL debug visualizer camera distance (between eye and camera target position), camera yaw and
        pitch and camera target position

        Args:
            distance (float): distance from eye to camera target position
            yaw (float): camera yaw angle (in radians) left/right
            pitch (float): camera pitch angle (in radians) up/down
            target_position (np.array[float[3]]): target focus point of the camera
        """
        pass

    ############################
    # Events (mouse, keyboard) #
    ############################

    def get_keyboard_events(self):
        """Get the key events.

        Returns:
            dict: {keyId: keyState}
                * `keyID` is an integer (ascii code) representing the key. Some special keys like shift, arrows,
                and others are are defined in pybullet such as `B3G_SHIFT`, `B3G_LEFT_ARROW`, `B3G_UP_ARROW`,...
                * `keyState` is an integer. 3 if the button has been pressed, 1 if the key is down, 2 if the key has
                been triggered.
        """
        pass

    def get_mouse_events(self):
        """Get the mouse events.

        Returns:
            list of mouse events:
                eventType (int): 1 if the mouse is moving, 2 if a button has been pressed or released
                mousePosX (float): x-coordinates of the mouse pointer
                mousePosY (float): y-coordinates of the mouse pointer
                buttonIdx (int): button index for left/middle/right mouse button. It is -1 if nothing,
                                 0 if left button, 1 if scroll wheel (pressed), 2 if right button
                buttonState (int): 0 if nothing, 3 if the button has been pressed, 4 is the button has been released,
                                   1 if the key is down (never observed), 2 if the key has been triggered (never
                                   observed).
        """
        pass

    def get_mouse_and_keyboard_events(self):
        """Get the mouse and key events.

        Returns:
            list: list of mouse events
            dict: dictionary of key events
        """
        pass


# Tests
if __name__ == '__main__':
    import os
    from itertools import count

    # create simulation
    sim = Dart(render=False)

    skeleton_id = sim.load_urdf(os.path.dirname(__file__) + '/../robots/urdfs/cubli/cubli.urdf')
    # skeleton_id = sim.load_urdf(os.path.dirname(__file__) + '/../robots/urdfs/rrbot/pendulum.urdf')
    skeleton = sim.world.getSkeleton(skeleton_id)
    print("World name: {}".format(sim.world.getName()))
    print("Gravity: {}".format(sim.world.getGravity()))
    print("Skeleton name from world: {}".format(sim.world.getSkeleton(0).getName()))
    print("Skeleton name: {}".format(skeleton.getName()))
    print("DoFs: {}".format(skeleton.getNumDofs()))
    print("Num of Body nodes: {}".format(skeleton.getNumBodyNodes()))
    print("Num of Joints: {}".format(skeleton.getNumJoints()))
    print("Positions: {}".format(skeleton.getPositions()))
    base = skeleton.getRootBodyNode()
    print("Base name: {}".format(base.getName()))
    print("Transform: {}".format(base.getTransform()))
    body1 = skeleton.getBodyNode(0)
    print("body1 name: {}".format(body1.getName()))
    body2 = skeleton.getBodyNode(1)
    print("body2 name: {}".format(body2.getName()))
    print("body2 transform: {}".format(body2.getTransform()))
    # print("body2 world transform: {}".format(body2.getWorldTransform()))
    # print("body2 relative transform: {}".format(body2.getRelativeTransform()))
    # sim.render()

    for dof in range(skeleton.getNumDofs()):
        skeleton.setPosition(dof, np.pi/5)

    for joint_id in range(skeleton.getNumJoints()):
        joint = skeleton.getJoint(joint_id)
        print("\njoint id: {}".format(joint_id))
        print("joint name: {}".format(joint.getName()))
        print("joint type: {}".format(joint.getType()))
        if hasattr(joint, 'getAxis'):
            print("joint axis: {}".format(joint.getAxis()))
        print("joint num DoFs: {}".format(joint.getNumDofs()))
        print("joint position: {}".format(joint.getPosition(0)))
        print("skeleton joint position: ", skeleton.getPosition(joint_id))

