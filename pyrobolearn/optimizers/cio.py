# This file implements the 'Contact Invariant Optimization' framework developed by Igor Mordatch.
# Ref: "Automated Discovery and Learning of Complex Movement Behaviors" (PhD thesis), Mordatch, 2015
# See also: presentation given CS294

import numpy as np
from scipy.interpolate as interp1d


class CIO(object):
    r"""
    The Contact Invariant Optimization (CIO) algorithm [1] consists to minimize the following cost:

    .. math::

        s* = argmin_s L(s)
           = argmin_s L_{CI}(s) + L_{physics}(s) + L_{task}(s) + L_{hint}(s)

    where :math:`s` is the state which contains :math:`x_k`, :math:`\dot{x}_k`, and :math:`c_k` for each phase/interval
    :math:`k`. The vector :math:`x_k` contains the torso and end-effector position and orientation, while
    the :math:`c_k` vector represents the auxiliary contact variables.

    Here is what each term represents in the total cost:
    - :math:`L_{CI}` is the contact invariant cost
    - :math:`L_{physics}` penalizes physics violation
    - :math:`L_{task}` describes the task objectives (i.e. high-level goals of the movement)
    - :math:`L_{hint}` provides hints to accelerate the optimization. This term is optional.

    The CIO consists of 3 phases:
    1. only :math:`L_{task}` is enabled
    2. All 4 terms (:math:`L_{task}`, :math:`L_{physics}`, :math:`L_{CI}`, :math:`L_{hint}`) are enabled but with
       :math:`L_{physics}` down-weighted by 0.1
    3. :math:`L_{task}`, :math:`L_{physics}`, and :math:`L_{CI}` are fully enabled

    Note that the solution obtained at the end of each phase is perturbed with small zero-mean Gaussian noise to
    break any symmetries, and used to initialize the next phase.

    From the optimized state :math:`s^*`, the optimal joints :math:`q^*` at each time step can be computed (using IK).
    Then, a PD controller can be used to move the joints to their desired configuration.

    Note that the framework do not take into account any sensory feedbacks.

    References:
        [1] "Automated Discovery and Learning of Complex Movement Behaviors" (PhD thesis), Mordatch, 2015
    """

    def __init__(self, robot, T, num_interval=20):
        # TODO: think about optimizing multiple actors
        self.robot = robot
        self.K = num_interval

        x = np.linspace(0, T, self.K)
        y = np.array(range(1, self.K+1))
        self.phase = scipy.interpolate.interp1d(x, y, kind='zero')

    def get_phase_index(self, t):
        return self.phase(t)

    def compute_state(self):
        base_pos = self.robot.getBasePosition()
        base_quat = self.robot.getBaseOrientation()
        end_effector_pos = self.robot.getEndEffectorPositions()
        end_effector_quat = self.robot.getEndEffectorOrientations()

    def optimize(self):
        pass
