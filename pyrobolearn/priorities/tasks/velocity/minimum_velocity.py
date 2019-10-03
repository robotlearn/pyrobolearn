# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the minimum velocity task.

The minimum velocity task minimizes the joint velocities, that is it minimizes:

.. math:: ||\dot{q}||^2,

which is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=I`, :math:`x=\dot{q}`,
and :math:`b=0`.

This minimum velocity task is often used in conjunction with other tasks such as `PosturalTask` or `CartesianTask`.

Note that this can also be achieved with the `tasks.velocity.PosturalTask` by setting the desired joint velocities to
zeros and not providing the desired joint positions.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MinVelocityTask(JointVelocityTask):
    r"""Minimum Velocity Task

    The minimum velocity task minimizes the joint velocities, that is it minimizes:

    .. math:: ||\dot{q}||^2,

    which is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=I`, :math:`x=\dot{q}`,
    and :math:`b=0`.

    This minimum velocity task is often used in conjunction with other tasks such as `PosturalTask` or `CartesianTask`.
    Note that this task can also be achieved with the `tasks.velocity.PosturalTask` by setting the desired joint
    velocities to zeros and not providing the desired joint positions.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            weight (float, np.array[float[N,N]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        # the variables A and b are initialized by default to be A=I and b=0
        super(MinVelocityTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # first update
        self.update()
