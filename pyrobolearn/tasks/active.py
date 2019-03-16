#!/usr/bin/env python
"""Define the active learning task.
"""

from imitation import ILTask

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ALTask(ILTask):
    r"""Active Learning Task

    Task used for active learning. This is pretty similar to imitation learning with the exception that the policy
    can decide to interact with the user (to ask for more demonstrations for instance). Thus the output of the policy
    is interpreted by the interface.
    """

    def __init__(self, environment, policies, interface=None, recorder=None):
        """Initialize the active learning task.

        Args:
            environment (Env): environment of the task (which contains the world)
            policies (Policy): rl to be trained by the task
            interface (Interface): input/output interface that allows to interact with the world.
            recorder (Recorder): if the interface doesn't have a recorder, it can be supplemented here.
                If the interface doesn't have a recorder, and it is not specified, it will create a recorder that
                record the states and actions (inferred from the rl).
        """
        super(ALTask, self).__init__(environment, policies, interface, recorder)
