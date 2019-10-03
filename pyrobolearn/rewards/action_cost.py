# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the costs used on actions.
"""

from abc import ABCMeta
import numpy as np

import pyrobolearn as prl
from pyrobolearn.rewards.cost import Cost


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ActionCost(Cost):
    r"""(Abstract) Action Cost."""
    __metaclass__ = ABCMeta
    pass


class ActionDifferenceCost(ActionCost):
    r"""Action Difference Cost

    This computes the difference between two actions:

    .. math:: \text{cost} = || a_{t-1} - a_{t} ||^2

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, action):
        """
        Initialize the action difference cost.

        Args:
            action (Action): action instance.
        """
        super(ActionDifferenceCost, self).__init__()
        if not isinstance(action, prl.actions.Action):
            raise TypeError("Expecting the given 'action' to be an instance of `Action`, instead got: "
                            "{}".format(type(action)))
        self.action = action

    def _compute(self):
        data = self.action.merged_data
        prev_data = self.action.merged_data
        return np.sum([- np.sum((curr - prev)**2) for curr, prev in zip(data, prev_data)])
