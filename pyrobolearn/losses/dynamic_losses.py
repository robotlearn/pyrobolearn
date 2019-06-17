#!/usr/bin/env python
"""Defines the dynamic losses in RL.

That is, the losses that are used with dynamic models.
"""

import torch

from pyrobolearn.losses import BatchLoss
from pyrobolearn.dynamics import DynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DynamicL2Loss(BatchLoss):
    r"""Dynamic L2 loss.

    This loss computes the Frobenius norm between the prediction of a dynamic model and the next states that are
    present in the batch.
    """

    def __init__(self, dynamic_model):
        r"""
        Initialize the dynamic L2 loss.

        Args:
            dynamic_model (DynamicModel): dynamic model.
        """
        # check the given dynamic model
        super(DynamicL2Loss, self).__init__()
        if not isinstance(dynamic_model, DynamicModel):
            raise TypeError("Expecting the given 'dynamic_model' to be an instance of `DynamicModel`, instead got: "
                            "{}".format(type(dynamic_model)))
        self._dynamic_model = dynamic_model

    def _compute(self, batch):
        """
        Compute the frobenius norm (i.e. L2-norm).

        Args:
            batch (Batch): batch containing the states, actions, and next states.

        Returns:
            torch.Tensor: scalar loss value.
        """
        # predict the next states using the dynamic model
        prediction = self._dynamic_model.predict(states=batch['states'], actions=batch['actions'], deterministic=False,
                                                 to_numpy=False, set_state_data=False)
        # compute and return the loss
        return 0.5 * (batch['next_states'] - prediction).pow(2).mean()
