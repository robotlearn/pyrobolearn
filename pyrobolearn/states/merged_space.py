# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the merged space.

This is a space that merges the various spaces together based on each dimension.
"""

import copy
import numpy as np
import gym


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MergedSpace(gym.spaces.Space):
    r"""Merged Space.


    """

    def __init__(self, spaces):
        self.spaces = spaces
        super(MergedSpace, self).__init__(shape=None, dtype=None)

    @property
    def spaces(self):
        return self._spaces

    @spaces.setter
    def spaces(self, spaces):
        if isinstance(spaces, gym.spaces.Space):
            spaces = [spaces]
        if not isinstance(spaces, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given spaces to be a list/tuple/np.ndarray of `gym.spaces.Space`, but "
                            "got instead: {}".format(type(spaces)))
        for i, space in enumerate(spaces):
            if not isinstance(space, gym.spaces.Space):
                raise TypeError("Expecting the {}th item to be an instance of `gym.spaces.Space`, but got instead: "
                                "{}".format(i, type(space)))
        self._spaces = spaces

    def sample(self):
        """
        Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError

    def seed(self, seed):
        """Set the seed for this space's pseudo-random number generator."""
        if seed is not None:
            for space in self.spaces:
                space.seed(seed)

    def contains(self, x):
        """
        Return boolean specifying if x is a valid member of this space.
        """
        raise NotImplementedError
