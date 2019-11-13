#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define some body terminal conditions for the environment.
"""

from abc import ABCMeta
import numpy as np

from pyrobolearn.robots.base import Body
from pyrobolearn.terminal_conditions.terminal_condition import TerminalCondition
from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_matrix_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BodyCondition(TerminalCondition):
    r"""Body Terminal Condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the dimensions of the body state are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the dimension of the body state is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)

    Body state includes the position and orientation for instance.
    """
    __metaclass__ = ABCMeta

    def __init__(self, body, bounds, dim=None, all=False, stay=False, out=False):
        """
        Initialize the body terminal condition.

        Args:
            body (Body): body instance
            dim (None, int, int[3]): dimensions that we should consider when looking at the bounds. If None, it will
                consider all 3 dimensions. If one dimension is provided it will only check along that dimension. If
                a np.array of 0 and 1 is provided, it will consider the dimensions that are equal to 1. Thus, [1,0,1]
                means to consider the bounds along the x and z axes.
            all (bool): this is only used if they are multiple dimensions. if True, all the dimensions of the state
                are checked if they are inside or outside the bounds depending on the other parameters. if False, any
                dimensions will be checked.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the state
                leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the state leaves the bounds, it results in a success.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
        """
        super(BodyCondition, self).__init__()
        self.body = body
        self.dim = dim
        self.bounds = bounds
        self._out = bool(out)
        self._stay = bool(stay)
        self._all = bool(all)

    ##############
    # Properties #
    ##############

    @property
    def body(self):
        """Return the body instance."""
        return self._body

    @body.setter
    def body(self, body):
        """Set the body instance."""
        if not isinstance(body, Body):
            raise TypeError("Expecting the given 'body' to be an instance of `Body`, instead got: "
                            "{}".format(type(body)))
        self._body = body

    @property
    def dim(self):
        """Return the dimension(s)."""
        return self._dim

    @dim.setter
    def dim(self, dim):
        """Set the dimensions."""
        if dim is not None:
            if not isinstance(dim, (int, np.ndarray)):
                if isinstance(dim, (list, tuple)):
                    dim = np.asarray(dim)
                else:
                    raise TypeError("Expecting the given 'dim' to be an int or an np.array of 3 int, but got instead: "
                                    "{}".format(type(dim)))
            if isinstance(dim, np.ndarray):
                if dim.size != 3:
                    raise ValueError("Expecting the given 'dim' np.array to be of size 3, but got instead a size of: "
                                     "{}".format(dim.size))
                dim = np.array([bool(d) for d in dim])
        self._dim = dim

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.body.simulator

    ###########
    # Methods #
    ###########

    def _check_bounds(self, bounds):
        """Check the given bounds."""
        # check the type of the bounds
        if not isinstance(bounds, (tuple, list, np.ndarray)):
            raise TypeError("Expecting the given bounds to be a tuple/list/np.ndarray of float, instead got: "
                            "{}".format(type(bounds)))

        # check that the bounds have a length of 2 (i.e. lower and upper bounds)
        if len(bounds) != 2:
            raise ValueError("Expecting the bounds to be of length 2 (i.e. lower and upper bounds), instead got a "
                             "length of {}".format(len(bounds)))

        # if one of the bounds is None, raise error
        if bounds[0] is None or bounds[1] is None:
            raise ValueError("Expecting the bounds to not have None, but got: {}".format(bounds))

        # reshape bounds if necessary
        bounds = np.asarray(bounds).reshape(2, -1)

        if self.dim is None:
            if bounds.shape[1] != 3:
                raise ValueError("Expecting the bounds to be of shape (2,3) but got instead a shape of: "
                                 "{}".format(bounds.shape))
        else:
            if isinstance(self.dim, int) and bounds.shape[1] != 1:
                raise ValueError("If you specified one dimension, we expect the shape of the bounds to be (2,1), but "
                                 "got instead a shape of: {}".format(bounds.shape))
            elif isinstance(self.dim, np.ndarray):
                if bounds.shape[1] != len(self.dim[self.dim]):
                    raise ValueError("Expecting each bound to have the same number of elements than the elements that "
                                     "are not zero in the given 'dim' attribute")
        return bounds

    def check(self):
        """
        Check if the terminating condition has been fulfilled, and return True or False accordingly
        """
        states = self._get_states()

        if self._all:  # all the dimension states
            if self._out:  # are outside a certain bounds
                if self._stay:  # and must stay outside these ones.
                    if np.any((self.bounds[0] <= states) & (states <= self.bounds[1])):  # one dimension went inside
                        self._btype = False     # failure
                        self._over = True       # it is over
                    else:  # they are still all outside
                        self._btype = True      # success
                        self._over = False      # it is not over
                else:  # and must go inside these ones
                    if np.all((self.bounds[0] <= states) & (states <= self.bounds[1])):  # they all went inside
                        self._btype = True      # success
                        self._over = True       # it is over
                    else:  # they are some still left outside
                        self._btype = False     # failure
                        self._over = False      # it is not over
            else:  # are inside a certain bounds
                if self._stay:  # and must stay inside these ones.
                    if not np.all((self.bounds[0] <= states) &
                                  (states <= self.bounds[1])):  # one dimension went outside
                        self._btype = False  # failure
                        self._over = True    # it is over
                    else:  # they are still all inside
                        self._btype = True  # success
                        self._over = False  # it is not over
                else:  # and must go outside these ones.
                    if np.any((self.bounds[0] <= states) & (states <= self.bounds[1])):  # they are still some inside
                        self._btype = False  # failure
                        self._over = False   # it is not over
                    else:  # they are all outside
                        self._btype = True   # success
                        self._over = True    # it is over

        else:  # any of the dimension states
            if self._out:  # is outside a certain bounds
                if self._stay:  # and still stays outside these ones.
                    if not np.all((self.bounds[0] <= states) &
                                  (states <= self.bounds[1])):  # at least one dim. is still outside
                        self._btype = True   # success
                        self._over = False   # it is not over
                    else:  # they are all inside
                        self._btype = False   # failure
                        self._over = True     # it is over
                else:  # and one must at least go inside these ones
                    if np.any((self.bounds[0] <= states) & (states <= self.bounds[1])):  # at least one state is inside
                        self._btype = True  # success
                        self._over = True   # it is over
                    else:  # they are still all outside
                        self._btype = False  # failure
                        self._over = False   # it is not over
            else:  # is inside a certain bounds
                if self._stay:  # and must stay inside these ones.
                    if np.any((self.bounds[0] <= states) &
                              (states <= self.bounds[1])):  # at least one state is still inside
                        self._btype = True   # success
                        self._over = False   # it is not over
                    else:  # they are all outside
                        self._btype = False  # failure
                        self._over = True    # it is over
                else:  # and must go outside these ones.
                    if np.all((self.bounds[0] <= states) & (states <= self.bounds[1])):  # they are all inside
                        self._btype = False  # failure
                        self._over = False   # it is not over
                    else:  # at least one went outside
                        self._btype = True   # success
                        self._over = True    # it is over

        return self._over

    def _get_states(self):
        """Get the base states. Has to be implemented in the child class."""
        raise NotImplementedError


class PositionCondition(BodyCondition):
    r"""World position terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the dimensions of the body position state are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the dimension of the body position state is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, body, bounds=(None, None), dim=None, out=False, stay=False, all=False):
        """
        Initialize the world position terminal condition.

        Args:
            body (Body): body instance.
            bounds (tuple of 2 float / np.array[3]): bounds on the body position.
            dim (None, int, int[3]): dimensions that we should consider when looking at the bounds. If None, it will
                consider all 3 dimensions. If one dimension is provided it will only check along that dimension. If
                a np.array of 0 and 1 is provided, it will consider the dimensions that are equal to 1. Thus, [1,0,1]
                means to consider the bounds along the x and z axes.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the position
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the position leaves the bounds, it results in a success.
            all (bool): this is only used if they are multiple dimensions. if True, all the dimensions of the state
                are checked if they are inside or outside the bounds depending on the other parameters. if False, any
                dimensions will be checked.
        """
        super(PositionCondition, self).__init__(body, bounds=bounds, dim=dim, out=out, stay=stay, all=all)
        # check the bounds
        self.bounds = self._check_bounds(bounds=bounds)

    def _get_states(self):
        """Return the state."""
        position = self.body.position
        if self.dim is None:
            print(position)
            return position

        print(position[self.dim])
        return position[self.dim]


class OrientationCondition(BodyCondition):
    r"""World orientation terminal condition

    This terminal condition describes 8 cases (4 failure and 4 success cases):

    1. all the dimensions of the body orientation (expressed as roll-pitch-yaw angles) state are:
        1. in a certain bounds and must stay between these bounds. Once one gets out, the terminal condition is over,
           and results in a failure. (all=True, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once they all get out, the terminal condition is over,
           and results in a success. (all=True, out=False, stay=False)
        3. outside a certain bounds and must get in. Once they all get in, the terminal condition is over, and results
           in a success. (all=True, out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once one gets in, the terminal condition is over,
           and results in a failure. (all=True, out=True, stay=True)
    2. any of the dimension of the body orientation (expressed as roll-pitch-yaw angles) state is:
        1. in a certain bounds and must stay between these bounds. Once they all get out, the terminal condition is
           over, and results in a failure. (all=False, out=False, stay=True)
        2. in a certain bounds and must get out of these bounds. Once one gets out, the terminal condition is over,
           and results in a success. (all=False, out=False, stay=False)
        3. outside a certain bounds and must get in. Once one gets in, the terminal condition is over, and results in
           a success. (all=False ,out=True, stay=False)
        4. outside a certain bounds and must stay outside these ones. Once they all get in, the terminal condition is
           over, and results in a failure. (all=False, out=True, stay=True)
    """

    def __init__(self, body, bounds=(None, None), dim=None, out=False, stay=False, all=False):
        """
        Initialize the world orientation terminal condition.

        Args:
            body (Body): body instance.
            bounds (tuple of 2 float / np.array[3]): bounds on the body orientation expressed as roll-pitch-yaw angles
                or axis-angle if the :attr:`axis` is provided.
            dim (None, int, int[3]): dimensions that we should consider when looking at the bounds. If None, it will
                consider all 3 dimensions. If one dimension is provided it will only check along that dimension. If
                a np.array of 0 and 1 is provided, it will consider the dimensions that are equal to 1. Thus, [1,0,1]
                means to consider the bounds along the x (roll) and z (yaw) axes.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the orientation
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the orientation leaves the bounds, it results in a success.
            all (bool): this is only used if they are multiple dimensions. if True, all the dimensions of the state
                are checked if they are inside or outside the bounds depending on the other parameters. if False, any
                dimensions will be checked.
        """
        super(OrientationCondition, self).__init__(body, bounds=bounds, dim=dim, out=out, stay=stay, all=all)
        # check the bounds
        self.bounds = self._check_bounds(bounds=bounds)

    def _get_states(self):
        """Return the state."""
        orientation = get_rpy_from_quaternion(self.body.orientation)
        if self.dim is None:
            return orientation
        return orientation[self.dim]


class BaseOrientationAxisCondition(BodyCondition):
    r"""Base orientation axis terminal condition

    This uses the cosine similarity function by computing the angle between the given axis and one of the axis
    of the base orientation (i.e. one of the columns of the rotation matrix).

    This terminal condition describes 4 cases (2 failure and 2 success cases); the angle is in:

    1. in a certain bounds and must stay between these bounds. Once it gets out, the terminal condition is over,
       and results in a failure. (stay=True, out=False --> must stay in)
    2. in a certain bounds and must get out of these bounds. Once it gets out, the terminal condition is over,
       and results in a success. (stay=False, out=False --> must not stay in)
    3. outside a certain bounds and must get in. Once it gets in, the terminal condition is over, and results
       in a success. (stay=False, out=True --> must not stay out)
    4. outside a certain bounds and must stay outside these ones. Once it gets in, the terminal condition is over,
       and results in a failure. (stay=True, out=True --> must stay out)
    """

    def __init__(self, body, angle=0.85, axis=(0., 0., 1.), dim=2, stay=False, out=False):
        """
        Initialize the base orientation axis terminal condition.

        Args:
            body (Body): body instance.
            angle (float): angle bound.
            axis (tuple/list[float[3]], np.array[float[3]]): axis.
            dim (int): column that we should consider for the rotation matrix.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the orientation
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the orientation leaves the bounds, it results in a success.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
        """
        bounds = np.array([[angle], [1.1]])  # 1.1 is just to be sure
        super(BaseOrientationAxisCondition, self).__init__(body, bounds=bounds, dim=dim, stay=stay, out=out)
        self.axis = np.asarray(axis)

    def _get_states(self):
        """Return the state."""
        axis = get_matrix_from_quaternion(self.body.orientation)[self.dim]
        return np.dot(axis, self.axis)


class BaseHeightCondition(BodyCondition):
    r"""Base Height terminal condition

    This terminal condition describes 4 cases (2 failure and 2 success cases); the base height (i.e. z-position) state
    is:

    1. in a certain bounds and must stay between these bounds. Once it gets out, the terminal condition is over,
       and results in a failure. (stay=True, out=False --> must stay in)
    2. in a certain bounds and must get out of these bounds. Once it gets out, the terminal condition is over,
       and results in a success. (stay=False, out=False --> must not stay in)
    3. outside a certain bounds and must get in. Once it gets in, the terminal condition is over, and results
       in a success. (stay=False, out=True --> must not stay out)
    4. outside a certain bounds and must stay outside these ones. Once it gets in, the terminal condition is over,
       and results in a failure. (stay=True, out=True --> must stay out)
    """

    def __init__(self, body, height, stay=False, out=False):
        """
        Initialize the base height terminal condition.

        Args:
            body (Body): body instance.
            height (float): max height which defines the bound; the bounds will be defined to be between 0 and height.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the position
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the position leaves the bounds, it results in a success.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
        """
        bounds = np.array([[0.], [height]])
        super(BaseHeightCondition, self).__init__(body, bounds=bounds, stay=stay, out=out)

    def _get_states(self):
        """Return the state."""
        return self.body.position[2]


class DistanceCondition(BodyCondition):
    r"""Distance terminal condition

    This is a bit similar than the ``PositionCondition``. The difference is that this class describes a nd-sphere,
    while the ``PositionCondition`` describes a nd-rectangle.

    This terminal condition describes 4 cases (2 failure and 2 success cases); the body distance with respect to the
    provided center must be:

    1. in a certain bounds and must stay between these bounds. Once it gets out, the terminal condition is over,
       and results in a failure. (stay=True, out=False --> must stay in)
    2. in a certain bounds and must get out of these bounds. Once it gets out, the terminal condition is over,
       and results in a success. (stay=False, out=False --> must not stay in)
    3. outside a certain bounds and must get in. Once it gets in, the terminal condition is over, and results
       in a success. (stay=False, out=True --> must not stay out)
    4. outside a certain bounds and must stay outside these ones. Once it gets in, the terminal condition is over,
       and results in a failure. (stay=True, out=True --> must stay out)
    """

    def __init__(self, body, distance=float("inf"), center=(0., 0., 0.), dim=None, stay=False, out=False):
        """
        Initialize the distance terminal condition.

        Args:
            body (Body): body instance.
            distance (float): max distance with respect to the specified :attr:`center`.
            center (np.array(float[3]), list[float[3]], tuple[float[3]]): center from which take the distance.
            dim (None, int, int[3]): dimensions that we should consider when looking at the bounds. If None, it will
                consider all 3 dimensions. If one dimension is provided it will only check along that dimension. If
                a np.array of 0 and 1 is provided, it will consider the dimensions that are equal to 1. Thus, [1,0,1]
                means to consider the distance along the x and z axes.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the position
                leaves the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the position leaves the bounds, it results in a success.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.

        """
        bounds = np.array([[0.], [distance]])
        super(DistanceCondition, self).__init__(body, bounds=bounds, dim=dim, stay=stay, out=out)
        self.center = np.asarray(center)

    def _get_states(self):
        """Return the state."""
        position = self.body.position - self.center
        if self.dim is None:
            return np.linalg.norm(position)
        return np.linalg.norm(position[self.dim])
