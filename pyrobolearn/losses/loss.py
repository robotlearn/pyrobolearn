#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the abstract loss class that is used by the learning algorithm / optimizer.

Losses are evaluated on model parameters, data batches, and / or storages.

Note that rewards / costs and losses are different concepts. Losses are minimized with respect to parameters to
optimize them, while rewards / costs depends on the state(s) and action(s) and are returned by the environment.
"""

from abc import ABCMeta
import operator
import copy
import collections

import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Loss(object):
    r"""Loss abstract class.

    Each loss is minimized.
    """
    __metaclass__ = ABCMeta

    def __init__(self, losses=None):
        """
        Initialize the loss abstract class.

        Args:
            losses (None, list of Loss): internal losses to compute.
        """
        self.losses = losses

    ##############
    # Properties #
    ##############

    @property
    def losses(self):
        """Return the inner losses."""
        return self._losses

    @losses.setter
    def losses(self, losses):
        """Set the inner losses."""
        if losses is None:
            losses = []
        elif isinstance(losses, collections.Iterable):
            for loss in losses:
                if not isinstance(loss, Loss):
                    raise TypeError("Expecting a Loss instance for each item in the iterator.")
        else:
            if not isinstance(losses, Loss):
                raise TypeError("Expecting losses to be an instance of Loss.")
            losses = [losses]
        self._losses = losses

    ###########
    # Methods #
    ###########

    def has_losses(self):
        """Check if it has losses."""
        return len(self._losses) > 0

    def compute(self, *args, **kwargs):
        """Compute the loss and return the scalar value."""
        pass

    def latex(self):  # TODO: check when using operators with latex formula
        """Return a latex formula of the loss."""
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string."""
        if not self.losses or self.losses is None:
            return self.__class__.__name__
        else:
            lst = [loss.__repr__() for loss in self.losses]
            return ' + '.join(lst)

    def __call__(self, *args, **kwargs):
        """Compute the loss."""
        return self.compute(*args, **kwargs)

    def __build_loss(self, other):
        # built the internal list of losses
        losses = self._losses if self.has_losses() else [self]
        if isinstance(other, Loss):
            if other.has_losses():
                losses.extend(other._losses)
            else:
                losses.append(other)
        return Loss(losses=losses)

    def __get_operation(self, other, op):
        if isinstance(other, Loss):  # callable
            def compute(*args, **kwargs):
                return op(self(*args, **kwargs), other(*args, **kwargs))
        else:
            def compute(*args, **kwargs):
                return op(self(*args, **kwargs), other)
        return compute

    def __add__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__add__)
        return loss

    def __div__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__div__)
        return loss

    def __floordiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__floordiv__)
        return loss

    def __iadd__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__iadd__)
        return loss

    def __idiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__idiv__)
        return loss

    def __ifloordiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__ifloordiv__)
        return loss

    def __imod__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__imod__)
        return loss

    def __imul__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__imul__)
        return loss

    def __ipow__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__ipow__)
        return loss

    def __isub__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__isub__)
        return loss

    def __itruediv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__itruediv__)
        return loss

    def __mod__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mod__)
        return loss

    def __mul__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mul__)
        return loss

    def __pow__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__pow__)
        return loss

    def __radd__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__add__)
        return loss

    def __rdiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__div__)
        return loss

    def __rfloordiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__floordiv__)
        return loss

    def __rmod__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mod__)
        return loss

    def __rmul__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mul__)
        return loss

    def __rpow__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__pow__)
        return loss

    def __rsub__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__sub__)
        return loss

    def __rtruediv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__truediv__)
        return loss

    def __sub__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__sub__)
        return loss

    def __truediv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__truediv__)
        return loss

    # binary comparison operators
    def __eq__(self, other):
        compute = self.__get_operation(other, operator.__eq__)
        return compute()

    def __ge__(self, other):
        compute = self.__get_operation(other, operator.__ge__)
        return compute()

    def __gt__(self, other):
        compute = self.__get_operation(other, operator.__gt__)
        return compute()

    def __le__(self, other):
        compute = self.__get_operation(other, operator.__le__)
        return compute()

    def __lt__(self, other):
        compute = self.__get_operation(other, operator.__lt__)
        return compute()

    def __ne__(self, other):
        compute = self.__get_operation(other, operator.__ne__)
        return compute()

    # unary operators
    def __abs__(self):
        loss = copy.copy(self)  # shallow copy

        def compute(*args, **kwargs):
            return operator.__abs__(self(*args, **kwargs))

        loss.compute = compute
        return loss

    def __neg__(self):
        loss = copy.copy(self)  # shallow copy

        def compute(*args, **kwargs):
            return operator.__neg__(self(*args, **kwargs))

        loss.compute = compute
        return loss

    def __pos__(self):
        loss = copy.copy(self)  # shallow copy

        def compute(*args, **kwargs):
            return operator.__pos__(self(*args, **kwargs))

        loss.compute = compute
        return loss


# TODO: the functions defined here should be the same for `reward.py` and `loss.py`. Thus, we need to check numpy
# TODO: or torch inside the functions.
def ceil(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.ceil(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.ceil(x)


def cos(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.cos(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.cos(x)


def cosh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.cosh(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.cosh(x)


# def degrees(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.degrees(x())
#         return y
#     else:
#         return torch.degrees(x)


def exp(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.exp(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.exp(x)


def expm1(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.expm1(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.expm1(x)


def floor(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.floor(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.floor(x)


# def frexp(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.frexp(x())
#         return y
#     else:
#         return torch.frexp(x)


def isinf(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.isinf(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.isinf(x)


def isnan(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.isnan(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.isnan(x)


def log(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.log(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.log(x)


def log10(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.log10(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.log10(x)


def log1p(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.log1p(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.log1p(x)


# def modf(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.modf(x())
#         return y
#     else:
#         return torch.modf(x)


# def radians(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.radians(x())
#         return y
#     else:
#         return torch.radians(x)


def sin(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.sin(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.sin(x)


def sinh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.sinh(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.sinh(x)


def sqrt(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.sqrt(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.sqrt(x)


def tan(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.tan(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.tan(x)


def tanh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.tanh(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.tanh(x)


def trunc(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.trunc(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.trunc(x)
