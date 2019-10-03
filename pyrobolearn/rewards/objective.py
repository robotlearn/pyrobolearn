# -*- coding: utf-8 -*-
# This file defines the main `Objective` class, which is inherited by the `Reward` and `Cost` classes, and
# all subsequent child classes which define common objective/reward/cost functions.
# Objectives can be maximized or minimized, while rewards are maximized, and costs are minimized.
# Allow to define objectives with different frameworks: numpy, pytorch, tensorflow, theano,...
# That is, by giving the framework we want to use as input to the children of the `Objective` class,
# it will return the objective in the correct format. Thus we don't need to define various classes for each possible
# different frameworks.
#
# Some objectives needs to access the robot's and/or environment's information.
#
# Objectives can be added together using `+` and multiplied by a weight number using `*`
#
# To see the doc in python interpreter:
# from pyrobolearn.objectives.objective import Objective
# print(Objective.__doc__)


from abc import ABCMeta, abstractmethod


class Objective(object):
    """Abstract `Objective` class which defines the objective function to be maximized or minimized.
    This class must be inherited by any classes which defines a reward or cost.

    .. seealso:
    """
    __metaclass__ = ABCMeta

    def __init__(self,  objective, maximize=True):
        self._objective = objective
        self.maximize = maximize

    def __neg__(self):
        """
        max f = min -f  <-->  min f = max -f
        This function is useful when using optimizers which only accepts to maximize xor minimize,
        thus by taking the negative of the objective, it will still respect the initial objective
        to optimize.
        :return: -objective
        """
        return Objective(-self.objective, maximize=not self.maximize)

    def __add__(self, other):
        """
        Define how to add two objectives. Adding two rewards or two costs make sense,
        but adding a reward with a cost does not.

        By default, if one of the objectives is to be maximized while
        the other one is to be minimized, it will return an objective to be maximized.
        :param other:
        :return:
        """
        if (self.maximize == other.maximize):
            return Objective(self.objective + other.objective, maximize=self.maximize)
        else:
            raise ValueError("Trying to add an objective to maximize with one to minimize. You probably meant: "
                             "objective_1 - objective_2.")

    def __sub__(self, other):
        """
        Substracting two rewards or two costs do not make any sense. However, substracting a reward
        and a cost makes sense.
        :param other:
        :return:
        """
        return self.__add__(-other)

    def __mul__(self, other): # self * other
        if other < 0:
            return Objective(abs(other)*self.objective, maximize = not self.maximize)
        return Objective(other * self.objective, maximize=self.maximize)

    def __rmul__(self, other): # other * self
        return self.__mul__(other)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self._objective = objective

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def loss_fct(self, *args, **kwargs):
        """
        Returns the loss fct.
        """
        pass

    def loss_value(self, *args, **kwargs):
        """
        Returns the loss value. It evaluate the loss function for a particular instance.
        """
        pass

    def fct(self, *args, **kwargs):
        """
        Define the objective function.
        """
        pass

    @abstractmethod
    def symbolic(self):
        """
        Return symbolic expression of the objective function.
        :return:
        """
        pass

    @abstractmethod
    def latex(self):
        """
        Returns the latex formula of the objective function.
        :return:
        """
        pass

