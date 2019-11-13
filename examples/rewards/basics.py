#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate the various features (operations) you can use with the ``Reward`` class.

To illustrate the various operations we use the ``FixedReward``.

See the other examples to see how to use more complex reward functions.
"""

import numpy as np

# You can import in two ways reward functions
import pyrobolearn as prl                       # import PRL
from pyrobolearn.rewards import FixedReward     # specify which function you want to import

# You can also import all the reward functions and the mathematical functions but I usually avoid it because
# we do not know while reading the code where the various classes and other functionalities come from
# from pyrobolearn.rewards import *


# define two fixed reward functions
r1 = prl.rewards.FixedReward(value=3)
r2 = FixedReward(value=2, range=(-2, 2))

# print their value by calling them
print("\nInitial rewards: r1 = {}, and r2 = {}".format(r1, r2))
print("Initial reward value: r1() = {}, and r2() = {}".format(r1(), r2()))
print("Initial reward range: range(r1) = {}, and range(r2) = {}".format(r1.range, r2.range))

# try to define a fixed reward function where the initial value is not in the defined range
try:
    r3 = FixedReward(value=2, range=(-1, 1))
except ValueError as e:
    print("\nTrying `r3=FixedReward(value=2, range=(-1,1))` results in an error: \n" + str(e) + "\n")


# perform some mathematical operations on them
r4 = 2 * prl.rewards.cos(r1) - 3 * r2

# print its value and range
print("Perform mathematical operations on r1 and r2:")
print("r4 = 2 * cos(r1) - 3 * r2 = {}".format(r4()))
print("2 * cos(3) - 3 * 2 = {}".format(2 * np.cos(3) - 3 * 2))
print("range(r4) = {}".format(r4.range))


# try to perform an operation which it not authorized
try:
    r5 = r1 / r2        # the range of r2 is [-2, 2] and thus there is a chance it could be 0 at one point
except ValueError as e:
    print("\nTrying `r5 = r1 / r2` results in an error: \n" + str(e) + "\n")


# The range of r2 is [-2, 2], if a reward function computes a reward value which is not in the range,
# it will automatically be clipped.
r2.value = 3        # you can only set the value for the FixedReward
print("Setting r2.value=3 while its range is [-2, 2], and computing the reward will result in: r2={}".format(r2()))
