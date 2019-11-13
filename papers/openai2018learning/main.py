#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This sample of code implements the "Learning Dexterous In-Hand Manipulation" paper [1] using the PyRoboLearn framework.

We use the same factors as described in the paper; i.e. the same robotic platform, same states and actions,
same policies, same rewards (and coefficients), same learning algorithm with same hyperparameters, and so on.

Reference:
    [1] "Learning Dexterous In-Hand Manipulation", OpenAI et al., 2019 (https://arxiv.org/abs/1808.00177)
"""

import time
from itertools import count
import pyrobolearn as prl
from env import Openai2018LearningEnv


# create environment (create world, states, rewards, actions)
environment = Openai2018LearningEnv()

# run environment
for t in count():
    environment.step()
    time.sleep(1./240)

# create policy

# create algo
