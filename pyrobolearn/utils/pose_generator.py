# -*- coding: utf-8 -*-
# This file defines the `PoseGenerator` class which generates possible or plausible poses for a robot.
# A pose is defined as:
# - the link positions/orientations, and the base position/orientation.
# - the joint positions, and the base position/orientation.


class PoseGenerator(object):

    def __init__(self, robot):
        self.robot = robot

    def generate_uniform_random_pose(self, jnts=None):
        pass

    def generate_gaussian_random_pose(self, jnts=None):
        """
        Put a gaussian distribution with the mean sets to jnt initial configuration, and the 2 times
        the standard deviation sets to ...
        :param jnts:
        :return:
        """
        pass

    def generate_plausible_pose(self, model, jnts=None):
        """
        Given a trained learning model (for instance a GAN or VAE), it generates a plausible pose of the robot.
        :param model: learning model
        :param jnts:
        :return:
        """
        pass

    def generate_random_pose(self, generator, jnts=None):
        """
        Based on the given distribution generator, it generates a pose.
        :param generator:
        :param jnts:
        :return:
        """
        pass