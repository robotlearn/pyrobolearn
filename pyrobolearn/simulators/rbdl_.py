#!/usr/bin/env python
"""Python wrapper around RBDL

The signature of each method defined here are inspired by the `Robot` class in PyRoboLearn and [1] but in accordance
with the PEP8 style guide [2]. Most of the documentation for the methods have been copied-pasted from [1] for
completeness purposes.

References:
    - [1] RBDL:
        - Webpage (with documentation): https://rbdl.bitbucket.io/
        - Bitbucket repository: https://bitbucket.org/rbdl/rbdl/
    - [2] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

import collections
import rbdl


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Martin Felis (martin@fysx.org)", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RBDL(object):
    r"""RBDL Interface.

    References:
    - [1] RBDL:
        - Webpage (with documentation): https://rbdl.bitbucket.io/
        - Bitbucket repository: https://bitbucket.org/rbdl/rbdl/
    - [2] PEP8: https://www.python.org/dev/peps/pep-0008/
    """

    def __init__(self, filename=None, verbose=False, floating_base=False):
        self.model = None
        self.load_model(filename, verbose=verbose, floating_base=floating_base)

    def load_model(self, filename, verbose=False, floating_base=False):
        """
        Load the given URDF model.

        Args:
            filename (str): path to the urdf model.
            verbose (bool): if True, it will output
            floating_base (bool): if True, the model will be considered to have a floating base, thus 6 more DoFs will
                be added.
        """
        self.model = rbdl.loadModel(filename.encode(), verbose=verbose, floating_base=floating_base)

    # alias
    load_urdf = load_model

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom (DoFs); that is, if the base is not fixed, 6 (= 3 degrees for
        translation + 3 degrees for orientation) + the joints that are not fixed.
        """
        return self.model.dof_count

    # alias
    dof_count = num_dofs

    @property
    def gravity(self):
        """Return the Cartesian gravity vector applied on the model."""
        return self.model.gravity

    @property
    def num_joints(self):
        """Return the number of joints: ROOT + Num DoFs"""
        return len(self.model.mJoints)

    @property
    def num_fixed_links(self):
        """Return the number of fixed links."""
        return len(self.model.mFixedBodies)

    @property
    def num_links(self):
        """Return the number of links."""
        return len(self.model.mBodies)

    @property
    def num_actuated_joints(self):
        """Return the number of actuated joints."""
        return self.model.q_size

    @property
    def fixed_joints(self):
        """Return the list of fixed joints"""
        return []  # TODO

    def get_link_names(self, link_ids):
        """Return the link names."""
        if isinstance(link_ids, collections.Iterable):
            return [self.model.GetBodyName(link_id) for link_id in link_ids]
        return self.model.GetBodyName(link_ids)


# Test
if __name__ == '__main__':
    import os
    import pyrobolearn as prl

    fixed_base = True

    robot = prl.robots.HyQ2Max(prl.simulators.Bullet(render=False), fixed_base=fixed_base)
    rbdl_ = RBDL(os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/hyq2max/hyq2max.urdf',
                 floating_base=not fixed_base)

    robots = [robot, rbdl_]

    def print_attr(msg, attr, *args, **kwargs):
        for rob in robots:
            a = getattr(rob, attr)
            if callable(a):
                print(msg.format(a(*args, **kwargs)))
            print(msg.format(a))

    print_attr("Num DoFs: {}", 'num_dofs')
    print_attr("Num of actuated joints: {}", 'num_actuated_joints')
    print_attr("Num of joints: {}", 'num_joints')
    print_attr("Num of links: {}", 'num_links')

    print(robot.get_link_names([-1] + list(range(robot.num_links))))
    print(rbdl_.get_link_names(range(rbdl_.num_links)))

    print("Num fixed links: {}".format(robot.num_links - robot.num_actuated_joints))
    print("Num fixed links: {}".format(rbdl_.num_fixed_links))
