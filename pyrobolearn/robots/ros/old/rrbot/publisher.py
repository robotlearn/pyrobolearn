#!/usr/bin/env python
"""Define the RRBot publisher.
"""

from pyrobolearn.robots.ros.publisher import RobotPublisher


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RRBotPublisher(RobotPublisher):

    def __init__(self, id_=None):
        """
        Initialize the RRBot Publisher.

        Args:
            id_ (int, None): robot id which is used when initializing the node. If None, a name will be
                auto-generated for the name using name as the base. See the documentation for the :attr:`anonymous`
                parameter in `rospy.init_node`.
        """
        super(RRBotPublisher, self).__init__(name='rrbot', id_=id_)
