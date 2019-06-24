#!/usr/bin/env python
"""Define the abstract robot subscriber.
"""

import numpy as np
import rospy

# import the messages
# from std_msgs import msg as std_msg
from sensor_msgs import msg as sensor_msg
# from geometry_msgs import msg as geometry_msg


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SubscriberData(object):
    r"""Subscriber data holder
    """

    def __init__(self, topic, data_class):
        self.subscriber = rospy.Subscriber(topic, data_class, callback=self.callback)
        self.attributes = set([attr for attr in [attr for attr in dir(data_class) if not attr.startswith('_')]
                               if not callable(getattr(data_class, attr))])
        self.subscriber_data = None

    def callback(self, data):
        self.subscriber_data = data

    def __getattr__(self, name):
        if self.subscriber_data is not None:
            return getattr(self.subscriber_data, name, None)

    def unregister(self):
        self.subscriber.unregister()


class Subscriber(object):
    r"""Subscriber class

    This Subscriber abstract class is the class from which all the other subscribers inherit from. It provides the
    common functionalities between the various subscribers.
    """

    def __init__(self, subscriber_id=None):
        """
        Initialize the subscriber.

        Args:
            subscriber_id (int, None): subscriber id which is used when initializing the node. If None, a name will be
                auto-generated for the name using name as the base. See the documentation for the :attr:`anonymous`
                parameter in `rospy.init_node`.
        """

        # initialize the node
        if subscriber_id is None:
            rospy.init_node(self.__class__.__name__, anonymous=True)
        else:
            rospy.init_node(self.__class__.__name__ + str(subscriber_id))

        # all subscribers
        self.subscribers = dict()

    def create_subscriber(self, name, topic, data_class):
        """
        Create a subscriber to the specific topic.

        Args:
            name (str): unique name of the subscriber. The name must be unique. You will be able to access to this
            topic (str): name of the topic.
            data_class (object): data type class to use for messages

        Returns:
            SubscriberData: the subscriber data holder.
        """
        subscriber = SubscriberData(topic, data_class)
        self.subscribers[name] = subscriber
        return subscriber

    def __getattr__(self, name):
        return self.subscribers[name]

    def unregister(self, name=None):
        if name is None:
            for subscriber in self.subscribers.values():
                subscriber.unregister()
        else:
            self.subscribers[name].unregister()

    def close(self):
        self.unregister()

    def __del__(self):
        self.unregister()


class RobotSubscriber(Subscriber):
    r"""Robot Subscriber class

    This Robot Subscriber class is the class from which all the robot subscribers inherit from.
    """

    def __init__(self, name, id_=None):
        r"""
        Initialize the robot subscriber.

        Args:
            name (str): name of the robot. This will be used to create the topics.
            id_ (int, None): robot id which is used when initializing the node. If None, a name will be
                auto-generated for the name using name as the base. See the documentation for the :attr:`anonymous`
                parameter in `rospy.init_node`.
        """
        super(RobotSubscriber, self).__init__(subscriber_id=id_)
        self.name = name.lower()

        # create Joint states
        self.create_subscriber('joint_states', self.name + '/joint_states', sensor_msg.JointState)

    def get_joint_positions(self, joint_ids):
        return np.asarray(self.joint_states.position)[joint_ids]

    def get_joint_velocities(self, joint_ids):
        return np.asarray(self.joint_states.velocity)[joint_ids]

    def get_joint_torques(self, joint_ids):
        return np.asarray(self.joint_states.effort)[joint_ids]


# Tests
if __name__ == '__main__':
    # NOTE: run roscore before hand and don't forget to run the publisher code
    import time
    from itertools import count

    subscriber = RobotSubscriber('walter')
    print("Published topics: {}".format(rospy.get_published_topics()))
    print("Robot joint state attributes: {}".format(subscriber.joint_states.attributes))

    for t in range(100):
        print(t)
        print("Joint position data: {}".format(subscriber.joint_states.position))
        time.sleep(0.1)
