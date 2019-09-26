#!/usr/bin/env python
"""Define the abstract robot publisher.
"""

import rospy

# import the messages
from std_msgs import msg as std_msg
from sensor_msgs import msg as sensor_msg
from geometry_msgs import msg as geometry_msg


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PublisherData(object):
    r"""Publisher data holder
    """

    def __init__(self, topic, data_class, queue_size=10):
        self.__dict__['publisher'] = rospy.Publisher(topic, data_class, queue_size=queue_size)
        self.__dict__['attributes'] = [attr for attr in [attr for attr in dir(data_class) if not attr.startswith('_')]
                                       if not callable(getattr(data_class, attr))]
        self.__dict__['publisher_data'] = data_class()

    def publish(self, data=None):
        if data is None:
            self.publisher.publish(self.publisher_data)
        else:
            self.publisher.publish(data)

    def __setattr__(self, key, value):
        if key in self.attributes:
            setattr(self.publisher_data, key, value)

    def __getattr__(self, key):
        return getattr(self.publisher_data, key)


class Publisher(object):
    r"""Publisher class

    This Publisher abstract class is the class from which all the other publishers inherit from. It provides the
    common functionalities between the various publishers.
    """

    def __init__(self, publisher_id=None):
        """
        Initialize the publisher.

        Args:
            publisher_id (int, None): publisher id which is used when initializing the node. If None, a name will be
                auto-generated for the name using name as the base. See the documentation for the :attr:`anonymous`
                parameter in `rospy.init_node`.
        """

        # initialize the node
        if publisher_id is None:
            rospy.init_node(self.__class__.__name__, anonymous=True)
        else:
            rospy.init_node(self.__class__.__name__ + str(publisher_id))

        # all publishers
        self.publishers = dict()

    def create_publisher(self, name, topic, data_class):
        """
        Create a publisher to the specific topic.

        Args:
            name (str): unique name of the publisher. The name must be unique. You will be able to access to this
            topic (str): name of the topic.
            data_class (object): data type class to use for messages

        Returns:
            PublisherData: the publisher data holder.
        """
        publisher = PublisherData(topic, data_class)
        self.publishers[name] = publisher
        setattr(self, name, publisher)
        return publisher

    def publish(self, name=None, data=None):
        if name is None and data is None:
            for publisher in self.publishers.values():
                publisher.publish()
        elif name is not None:
            self.__dict__[name].publish(data)

    # def __getattr__(self, name):
    #     return self.publishers[name]


class RobotPublisher(Publisher):
    r"""Robot Publisher class

    This Robot Publisher class is the class from which all the robot publishers inherit from.
    """

    def __init__(self, name, id_=None):
        r"""
        Initialize the robot publisher.

        Args:
            name (str): name of the robot. This will be used to create the topics.
            id_ (int, None): robot id which is used when initializing the node. If None, a name will be
                auto-generated for the name using name as the base. See the documentation for the :attr:`anonymous`
                parameter in `rospy.init_node`.
        """
        super(RobotPublisher, self).__init__(publisher_id=id_)
        self.name = name.lower()

        # create Joint states
        # self.create_publisher('joint_states', self.name + '/joint_states', sensor_msg.JointState)
        self.joint_states = PublisherData(self.name + '/joint_states', sensor_msg.JointState)
        self.publishers['joint_states'] = self.joint_states

    def set_joint_positions(self, joint_ids, positions):
        # self.joint_states.position[joint_ids] = positions
        self.joint_states.position = positions

    def set_joint_velocities(self, joint_ids, velocities):
        self.joint_states.velocity[joint_ids] = velocities

    def set_joint_torques(self, joint_ids, torques):
        self.joint_states.effort[joint_ids] = torques


# Tests
if __name__ == '__main__':
    # NOTE: run roscore before hand
    import numpy as np
    from itertools import count
    import time

    publisher = RobotPublisher('walter')
    print("Published topics: {}".format(rospy.get_published_topics()))
    print("Robot joint state attributes: {}".format(publisher.joint_states.attributes))

    publisher.joint_states.position = np.array(range(3))

    for t in count():
        print(t)
        publisher.publish()
        time.sleep(0.1)
