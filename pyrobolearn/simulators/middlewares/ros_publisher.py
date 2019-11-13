#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the abstract robot publisher.
"""

import collections

import rospy

# import the messages
from std_msgs import msg as std_msg
from sensor_msgs import msg as sensor_msg
from geometry_msgs import msg as geometry_msg


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PublisherData(object):
    r"""Publisher data holder

    This just instantiates the ROS publisher but also allows to easily access the message attributes from this class.
    You can also publish the last message that has been set, or send a new one.

    For instance, the `std_msgs.String` has one attribute called `data`, you can then instantiate a `PublisherData`
    like `pub = PublisherData(topic_name, std_msgs.String, queue_size=10)`. You can then access to the message instance
    with `pub.msg`, and you can directly access to the attribute using `pub.data` (you can also access it with
    `pub.msg.data`).
    """

    def __init__(self, topic, msg_class, queue_size=10):
        """
        Initialize the PublisherData that publishes the given message data.

        Args:
            topic (str, list[str]): topic name(s). If multiple topics are given, it will group them. Note that you can
              only group topics that use the same message class.
            msg_class (class): message class for serialization.
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. When None is passed all publishing will
              happen synchronously and a warning message will be printed.
        """
        # self.__dict__['publisher'] = rospy.Publisher(topic, msg_class, queue_size=queue_size)
        # # set the message attributes to be part of this class attributes
        # self.__dict__['attributes'] = [attr for attr in [attr for attr in dir(msg_class) if not attr.startswith('_')]
        #                                if not callable(getattr(msg_class, attr))]
        # self.__dict__['msg'] = msg_class()

        self.topic = topic
        self.queue_size = queue_size
        self.msg_class = msg_class
        if isinstance(topic, str):
            self.is_group = False
            self.publisher = rospy.Publisher(topic, msg_class, queue_size=queue_size)
            self.msg = msg_class()
        elif isinstance(topic, collections.Iterable):
            self.is_group = True
            self.publisher = [rospy.Publisher(t, msg_class, queue_size=queue_size) for t in topic]
            self.msg = [msg_class() for _ in topic]
        else:
            raise TypeError("Expecting the given 'topic' to a str, or a list of str, but instead got: "
                            "{}".format(type(topic)))

    def publish(self, data=None, indices=None, replace=True):
        """
        Publish the given data.

        Args:
            data (None, class, list[class]): message class instance(s) that holds the data. If None, it will sent the
              last message.
            indices (None, list[int], int): if multiple topics are defined for this class, you can specify which index
              to use.
            replace (bool): if True, it will replace the message by the given data.
        """
        if data is None:
            replace = False
            data = self.msg
            if isinstance(indices, int):
                data = data[indices]

        # if multiple publisher
        if isinstance(self.publisher, collections.Iterable):
            if indices is None:  # publish to every topics the corresponding data
                for idx, (pub, msg) in enumerate(zip(self.publisher, data)):
                    pub.publish(msg)
                    if replace:
                        self.msg[idx] = msg
            else:  # if indices are specified, send it to the specified indices
                if isinstance(indices, int):
                    self.publisher[indices].publish(data)
                    if replace:
                        self.msg[indices] = data
                else:
                    for i, index in enumerate(indices):
                        self.publisher[index].publish(data[i])
                        if replace:
                            self.msg[index] = data[i]

        # if one publisher
        else:
            self.publisher.publish(data)
            if replace:
                self.msg = data

    # def __setattr__(self, key, value):
    #     """Set the given attribute to the message."""
    #     if key in self.attributes:
    #         setattr(self.msg, key, value)
    #
    # def __getattr__(self, key):
    #     """Get the specified attribute value from the message."""
    #     return getattr(self.msg, key)

    def set_attributes(self, key, values, indices=None):
        """
        Set the given value(s) to the message attributes.

        Args:
            key (str): message attribute name.
            values (object): message attribute value(s).
            indices (None, list[int], int): if multiple topics are defined for this class, you can specify which index
              to use.
        """
        # TODO: use `rsetattr` (which is implemented in pyrobolearn/utils/__init__.py)
        if self.is_group:
            if indices is None:  # set every message attribute
                if isinstance(values, collections.Iterable):
                    for msg, value in zip(self.msg, values):
                        setattr(msg, key, value)
                else:
                    for msg in self.msg:
                        setattr(msg, key, values)
            else:
                if isinstance(indices, int):
                    setattr(self.msg[indices], key, values)
                else:
                    if isinstance(values, collections.Iterable):
                        for i, index in enumerate(indices):
                            setattr(self.msg[index], key, values[i])
                    else:
                        for index in indices:
                            setattr(self.msg[index], key, values)

        else:
            setattr(self.msg, key, values)

    def get_attributes(self, key, indices=None):
        """
        Get the given message attribute(s) specified by the given key name.

        Args:
            key (str): message attribute name to get.
            indices (None, list[int], int): if multiple topics are defined for this class, you can specify which index
              to use.

        Returns:
            object: message attribute value(s).
        """
        # TODO: use `rgetattr` (which is implemented in pyrobolearn/utils/__init__.py)
        if self.is_group:
            if indices is None:
                return [getattr(msg, key) for msg in self.msg]
            elif isinstance(indices, int):
                return getattr(self.msg[indices], key)
            elif isinstance(indices, collections.Iterable):
                return [getattr(self.msg[index], key) for index in indices]
            else:
                raise TypeError("Expecting the given indices to be an int, None, or list of int, but got instead: "
                                "{}".format(type(indices)))
        return getattr(self.msg, key)

    def unregister(self):
        """
        Unsubscribe from a topic. Topic instance is no longer valid after this call. Additional calls to `unregister()`
        have no effect.
        """
        if isinstance(self.publisher, collections.Iterable):
            for publisher in self.publisher:
                publisher.unregister()
        else:
            self.publisher.unregister()

    def __del__(self):
        """
        Close all topics.
        """
        self.unregister()


class Publisher(object):
    r"""Publisher class

    This Publisher abstract class is the class from which all the other publishers inherit from. It provides the
    common functionalities between the various publishers.

    It contains one or more `PublisherData` instances to send the various data.
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
        # # try:
        # if publisher_id is None:
        #     rospy.init_node(self.__class__.__name__, anonymous=True)
        # else:
        #     rospy.init_node(self.__class__.__name__ + str(publisher_id))
        # # except rospy.exceptions.ROSException:  # node already initialized
        # #     pass

        # all publishers {publisher name: PublisherData}
        self.publishers = dict()
        # all topics {topic: publisher name}
        self.topics_to_publisher_name = dict()

    def create_publisher(self, name, topic, msg_class, queue_size=10):
        """
        Create a publisher to the specific topic. If the publisher already exists, it unregister the previous one
        and replace it by the new one.

        Args:
            name (str): unique name of the publisher. The name must be unique. You will be able to access to this
              publisher using its name.
            topic (str, list[str]): name of the topic(s).
            msg_class (object): data type class to use for messages
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. When None is passed all publishing will
              happen synchronously and a warning message will be printed.

        Returns:
            PublisherData: the publisher data holder.
        """
        # if the publisher already exists, unregister and remove it
        self.remove_publisher(name)

        # create new publisher
        publisher = PublisherData(topic=topic, msg_class=msg_class, queue_size=queue_size)
        self.publishers[name] = publisher
        if isinstance(topic, collections.Iterable):
            for t in topic:
                self.topics_to_publisher_name[t] = name
        else:
            self.topics_to_publisher_name[topic] = name

        return publisher

    def remove_publisher(self, name):
        """
        Remove a publisher from the list of inner publishers. This will also unregister it.

        Args:
            name (str): unique name of the publisher.
        """
        if name in self.publishers:
            self.unregister(name)
            self.publishers.pop(name)

    def has_publisher(self, name):
        """
        Return True if the given publisher name has been created.

        Args:
            name (str): unique name of the publisher.

        Returns:
            bool: True if the given publisher name exists.
        """
        return name in self.publishers

    def get_publisher(self, name):
        """
        Return the associated `PublisherData` given its unique name.

        Args:
            name (str): unique name of the publisher.

        Returns:
            PublisherData, None: the publisher data holder. None if the publisher associated with the given name
              doesn't exist.
        """
        return self.publishers.get(name)

    def has_topic(self, name):
        """
        Return True if the given topic is used by the publisher.

        Args:
            name (str): topic name.

        Returns:
            bool: True if the given topic name exists.
        """
        return name in self.topics_to_publisher_name

    def get_publisher_name_from_topic(self, topic_name):
        """
        Return the publisher's name associated with the given topic name.

        Args:
            topic_name (str): topic name.

        Returns:
            str, None: name of the publisher. None, if no publisher name is associated with the given topic name.
        """
        return self.topics_to_publisher_name.get(topic_name)

    def change_topic(self, old_topic, new_topic, new_msg=None, queue_size=None):
        """
        Change a publisher's topic name to a new one with possibly a new message class and queue size.

        Args:
            old_topic (str): old topic name.
            new_topic (str): new topic name.
            new_msg (object): message class serialization. If None, it will use the same message class than the old
              topic.
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. If None, it will use the same queue size
              than the old topic.

        Returns:
            PublisherData: the publisher data holder.
        """
        if not old_topic in self.topics_to_publisher_name:
            raise ValueError("The given 'old_topic' name ({}) doesn't exist in this publisher, are you sure it is the "
                             "correct topic name?".format(old_topic))
        name = self.topics_to_publisher_name[old_topic]
        publisher = self.publishers[name]

        if new_msg is None:
            new_msg = publisher.msg_class
        if queue_size is None:
            queue_size = publisher.queue_size

        return self.create_publisher(name, new_topic, new_msg, queue_size=queue_size)

    def publish(self, name=None, data=None, indices=None):
        """
        Publish the given data using the given publisher name.

        Args:
            name (str): unique name of the publisher.
            data (class, None): message class instance that holds the data. If None, it will sent the last message.
            indices (None, list[int], int): if multiple topics are defined for this class, you can specify which index
              to use.
        """
        if name is None and data is None:
            for publisher in self.publishers.values():
                publisher.publish()
        elif name is not None:
            # self.__dict__[name].publish(data)
            self.publishers[name].publish(data=data, indices=indices)

    # def __getattr__(self, name):
    #     return self.publishers[name]

    def unregister(self, name=None):
        """
        Unsubscribe from a topic. Topic instance is no longer valid after this call. Additional calls to `unregister()`
        have no effect.

        Args:
            name (str, None): name of the topic to unsubscribe. If None, it will unsubscribe from all topics.
        """
        if name is None:
            for publisher in self.publishers.values():
                publisher.unregister()
        else:
            self.publishers[name].unregister()

    def close(self):
        """
        Close all topics.
        """
        self.unregister()

    def __del__(self):
        """
        Close all topics.
        """
        self.unregister()


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
        # self.joint_states = PublisherData(self.name + '/joint_states', sensor_msg.JointState)
        # self.publishers['joint_states'] = self.joint_states

        # self.joint_cmds = self.create_publisher('joint_cmds', self.name + '/joint_commands', sensor_msg.JointState)

        # joint position/velocity/torque commands
        self.q_cmd, self.dq_cmd, self.tau_cmd = None, None, None
        self.q_attr, self.dq_attr, self.tau_attr = None, None, None

    @staticmethod
    def __check_publisher_and_attribute(publisher, attribute_name):
        """
        Check the type of the publisher and attribute name.

        Args:
            publisher (PublisherData): publisher.
            attribute_name (str): message attribute name.
        """
        if not isinstance(publisher, PublisherData):
            raise TypeError("Expecting the given 'publisher' to be an instance of `PublisherData`, instead got: "
                            "{}".format(type(publisher)))
        if not isinstance(attribute_name, str):
            raise TypeError("Expecting the given 'attribute_name' to be a string, instead got: "
                            "{}".format(type(attribute_name)))

    def init_set_joint_positions(self, publisher, msg_attribute_name):
        """
        Initialize set joint positions.

        Args:
            publisher (PublisherData): publisher.
            msg_attribute_name (str): message attribute name.
        """
        self.__check_publisher_and_attribute(publisher, msg_attribute_name)
        self.q_cmd = publisher
        self.q_attr = msg_attribute_name

    def set_joint_positions(self, positions, q_indices=None):
        """
        Set the given joint positions.

        Args:
            positions (float, np.array[float]): joint position(s) to set.
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will consider all
              the joints.
        """
        if self.q_cmd is not None:
            self.q_cmd.set_attributes(key=self.q_attr, values=positions, indices=q_indices)

    def init_set_joint_velocities(self, publisher, msg_attribute_name):
        """
        Initialize set joint velocities.

        Args:
            publisher (PublisherData): publisher.
            msg_attribute_name (str): message attribute name.
        """
        self.__check_publisher_and_attribute(publisher, msg_attribute_name)
        self.dq_cmd = publisher
        self.dq_attr = msg_attribute_name

    def set_joint_velocities(self, velocities, q_indices=None):
        """
        Set the given joint velocities.

        Args:
            velocities (float, np.array[float]): joint velocity(ies) to set.
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will consider all
              the joints.
        """
        if self.dq_cmd is not None:
            self.dq_cmd.set_attributes(key=self.dq_attr, values=velocities, indices=q_indices)

    def init_set_joint_torques(self, publisher, msg_attribute_name):
        """
        Initialize set joint torques.

        Args:
            publisher (PublisherData): publisher.
            msg_attribute_name (str): message attribute name.
        """
        self.__check_publisher_and_attribute(publisher, msg_attribute_name)
        self.tau_cmd = publisher
        self.tau_attr = msg_attribute_name

    def set_joint_torques(self, torques, q_indices=None):
        """
        Set the given joint torques.

        Args:
            torques (float, np.array[float]): joint torques to set.
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will consider all
              the joints.
        """
        if self.tau_cmd is not None:
            self.tau_cmd.set_attributes(key=self.tau_attr, values=torques, indices=q_indices)

    def set_pid(self, pid, q_indices=None):
        """
        Set the given PID coefficients to the given joint ids.

        Args:
            pid (list[np.array[float[3]]]): list of PID coefficients for each joint. If one of the value is -1, it
              will left untouched the associated PID value to the previous one.
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will consider all
              the joints.
        """
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     self.reset_srv()
        # except rospy.ServiceException as e:
        #     print("/gazebo/reset_simulation service call failed")
        pass  # /rrbot/joint1_position_controller/pid/set_parameters


# Tests
if __name__ == '__main__':
    # NOTE: run roscore before hand
    import numpy as np
    from itertools import count
    import time

    publisher = RobotPublisher('walter')
    print("Published topics: {}".format(rospy.get_published_topics()))
    print("Robot joint state attributes: {}".format(publisher.joint_cmds.msg.attributes))

    publisher.joint_cmds.position = np.array(range(3))

    for t in count():
        print(t)
        publisher.publish()
        time.sleep(0.1)
