# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the abstract robot subscriber.
"""

import collections

import numpy as np
import rospy

# import the messages
# from std_msgs import msg as std_msg
from sensor_msgs import msg as sensor_msg
# from geometry_msgs import msg as geometry_msg


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SubscriberData(object):
    r"""Subscriber data holder

    This instantiates the ROS subscriber, save the received data, and allow to easily access the message attributes
    from this class.
    """

    def __init__(self, topic, data_class):
        """
        Initialize the SubscriberData that subscribes to the given topic.

        Args:
            topic (str, list[str]): topic name(s). If multiple topics are given, it will group them. Note that you can
              only group topics that use the same message class.
            data_class (class): message class for serialization.
        """
        # self.subscriber = rospy.Subscriber(topic, data_class, callback=self.callback)
        # # set the message attributes to be part of this class attributes
        # self.attributes = set([attr for attr in [attr for attr in dir(data_class) if not attr.startswith('_')]
        #                        if not callable(getattr(data_class, attr))])
        # self.subscriber_data = data_class()

        self.topic = topic
        self.msg_class = data_class
        if isinstance(topic, collections.Iterable):
            self.is_group = True
            self.subscriber = [rospy.Subscriber(t, data_class, callback=self.callback, callback_args=idx)
                               for idx, t in enumerate(topic)]
            self.msg = [data_class() for _ in topic]
        else:
            self.is_group = False
            self.subscriber = rospy.Subscriber(topic, data_class, callback=self.callback)
            self.msg = data_class()

    def callback(self, data, idx=None):
        """
        Callback function that saves the data in the current instance.

        Args:
            data (object): message class instance.
            idx (int, None): message index.
        """
        if idx is None:
            self.msg = data
        else:
            self.msg[idx] = data

    # def __getattr__(self, name):
    #     """Get the specified attribute value given its name."""
    #     return getattr(self.subscriber_data, name)

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
        if isinstance(self.subscriber, collections.Iterable):
            for subscriber in self.subscriber:
                subscriber.unregister()
        else:
            self.subscriber.unregister()

    def __del__(self):
        """
        Close all topics.
        """
        self.unregister()


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

        # all subscribers {subscriber name: SubscriberData}
        self.subscribers = dict()
        # all topics {topic: subscriber name}
        self.topics_to_subscriber_name = dict()

    def create_subscriber(self, name, topic, data_class):
        """
        Create a subscriber to the specific topic. If the subscriber already exists, it unregister the previous one
        and replace it by the new one.

        Args:
            name (str): unique name of the subscriber. The name must be unique. You will be able to access to this
              subscriber using its name.
            topic (str, list[str]): name of the topic(s).
            data_class (object): data type class to use for messages

        Returns:
            SubscriberData: the subscriber data holder.
        """
        # if the subscriber already exists, unregister and remove it
        self.remove_subscriber(name)

        # create new subscriber
        subscriber = SubscriberData(topic, data_class)
        self.subscribers[name] = subscriber
        self.topics_to_subscriber_name[topic] = name
        return subscriber

    def remove_subscriber(self, name):
        """
        Remove a subscriber from the list of inner subscribers. This will also unregister it.

        Args:
            name (str): unique name of the suscriber.
        """
        if name in self.subscribers:
            self.unregister(name)
            self.subscribers.pop(name)

    def has_subscriber(self, name):
        """
        Return True if the given subscriber name has been created.

        Args:
            name (str): unique name of the subscriber.

        Returns:
            bool: True if the given subscriber name exists.
        """
        return name in self.subscribers

    def get_subscriber(self, name):
        """
        Return the associated `SubscriberData` given its unique name.

        Args:
            name (str): unique name of the subscriber.

        Returns:
            SubscriberData, None: the subscriber data holder. None if the subscriber associated with the given name
              doesn't exist.
        """
        return self.subscribers.get(name)

    def has_topic(self, name):
        """
        Return True if the given topic is used by the subscriber.

        Args:
            name (str): topic name.

        Returns:
            bool: True if the given topic name exists.
        """
        return name in self.topics_to_subscriber_name

    def get_subscriber_name_from_topic(self, topic_name):
        """
        Return the subscriber's name associated with the given topic name.

        Args:
            topic_name (str): topic name.

        Returns:
            str, None: name of the subscriber. None, if no subscriber name is associated with the given topic name.
        """
        return self.topics_to_subscriber_name.get(topic_name)

    def change_topic(self, old_topic, new_topic, new_msg=None):
        """
        Change a subscriber's topic name to a new one with possibly a new message class.

        Args:
            old_topic (str): old topic name.
            new_topic (str): new topic name.
            new_msg (object): message class serialization. If None, it will use the same message class than the old
              topic.

        Returns:
            SubscriberData: the subscriber data holder.
        """
        if not old_topic in self.topics_to_subscriber_name:
            raise ValueError("The given 'old_topic' name ({}) doesn't exist in this subscriber, are you sure it is "
                             "the correct topic name?".format(old_topic))
        name = self.topics_to_subscriber_name[old_topic]

        if new_msg is None:
            subscriber = self.subscribers[name]
            new_msg = subscriber.msg_class

        return self.create_subscriber(name, new_topic, new_msg)

    # def __getattr__(self, name):
    #     return self.subscribers[name]

    def unregister(self, name=None):
        """
        Unsubscribe from a topic. Topic instance is no longer valid after this call. Additional calls to `unregister()`
        have no effect.

        Args:
            name (str, None): name of the topic to unsubscribe. If None, it will unsubscribe from all topics.
        """
        if name is None:
            for subscriber in self.subscribers.values():
                subscriber.unregister()
        else:
            self.subscribers[name].unregister()

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


class RobotSubscriber(Subscriber):
    r"""Robot Subscriber class

    This Robot Subscriber class is the class from which all the robot subscribers inherit from.
    """

    def __init__(self, name, id_=None, joint_state_topics=None, joint_state_msg_class=None):
        r"""
        Initialize the robot subscriber.

        Args:
            name (str): name of the robot. This will be used to create the topics.
            id_ (int, None): robot id which is used when initializing the node. If None, a name will be
              auto-generated for the name using name as the base. See the documentation for the :attr:`anonymous`
              parameter in `rospy.init_node`.
            joint_state_topics (str, list[str]): joint state topic(s). If not provided the joint state topic will be
              set to '/<robot_name>/joint_states'.
            joint_state_msg_class (class): message serialization class used for the provided joint state topic. By
              default, it will be set to 'sensor_msg.JointState'.
        """
        super(RobotSubscriber, self).__init__(subscriber_id=id_)
        self.name = name.lower()

        # create Joint states subscriber (automatically)
        if joint_state_topics is None:
            joint_state_topics = '/' + self.name + '/joint_states'
        if joint_state_msg_class is None:
            joint_state_msg_class = sensor_msg.JointState
        self._joint_states = self.create_subscriber('joint_states', joint_state_topics, joint_state_msg_class)

    ##############
    # Properties #
    ##############

    @property
    def joint_states(self):
        """Get the joint states subscriber."""
        return self._joint_states

    @joint_states.setter
    def joint_states(self, subscriber):
        """Set the joint states subscriber."""
        if not isinstance(subscriber, SubscriberData):
            raise TypeError("Expecting the given 'joint_states' subscriber to be an instance of `SubscriberData` but "
                            "got instead: {}".format(type(subscriber)))
        if self._joint_states is not None:
            self._joint_states.unregister()
        self._joint_states = subscriber

    ###########
    # Methods #
    ###########

    def get_joint_positions(self, q_indices=None):
        """
        Get the joint positions.

        Args:
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will return all the
              joint positions.

        Returns:
            float, np.array[float]: joint positions.
        """
        if q_indices is None:
            return np.asarray(self.joint_states.msg.position)
        return np.asarray(self.joint_states.msg.position)[q_indices]

    def get_joint_velocities(self, q_indices=None):
        """
        Get the joint velocities.

        Args:
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will return all the
              joint velocities.

        Returns:
            float, np.array[float]: joint velocities.
        """
        if q_indices is None:
            return np.asarray(self.joint_states.msg.velocity)
        return np.asarray(self.joint_states.msg.velocity)[q_indices]

    def get_joint_torques(self, q_indices=None):
        """
        Get the joint torques.

        Args:
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will return all the
              joint torques.

        Returns:
            float, np.array[float]: joint torques.
        """
        if q_indices is None:
            return np.asarray(self.joint_states.msg.effort)
        return np.asarray(self.joint_states.msg.effort)[q_indices]

    def get_pid(self, q_indices=None):
        """
        Get the PID coefficients associated to the given joint ids.

        Args:
            q_indices (int, list[int], np.array[int], None): joint q index / indices. If None, it will return all the
              joint PIDs.

        Returns:
            np.array[float[3]], list[np.array[float[3]]]: list of PID coefficients for each joint.
        """
        pass

    def get_jacobian(self, link_id, q=None, local_position=None):
        """
        Return the full geometric jacobian.

        Args:
            link_id (int): link id.
            q (np.array[float[N]], None): joint positions of size N, where N is the number of DoFs. If None, it will
              compute q based on the current joint positions.
            local_position (None, np.array[float[3]]): the point on the specified link to compute the Jacobian (in link
              local coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
        """
        pass

    def get_inertia_matrix(self, q=None):
        """
        Return the inertia matrix.

        Args:
            q (np.array[float[N]], None): joint positions of size N, where N is the total number of DoFs. If None, it
              will get the current joint positions.
        """
        pass


# Tests
if __name__ == '__main__':
    # NOTE: run roscore before hand and don't forget to run the publisher code
    import time
    from itertools import count

    subscriber = RobotSubscriber('walter')
    print("Published topics: {}".format(rospy.get_published_topics()))
    print("Robot joint state attributes: {}".format(subscriber.joint_states.msg.attributes))

    for t in count():
        print(t)
        print("Joint position data: {}".format(subscriber.joint_states.msg.position))
        time.sleep(0.1)
