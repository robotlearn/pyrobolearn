#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the abstract ROS service.
"""

import collections.abc
import rospy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ROSService(object):
    r"""ROS service.

    This just group the various services into one common data structure.
    """

    def __init__(self, name, service_class):
        """
        Initialize the ROS service.

        Args:
            name (str, list[str]): service name(s).
            service_class (class, list[class]): service class(es) for serialization.
        """
        if isinstance(name, str):
            name = [name]
        elif not isinstance(name, collections.abc.Iterable):
            raise TypeError("Expecting the given 'name' to be a string, list of string, but got instead: "
                            "{}".format(type(name)))
        self.names = name

        if not isinstance(service_class, collections.abc.Iterable):
            service_class = [service_class]
        self.service_classes = service_class

        self.services = dict()
        for name, service_class in zip(self.names, self.service_classes):
            service = rospy.ServiceProxy(name, service_class)
            self.services[name] = service

    def call(self, name, *args, **kwargs):
        """
        Calls the given service, and returns the possible response.

        Args:
            name (str): service name.
            *args (list): list of parameters to be given to the service.
            **kwargs (dict): dictionary of parameters to be given to the service.
        """
        rospy.wait_for_service(name)
        try:
            service = self.services[name]
            response = service(*args, **kwargs)
            return response
        except rospy.ServiceException as e:
            print(name + " service call failed...\n" + str(e))

    def create_service(self, names, service_classes):
        """
        Create a ros service. This will be added to the list of inner ROS services.

        Args:
            names (str, list[str]): service name(s).
            service_classes (class, list[class]): service class(es) for serialization.

        Returns:
            rospy.ServiceProxy, list[rospy.ServiceProxy]: created ros services.
        """
        if isinstance(names, str):
            names = [names]
        elif not isinstance(names, collections.abc.Iterable):
            raise TypeError("Expecting the given 'names' to be a string, list of string, but got instead: "
                            "{}".format(type(names)))
        if not isinstance(service_classes, collections.abc.Iterable):
            service_classes = [service_classes]

        for name, service_class in zip(names, service_classes):
            service = rospy.ServiceProxy(name, service_class)
            self.services[name] = service
