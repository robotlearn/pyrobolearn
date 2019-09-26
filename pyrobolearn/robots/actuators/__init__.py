# -*- coding: utf-8 -*-

import os
import importlib
import inspect
import re

# import basic actuator
from .actuator import Actuator

# import joint actuators
from .joints import JointActuator, JointPositionActuator, JointVelocityActuator, JointPositionVelocityActuator, \
    JointTorqueActuator

# import speaker
from .speaker import Speaker


# get a list of implemented actuators
path = os.path.dirname(__file__)
implemented_actuators = set([f[:-3] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                          and f.endswith('.py')])
# remove few items from the set
for s in ['__init__', 'misc', 'light']:
    if s in implemented_actuators:
        implemented_actuators.remove(s)

implemented_actuators = list(implemented_actuators)

# create dictionary that maps actuator names to actuator classes
actuator_names_to_classes = {}
for actuator_name in implemented_actuators:
    module = importlib.import_module('pyrobolearn.robots.actuators.' + actuator_name)

    for name, cls in inspect.getmembers(module):
        # check if it is a class, and the names match
        if inspect.isclass(cls) and issubclass(cls, Actuator):
            if name.lower() == ''.join(actuator_name.split('_')):
                if actuator_name.endswith('_actuator'):
                    actuator_name = actuator_name[:-9]
                elif actuator_name.endswith('actuator') and len(actuator_name) > 8:
                    actuator_name = actuator_name[:-8]
                actuator_names_to_classes[actuator_name] = cls
                name = actuator_name
            else:
                name_list = re.findall('[0-9]*[A-Z]+[0-9]*[a-z]*', name)
                name = '_'.join([n.lower() for n in name_list])
                if name.endswith('_actuator'):
                    name = name[:-9]
                elif name.endswith('actuator') and len(name) > 8:
                    name = name[:-8]
                actuator_names_to_classes[name] = cls

implemented_actuators = set(list(actuator_names_to_classes.keys()))
# print(implemented_actuators)
# print(actuator_names_to_classes)
