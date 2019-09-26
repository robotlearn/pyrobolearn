
import os
import importlib
import inspect
import re

# import basic sensor
from .sensor import Sensor

# import joint + encoder sensors
from .joints import JointSensor, JointEncoderSensor

# import link sensors
from .links import LinkSensor

# import ft sensors
from .force_torque import JointTorqueSensor, JointForceTorqueSensor

# import imu sensors
from .imu import IMUSensor

# import contact sensors
from .contact import ContactSensor

# import camera sensors
from .camera import CameraSensor, DepthCameraSensor, RGBCameraSensor

# import rays
from .ray import RaySensor, RayBatchSensor, HeightmapSensor

# import light / laser sensors
# from .light import *

# import miscellaneous sensors
# from .misc import *


# get a list of implemented sensors
path = os.path.dirname(__file__)
implemented_sensors = set([f[:-3] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                          and f.endswith('.py')])
# remove few items from the set
for s in ['__init__', 'misc', 'light']:
    if s in implemented_sensors:
        implemented_sensors.remove(s)

implemented_sensors = list(implemented_sensors)

# create dictionary that maps sensor names to sensor classes
sensor_names_to_classes = {}
for sensor_name in implemented_sensors:
    module = importlib.import_module('pyrobolearn.robots.sensors.' + sensor_name)

    for name, cls in inspect.getmembers(module):
        # check if it is a class, and the names match
        if inspect.isclass(cls) and issubclass(cls, Sensor):
            if name.lower() == ''.join(sensor_name.split('_')):
                if sensor_name.endswith('_sensor'):
                    sensor_name = sensor_name[:-7]
                elif sensor_name.endswith('sensor') and len(sensor_name) > 6:
                    sensor_name = sensor_name[:-6]
                sensor_names_to_classes[sensor_name] = cls
                name = sensor_name
            else:
                name_list = re.findall('[0-9]*[A-Z]+[0-9]*[a-z]*', name)
                name = '_'.join([n.lower() for n in name_list])
                if name.endswith('_sensor'):
                    name = name[:-7]
                elif name.endswith('sensor') and len(name) > 6:
                    name = name[:-6]
                # TODO: improve regex
                if name == 'rgbcamera':
                    name = 'rgb_camera'
                sensor_names_to_classes[name] = cls

implemented_sensors = set(list(sensor_names_to_classes.keys()))
# print(implemented_sensors)
# print(sensor_names_to_classes)
