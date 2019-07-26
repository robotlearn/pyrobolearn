#!/usr/bin/env python
"""Define the Converter class which allows to convert from one type of file (urdf, sdf, mjcf, and others) to another
format.
"""

from pyrobolearn.utils.parsers.robots import URDFParser, MuJoCoParser

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Converter(object):
    r"""Converter"""

    def __init__(self):
        pass

    def convert(self, from_filename, to_filename):
        pass
