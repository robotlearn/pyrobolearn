#!/usr/bin/env python
"""Define the Converter class which allows to convert from one type of file (urdf, sdf, mjcf, and others) to another
format.
"""

from pyrobolearn.utils.parsers.robots import URDFParser, MuJoCoParser, SDFParser

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Converter(object):
    r"""Converter

    This converts one world / robot file to another one.
    """

    def __init__(self):
        pass

    def convert(self, from_filename, to_filename):
        """
        Convert one robot/world file to another one. If it is a world file to a robot file, it will create a robot file
        for each model that were in the world.

        Args:
            from_filename (str): file to parse (specified with the extension).
            to_filename (str, list of str): file to generate (specified with the extension). You can also only
                specified the extension if you wish. If that is the case, the name will be taken from the file that
                is being parsed.
        """
        # check the types
        if not isinstance(from_filename, str):
            raise TypeError("Expecting the 'from_filename' to be a str, but got instead: "
                            "{}".format(type(from_filename)))
        if not isinstance(to_filename, str):
            raise TypeError("Expecting the 'to_filename' to be a str, but got instead: "
                            "{}".format(type(to_filename)))

        # extension for the 'from_filename'
        from_extension = from_filename.split('.')[-1]
        if from_extension == 'urdf':  # URDF
            parser = URDFParser(filename=from_filename)
        elif from_extension == 'sdf' or from_extension == 'world':  # SDF
            parser = SDFParser(filename=from_filename)
        elif from_extension == 'mjcf' or from_extension == 'xml':  # MuJoCo
            parser = MuJoCoParser(filename=from_filename)
        elif from_extension == 'proto':
            # parser = ProtoParser(filename=from_filename)
            raise NotImplementedError("The proto parser has not been implemented yet")
        else:
            raise ValueError("Got the extension '{}' from 'from_filename', however this format is not "
                             "known".format(type(from_extension)))

        # extension for the 'to_filename'
        to_extension = to_filename.split('.')
        if len(to_extension) == 1:
            to_extension = to_filename
        else:
            to_extension = to_extension[-1]

        # generator
        if to_extension == 'urdf':  # URDF
            generator = URDFParser()
        elif to_extension == 'sdf' or to_extension == 'world':  # SDF
            generator = SDFParser()
        elif to_extension == 'mjcf' or to_extension == 'xml':  # MuJoCo
            generator = MuJoCoParser()
        elif to_extension == 'proto':
            # generator = ProtoParser()
            raise NotImplementedError("The proto parser has not been implemented yet")
        else:
            raise ValueError("Got the extension '{}' from 'to_filename', however this format is not "
                             "known".format(type(from_extension)))

        # generate the files
        # TODO
