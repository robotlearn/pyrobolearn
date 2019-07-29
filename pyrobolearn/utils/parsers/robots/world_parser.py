#!/usr/bin/env python
"""Define the abstract world parser.
"""

# import XML parser
import xml.etree.ElementTree as ET
from xml.dom import minidom  # to print in a pretty way the XML file

from pyrobolearn.utils.parsers.robots.data_structures import World, Tree

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WorldParser(object):
    r"""World Parser and Generator."""

    def __init__(self, filename=None):
        """
        Initialize the world parser.

        Args:
            filename (str, None): path to the file to parse.
        """
        self.root = None
        self.world = None
        self.worlds = []
        self.filename = filename
        if filename is not None:
            self.parse(filename)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        if root is not None and not isinstance(root, ET.Element):
            raise TypeError("Expecting the root to be an instance of `ET.Element`, but got instead: "
                            "{}".format(type(root)))
        self._root = root

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, world):
        if world is not None and not isinstance(world, World):
            raise TypeError("Expecting the given world to be an instance of `World` but got instead: "
                            "{}".format(type(world)))
        self._world = world

    def parse(self, filename):
        """
        Load and parse a given file.

        Args:
            filename (str): path to the file to parse.
        """
        pass

    def get_tree(self, index=None, tag=None):
        """
        Get the specified tree(s).

        Args:
            index (int, None): tree index. If None, it will return all the trees.
            tag (str, None): tag of the root that we want.

        Returns:
            (list of) Tree: tree data structure(s).
        """
        pass

    def get_world(self):
        """
        Return the world containing all the elements that compose that world.

        Returns:
            World: World data structure.
        """
        return self.world

    def generate(self, world=None):
        """
        Generate the XML world from the `World` data structure.

        Args:
            world (World): world data structure.

        Returns:
            ET.Element: root element in the XML file.
        """
        pass

    def get_string(self, root=None):
        """
        Return the XML string from the root element.

        Args:
            root (ET.Element): root element in the XML file.

        Returns:
            str: string representing the XML file.
        """
        if root is None:
            root = self.root
        if not isinstance(root, ET.Element):
            raise ValueError("Expecting the root to be an instance of `ET.Element`, but got instead: "
                             "{}".format(type(root)))
        return minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")

    def write(self, filename, root=None):
        """
        Write the XML world in the specified XML file.

        Args:
            filename (str): path to the file to write the XML in.
            root (ET.Element): root element in the XML file.
        """
        xml_str = self.get_string(root)
        with open(filename, "w") as f:
            f.write(xml_str)  # .encode('utf-8'))
