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

    ##############
    # Properties #
    ##############

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

    ###########
    # Methods #
    ###########

    def parse(self, filename):
        """
        Load and parse a given file.

        Args:
            filename (str): path to the file to parse.
        """
        pass

    def parse_from_string(self, string):
        """
        Parse the string which contains the description of the world world.

        Args:
            string (str): string containing the description of the world.
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

    def get_string(self, root=None, pretty_format=False):
        r"""
        Return the XML string from the root element.

        Args:
            root (ET.Element): root element in the XML file.
            pretty_format (bool): if we should return the string in a pretty format or not. Note that the pretty
              format add some `\n` and `\t` in the string which might not be good for some simulators.

        Returns:
            str: string representing the XML file.
        """
        if root is None:
            root = self.root
        if not isinstance(root, ET.Element):
            raise ValueError("Expecting the root to be an instance of `ET.Element`, but got instead: "
                             "{}".format(type(root)))
        if pretty_format:
            return minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        return ET.tostring(root).decode("utf-8")

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

    #######################
    # XML related methods #
    #######################

    def get_root(self):
        """
        Return the root element in the XML file.

        Returns:
            ET.Element, None: root element. None, if the parser doesn't have a root.
        """
        return self._root

    def create_root(self, name):
        """
        Create the root element in the XML file.

        Args:
            name (str): name of the root element.

        Returns:
            ET.Element: created root element.
        """
        self._root = ET.Element(name)
        return self._root

    @staticmethod
    def get_element_name(element):
        """
        Return the given element's name.

        Args:
            element (ET.Element): tag element in the XML file.

        Returns:
            str: name of the tag element.
        """
        return element.tag

    @staticmethod
    def get_element_attributes(element):
        """
        Return the given element's attributes.

        Args:
            element (ET.Element): tag element in the XML file.

        Returns:
            dict: dictionary of attributes {name: value}
        """
        return element.attrib

    @staticmethod
    def add_element(name, parent_element, attributes={}):
        """
        Add a new element to the given parent element.

        Args:
            name (str): name of the new element.
            parent_element (ET.Element): parent element in the XML file.
            attributes (dict): attributes of the element.

        Returns:
            ET.Element: the new created element.
        """
        element = ET.SubElement(parent_element, name, attrib=attributes)
        return element

    @staticmethod
    def remove_element(element, parent_element):
        """
        Remove an element from the given parent element.

        Args:
            element (ET.Element): element in the XML file to be removed.
            parent_element (ET.Element): the parent element from which the element is removed.
        """
        parent_element.remove(element)

    @staticmethod
    def get_element(name, parent_element):
        """
        Return the element associated with the given name from the given parent element.

        Args:
            name (str): name of the ET.Element.
            parent_element (ET.Element):

        Returns:
            ET.Element: XML element corresponding to the given name.
        """
        return parent_element.find(name)

    def find(self, name):
        """
        Find the specified element name from root.

        Args:
            name (str): name of the specified element.

        Returns:
            ET.Element: XML element corresponding to the given name.
        """
        return self.root.find(name)
