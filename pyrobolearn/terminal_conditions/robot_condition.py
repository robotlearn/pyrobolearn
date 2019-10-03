# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define some robot terminal conditions for the environment.
"""

from abc import ABCMeta
import numpy as np

from pyrobolearn.robots.base import Body
from pyrobolearn.robots.robot import Robot
from pyrobolearn.terminal_conditions.body_conditions import BodyCondition


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotCondition(BodyCondition):
    r"""Robot Terminal Condition

    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, bounds=(None, None), dim=None, out=False, stay=False, all=False):
        """
        Initialize the robot terminal condition.

        Args:
            robot (Robot): robot instance
            dim (None, int, int[3]): dimensions that we should consider when looking at the bounds. If None, it will
                consider all 3 dimensions. If one dimension is provided it will only check along that dimension. If
                a np.array of 0 and 1 is provided, it will consider the dimensions that are equal to 1. Thus, [1,0,1]
                means to consider the bounds along the x and z axes.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the state
                leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the state leaves the bounds, it results in a success.
            all (bool): this is only used if they are multiple dimensions. if True, all the dimensions of the state
                are checked if they are inside or outside the bounds depending on the other parameters. if False, any
                dimensions will be checked.
        """
        super(RobotCondition, self).__init__(robot, bounds=bounds, dim=dim, out=out, stay=stay, all=all)
        self.robot = robot

    @property
    def robot(self):
        """Return the robot instance."""
        return self._robot

    @robot.setter
    def robot(self, robot):
        """Set the robot instance."""
        if not isinstance(robot, Robot):
            raise TypeError("Expecting the given 'robot' to be an instance of `Robot`, instead got: "
                            "{}".format(type(robot)))
        self._robot = robot


class ContactCondition(RobotCondition):
    r"""Contact condition.

    This terminal condition check if the given links are in or not in contact with another body or another link of a
    body, and describes 8 cases (4 failure and 4 success cases):

    1. all the links are with respect to the specified body/link:
        1. in contact and must remain in contact. Once one is no more in contact, the terminal condition is over, and
           results in a failure. (all=True, stay=True, out=False --> all must stay in (contact))
        2. in contact and must no more be in contact. Once they are all no more in contact, the terminal condition is
           over, and results in a success. (all=True, stay=False, out=False --> all must not stay in (contact))
        3. not in contact initially but must all be in contact at the end. Once they all are in contact, the terminal
           condition is over, and results in a success. (all=True, stay=False, out=True --> all must not stay out)
        4. not in contact initially and must not be in contact at anytime. Once one link is in contact, the terminal
           condition is over, and results in a failure. (all=True, stay=True, out=True --> all must stay out)
    2. any of the links are with respect to the specified body/link:
        1. in contact and must remain in contact. Once they all are not in contact anymore, the terminal condition is
           over, and results in a failure. (all=False, stay=True, out=False --> any must stay in (contact))
        2. in contact and must no more be in contact. Once one link is no more in contact, the terminal condition is
           over, and results in a success. (all=False, stay=False, out=False --> any must not stay in (contact))
        3. not in contact but must get in contact. Once one link gets in contact, the terminal condition is over,
           and results in a success. (all=False, stay=False, out=True --> any must not stay out)
        4. not in contact and must not be in contact at anytime. Once they all are in contact, the terminal condition
           is over, and results in a failure. (all=False, stay=True, out=True --> any must stay out)
    """

    def __init__(self, robot, link_ids, wrt_body=None, wrt_link=-1, out=False, stay=False, all=False, complement=False):
        """
        Initialize the robot terminal condition.

        Args:
            robot (Robot): robot instance.
            link_ids (int, list[int]): link ids that we must check if they are in contact or not (depending on the
                next parameter `out`) with the specified body (see :attr:`wrt_body`) or link (see :attr:`wrt_link`).
            wrt_body (Body, int, None): the body that we have to check if we are colliding with this one.
            wrt_link (int, None): the link of :attr:`wrt_body` that we have to check if we are colliding with. If None,
                it will consider all the links of :attr:`wrt_body`.
            out (bool): if True, we are not in contact. If False, we are in contact.
            stay (bool): if True, it must stay in contact; if one link or all links (depending on the next parameter
                :attr:`all`) is/are no more in contact, it results in a failure. if :attr:`stay` is False, it must no
                more be in contact at a later stage; if one or all (depending on the next parameter :attr:`all`) is/are
                no more in contact, it results in a success.
            all (bool): this is only used if they are multiple links. if True, all the links are checked if they are
                in contact or not depending on the other parameters. if False, any links will be checked if they are
                in contact or not.
            complement (bool): if True, it will take the complement set of the specified link ids. For instance, if
                the feet links are provided as an attribute, it will consider all the other links which are not feet.
        """
        super(ContactCondition, self).__init__(robot, out=out, stay=stay, all=all)

        # check body ids
        self.body_id1 = self.robot.id
        if isinstance(wrt_body, Body):
            wrt_body = wrt_body.id
        self.body_id2 = wrt_body

        # check link ids
        if not isinstance(link_ids, (list, tuple, np.ndarray)):
            link_ids = [link_ids]
        for link_id in link_ids:
            if not isinstance(link_id, int):
                raise TypeError("Expecting each given link id must be an int, but we got instead: "
                                "{}".format(type(link_id)))
        # if complement
        if complement:
            links = [-1] + list(range(self.robot.num_links))
            links = set(links)
            link_ids = list(links.difference(link_ids))

        self.link_ids = link_ids
        self.link_dict = {link_id: idx for idx, link_id in enumerate(self.link_ids)}

        # wrt_link
        if wrt_body is None:
            wrt_link = wrt_body
        if wrt_link is not None and not isinstance(wrt_link, int):
            raise TypeError("Expecting the given 'wrt_link' to be an int, instead got: {}".format(type(wrt_link)))
        self.wrt_link = wrt_link

        # define bounds
        self.bounds = np.ones((2, len(self.link_ids)))  # (2, N)

    def _get_states(self):
        """Get the contact states."""

        # contact_state = []
        # for link_id in self.link_ids:
        #     contacts = self.simulator.get_contact_points(body1=self.body_id1, body2=self.body_id2, link1_id=link_id,
        #                                                  link2_id=self.wrt_link)
        #     if contacts is None:
        #         contact_state.append(0)
        #     else:
        #         n = len(contacts)
        #         n = 1 if n > 0 else 0
        #         contact_state.append(n)
        # return np.array(contact_state)

        contact_state = np.zeros(len(self.link_ids))
        contacts = self.simulator.get_contact_points(body1=self.body_id1, body2=self.body_id2, link2_id=self.wrt_link)
        for contact in contacts:
            link_id = contact[3]
            if link_id in self.link_dict:
                contact_state[self.link_dict[link_id]] = 1
        return contact_state
