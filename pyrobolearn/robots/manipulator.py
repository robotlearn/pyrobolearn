#!/usr/bin/env python
"""Provide the manipulator abstract classes.
"""

from robot import Robot


class ManipulatorRobot(Robot):
    r"""Manipulator robot

    Manipulator robots are robots that use some of its end-effectors to manipulate objects in its environment.
    """

    def __init__(self,
                 simulator,
                 urdf_path,
                 init_pos=(0, 0, 0.),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.):
        super(ManipulatorRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

        self.arms = []  # list of arms where an arm is a list of links
        self.hands = []  # list of end-effectors/hands

    ##############
    # Properties #
    ##############

    @property
    def num_arms(self):
        """Return the number of arms"""
        return len(self.arms)

    @property
    def num_hands(self):
        """Return the number of hands; this is normally equal to the number of arms"""
        return len(self.hands)

    ###########
    # Methods #
    ###########

    def getNumberOfArms(self):
        """
        Return the number of arms/hands.

        Returns:
            int: the number of arms/hands
        """
        return self.num_arms

    # alias (normally this is correct)
    getNumberOfHands = getNumberOfArms

    def getArmLinkIds(self, armLink=None):
        """
        Return the arm's link id(s) from the name(s) or index(ices).

        Args:
            armLink (str, int, list of str/int, None): if str, it will get the arm's link id associated to the given
                name. If int, it will get the arm's link id associated to the given index. If it is a list of str
                and/or int, it will get the corresponding arm's link ids. If None, it will return all the arm's
                link ids.

        Returns:
            if 1 arm's link:
                int: link id
            if multiple arm's links:
                int[N]: link ids
        """
        if armLink is None:
            return self.arms

        def getIndex(link):
            if isinstance(link, str):
                return self.arm_names[link]
            elif isinstance(link, int):
                return self.arms[link]
            else:
                raise TypeError("Expecting an int or str.")

        # list of links in the arm
        if isinstance(armLink, collections.Iterable) and not isinstance(armLink, str):
            return [getIndex(link) for link in armLink]

        # one link in the arm
        return getIndex(armLink)

    def getArmLinkNames(self, armLinkId=None):
        """
        Return the name of the given arm's link(s).

        Args:
            armLinkId (int, int[N], None): link id, or list of desired link ids. If None, get the name of all links
                in the arms.

        Returns:
            if 1 arm's link:
                str: link name
            if multiple arm's links:
                str[N]: link names
        """
        if isinstance(armLinkId, int):
            return self.sim.getJointInfo(self.id, armLinkId)[12]
        if armLinkId is None:
            armLinkId = [link for arm in self.arms for link in arm]
        return [self.sim.getJointInfo(self.id, link)[12] for link in armLinkId]

    def getHandIds(self):
        pass

    def getHandNames(self):
        pass


class BiManipulatorRobot(ManipulatorRobot):
    r"""Bi-manipulator Robot

    Bi-manipulators are robots that have two manipulators to manipulate objects in the environment.
    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.5), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(BiManipulatorRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

        self.left_arm_id = 0
        self.left_hand_id = 0
        self.right_arm_id = 1
        self.right_hand_id = 1

    ##############
    # Properties #
    ##############

    @property
    def left_arm(self):
        return self.arms[self.left_arm_id]

    @property
    def right_arm(self):
        return self.arms[self.right_arm_id]

    @property
    def left_hand(self):
        return self.hands[self.left_hand_id]

    @property
    def right_hand(self):
        return self.hands[self.right_arm_id]

    ###########
    # Methods #
    ###########

    def getLeftArmIds(self):
        """Return the left arm joint ids"""
        return self.arms[self.left_arm_id]

    def getLeftHandId(self):
        """Return the left hand id"""
        return self.hands[self.left_hand_id]

    def getRightArmIds(self):
        """Return the right arm joint ids"""
        return self.arms[self.right_arm_id]

    def getRightHandId(self):
        """Return the right hand id"""
        return self.hands[self.right_hand_id]
