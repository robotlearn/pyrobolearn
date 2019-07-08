#!/usr/bin/env python
"""Define the various actuators used in robotics.

This is decoupled from the robots such that actuators can be defined outside the robot class and can be selected at
run-time. This is useful for instance when a version of the robot has specific joint motors while another version has
other joint actuators. Additionally, this is important as more realistic motors can result in a better transfer from
simulation to reality.
"""

from pyrobolearn.robots.actuators import Actuator
from pyrobolearn.utils.data_structures.queues import FIFOQueue
import pyrobolearn as prl


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Speaker(Actuator):
    r"""Speaker class
    """

    def __init__(self, capacity=1, check_same_text=False, verbose=False):
        """
        Initialize the speaker actuator.

        Args:
            capacity (int): maximum capacity of the queue. Every given text is appended to that queue. If set to 0,
                it will an infinite capacity.
            check_same_text (bool): if True, it will check that the same text is not being said twice.
            verbose (bool): if True, it will print the messages returned by the speaker interface.
        """
        super(Speaker, self).__init__()
        self.queue = FIFOQueue(maxsize=int(capacity))
        self.check_same_text = bool(check_same_text)
        self.last_text = None
        self.interface = prl.tools.interfaces.audio.SpeakerInterface(use_thread=True, verbose=verbose)

    ###########
    # Methods #
    ###########

    def say(self, text=None):
        # if we have something to say, add it to the queue
        if text is not None:
            if not isinstance(text, str):
                raise TypeError("Expecting the 'given' text to be an instance of ")
            if not self.check_same_text or self.last_text != text:
                self.queue.append(text)

        # if the interface is ready and we have something to say
        if not self.interface.updated and not self.queue.empty():
            data = self.queue.get()
            self.interface.data = data

    def compute(self, text=None):
        self.say(text)

    def __del__(self):
        self.interface.close()


# Test the actuator
if __name__ == '__main__':
    import time

    speaker = Speaker(capacity=1)

    speaker.say('Hello world!')
    speaker.say('My name is Boxy!')

    for t in range(700):
        if t == 300:
            speaker.say('How are you today?')
        time.sleep(0.01)

    # TODO: that should be automatic
    del speaker
