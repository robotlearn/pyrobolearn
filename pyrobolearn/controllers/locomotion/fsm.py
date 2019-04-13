#!/usr/bin/env python
"""Provide the various states and transitions in a finite state machine (FSM) used in locomotion.

References:
    [1] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Songyan Xin, 2018
"""


__author__ = ["Songyan Xin", "Brian Delhaisse"]
# S.X. wrote the main initial code
# B.D. integrated it in the PRL framework, cleaned it, added the documentation, and made it more modular and flexible
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class State(object):
    """
    We define a state object which provides some utility functions for the individual states within the state machine.
    """

    def __init__(self):
        self.name = str(self)

    def __str__(self):
        """
        Returns the name of the State.
        """
        return self.__class__.__name__


# sagittal hopping states
class Stance(State):
    """
    State: Stance
    """

    def on_event(self, event, pre_state):
        if event == 'TO':  # takeoff
            return Flight()
        else:
            return self


# lateral hopping states
class LeftStance(State):
    """
    State: LeftStance.
    """
    def on_event(self, event):
        if event == 'TO':  # takeoff
            return FlightL2R()
        else:
            return self


class RightStance(State):
    """
    State: RightStance.
    """
    def on_event(self, event):
        if event == 'TO':  # takeoff
            return FlightR2L()
        else:
            return self


class Flight(State):
    """
    State: Flight
    """

    def on_event(self, event, pre_state):

        if event == 'TD':  # touchdown
            if pre_state.name is 'LeftStance':
                return RightStance(), Flight()
            if pre_state.name is 'RightStance':
                return LeftStance(), Flight()
        else:
            return self


class FlightL2R(State):
    """
    State: Flight Left to Right (L2R)
    """

    def on_event(self, event):
        if event == 'TD':  # touchdown
            return RightStance()
        else:
            return self


class FlightR2L(State):
    """
    State: Flight Right to Left (R2L)
    """

    def on_event(self, event):
        if event == 'TD':  # touchdown
            return LeftStance()
        else:
            return self


# state machine
class HoppingStateMachine(object):
    """Hopping state machine

    A simple state machine that mimics the functionality of a device from a high level.
    """

    def __init__(self):
        """ Initialize the components. """

        # Start with a default state.
        self.curr_state = FlightR2L()
        self.prev_state = RightStance()
        print("Initial State: {}".format(self.curr_state))

        # event flag
        self.TD_flag = False
        self.TO_flag = False

    def on_event(self, event, debug=True):
        """
        This is the bread and butter of the state machine. Incoming events are
        delegated to the given states which then handle the event. The result is
        then assigned as the new state.
        """

        # The next state will be the result of the on_event function.
        self.prev_state = self.curr_state
        self.curr_state = self.curr_state.on_event(event)

        if event is "TD": # touchdown
            self.TD_flag = True
            self.TO_flag = False
        elif event is "TO": # takeoff
            self.TD_flag = False
            self.TO_flag = True

        if debug:
            print("{} -> {} -> {}".format(self.prev_state, event, self.curr_state))


def test_hopping_state_machine():
    hopping_state_machine = HoppingStateMachine()
    hopping_state_machine.on_event('TD')
    hopping_state_machine.on_event('TO')
    hopping_state_machine.on_event('TD')
    hopping_state_machine.on_event('TO')
    hopping_state_machine.on_event('TD')


# Tests
if __name__ == '__main__':
    test_hopping_state_machine()
