#!/usr/bin/env python
"""Define the abstract interfaces

This defines the Input/Output abstract interfaces. They allowed to get information from input systems (such as
the mouse, keyboard, microphone, webcam/kinect, VR/AR tools, game controllers, and so on), and/or send information
to output systems (such as speakers, game controllers, VR/AR tools, etc).
These interfaces are independent from the rest of the code; they are not coupled to the simulator, robots, or world.
The code that connects the interfaces to the simulator or elements inside this last one are the `bridges` defined
in `pyrobolearn/tools/bridges`.
"""

# TODO: when using thread makes it in a safe way and in an asynchronous manner

import threading
import time


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def thread_loop(run):
    """decorator to make the function run in a loop if it is a thread"""
    def fct(self, *args, **kwargs):
        if self.use_thread:
            while True:
                run(*args, **kwargs)
        else:
            run(*args, **kwargs)
    return fct


class Interface(object):
    r"""Interface (abstract class)

    The interface links input systems (such as the mouse, keyboard, microphone, webcam/kinect, VR/AR tools, game
    controllers, and so on), and/or output systems (such as speakers, game controllers, VR/AR tools, etc) to the
    simulator. Specifically, it can connects such systems to the `world`, `simulator`, or `robot`.

    They can be divided into 3 classes:
    * Input interfaces: these interfaces interpret the signals received by an input system, and act on the world.
                        Examples include the mouse, keyboard, microphone, webcam/kinect, joysticks, and others.
                        For more info, see the `InputInterface` class.
    * Output interfaces: these interfaces received signals from the world, and output these through an output system.
                         For instance, a robot could 'speak' in the world, and this signal could be redirected to
                         computer speakers or headphones. For more info, see the `OutputInterface` class.
    * Input and Output interfaces: these interfaces connects with a system that can be used as inputs and outputs.
                                   Such instances include controllers with feedback (using vibration or force),
                                   VR/AR tools, phones, etc. For more info, see the `InputOutputInterface` class.
    """

    use_thread = False

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        self.use_thread = use_thread
        self.dt = sleep_dt
        self.data = None
        self.verbose = verbose

        self.stop_thread = False

        if self.use_thread:
            self.thread = threading.Thread(target=self._run)
            self.thread.start()

    def step(self):
        """
        Perform one step with the interface; the interface checks the events (if it is not run in a separate thread)
        """
        if not self.use_thread:
            return self._run()

    def _run(self, *args, **kwargs):
        """
        Code that calls the `run` method implemented by the user, and loop over it if we are in a thread.
        """
        if self.use_thread:
            while True:
                if self.stop_thread:  # if the thread should stop
                    if self.verbose:
                        print("Stopping the thread")
                    break
                self.run(*args, **kwargs)
                time.sleep(self.dt)
        else:
            return self.run(*args, **kwargs)

    # @thread_loop
    def run(self, *args, **kwargs):
        """
        Code to be run by the interface. This needs to be implemented by the user.
        """
        pass

    def close(self):
        """
        Stop and close the interface.
        """
        if self.use_thread:
            if self.verbose:
                print("Asking for thread to stop...")
            self.stop_thread = True

    #############
    # Operators #
    #############

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    def __call__(self):
        self.step()

    def __del__(self):
        self.close()


class InputInterface(Interface):
    r"""Input Interface (abstract class)

    These interfaces interpret the signals received by an input system (such as a mouse, keyboard, microphone,
    webcam/kinect, game controllers, VR/AR controllers), and act on the world.
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring the
                next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(InputInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)


class OutputInterface(Interface):
    r"""Output Interface (abstract class)

    These interfaces received signals from the world, and output these through an output system. For instance,
    a robot could 'speak' in the world, and this signal could be redirected to computer speakers or headphones.
    The screen of your computer is another obvious output system.
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the output interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before setting the
                next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(OutputInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)


class InputOutputInterface(Interface):
    r"""Input and Output interface (abstract class)

    These interfaces can receive and interpret signals from certain systems, and redirect or produce signals to be
    sent to these same systems. Such instances include controllers with feedback (using vibration or force),
    VR/AR tools, phones, and others.

    Specifically, you could send images captured by a camera in the world/simulator, stream these to a phone or VR/AR
    headsets. At the same time, you could capture inputs from the screen (or other sensors) of the phone, or from VR
    controllers, and stream these to this interface that will act on the world/simulator.
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the input + output interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring /
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(InputOutputInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
