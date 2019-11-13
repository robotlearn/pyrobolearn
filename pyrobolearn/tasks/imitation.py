#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the imitation learning task.

Dependencies:
- `pyrobolearn.recorders`
- `pyrobolearn.tools` (interfaces / bridges to be used to demonstrate a certain skill).
"""

import collections
from itertools import count
import time
import numpy as np

from pyrobolearn.tasks.task import Task
from pyrobolearn.recorders import Recorder, StateRecorder, ActionRecorder
# from pyrobolearn.tools.interfaces import Interface
from pyrobolearn.tools.bridges import Bridge
from pyrobolearn.tools.bridges.mouse_keyboard import BridgeMouseKeyboardImitationTask

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ILTask(Task):
    r"""Imitation Learning Task

    Imitation learning consists for an agent to generalize to reproduce/perform a certain task from possibly few
    demonstrations [1].

    References:
        [1] "Learning from Humans", Billard et al., 2016
    """

    def __init__(self, environment, policies, interface=None, recorders=None):
        """
        Initialize the imitation learning task.

        Args:
            environment (Env): environment of the task (which contains the world)
            policies (Policy): rl to be trained by the task
            interface (Bridge, Interface): input/output interface that allows to interact with the world.
                If None is provided, it will create by default the `MouseKeyboardInterface`.
            recorders (Recorder, [Recorder]): if the interface doesn't have a recorder, it can be supplemented here.
                If the interface doesn't have a recorder, and it is not specified, it will create a recorder that
                record the states and actions (inferred from the rl).
        """
        super(ILTask, self).__init__(environment, policies)

        # recorder
        self.recorders = recorders

        # interface
        self.bridge = interface

        # few useful variables (that are accessed by the bridge)
        self.recording_enabled = False
        self.end_recording = False
        self.training_enabled = False
        self.end_training = False
        self.testing_enabled = False
        self.end_testing = False
        self.end_task = False

        # use the following code to check the caller class and method
        # import inspect
        # stack = inspect.stack()
        # the_class = stack[1][0].f_locals["self"].__class__
        # the_method = stack[1][0].f_code.co_name

    ##############
    # Properties #
    ##############

    @property
    def recording_enabled(self):
        return self._recording_enabled

    @recording_enabled.setter
    def recording_enabled(self, boolean):
        self._recording_enabled = boolean

        # if you are recording, you can not train or test
        if self._recording_enabled:
            self._training_enabled = False
            self._testing_enabled = False

    @property
    def training_enabled(self):
        return self._training_enabled

    @training_enabled.setter
    def training_enabled(self, boolean):
        self._training_enabled = boolean

        # if you are training, you can not record or test
        if self._training_enabled:
            self._recording_enabled = False
            self._testing_enabled = False

    @property
    def testing_enabled(self):
        return self._testing_enabled

    @testing_enabled.setter
    def testing_enabled(self, boolean):
        self._testing_enabled = boolean

        # if you are testing, you can not record or train
        if self._testing_enabled:
            self._recording_enabled = False
            self._training_enabled = False

    @property
    def recorders(self):
        """Return the list of recorders."""
        return self._recorders

    @recorders.setter
    def recorders(self, recorders):
        """Set the recorders."""
        if recorders is None:
            recorders = [StateRecorder(self.policies.states), ActionRecorder(self.policies.actions)]
        elif isinstance(recorders, Recorder):  # or not isinstance(recorders, collections.Iterable):
            recorders = [recorders]
        if not isinstance(recorders, collections.Iterable):
            raise TypeError("Expecting a list of recorders.")
        # check each recorder
        for recorder in recorders:
            if not isinstance(recorder, Recorder):
                raise TypeError("Expecting each recorder to be an instance of `pyrobolearn.recorders.Recorder`, "
                                "instead got: {}".format(type(recorder)))
        self._recorders = recorders

    @property
    def bridge(self):
        """Return the bridge to the interface."""
        return self._bridge

    @bridge.setter
    def bridge(self, bridge):
        """Set the bridge to the interface. If None, create a Bridge to a mouse-keyboard interface."""
        # If no bridge given, create one
        if bridge is None:
            bridge = BridgeMouseKeyboardImitationTask()
        # elif isinstance(bridge, Interface):
        #     pass

        # check if bridge instance
        if not isinstance(bridge, Bridge):
            raise TypeError("Expecting the bridge to be an instance of Bridge, got instead {}".format(type(bridge)))

        # set bridge and task
        self._bridge = bridge
        self._bridge.task = self

    @property
    def interface(self):
        """Return the interface instance associated with the bridge."""
        return self._bridge.interface

    ###########
    # Methods #
    ###########

    def step_bridge(self, update_interface=True):
        """
        Perform one step with the interface and bridge.

        Args:
            update_interface (bool): if True, it will perform one step with the interface.
        """
        # perform one step with the interface and bridge
        self.bridge.step(update_interface=update_interface)

    def step(self, deterministic=True, render=True):
        """
        Perform one step with the environment and the policies.
        """
        if render:
            self.env.render()

        for policy in self.policies:
            # prev_obs = copy.deepcopy(policy.states.data)
            if self.testing_enabled:
                actions = policy.act(policy.states, deterministic=deterministic, to_numpy=True, apply_action=True)
            else:
                actions = None
            obs, rew, done, info = self.env.step()
            self._done = done
            # d = {'prev_obs': prev_obs, 'actions': copy.deepcopy(actions.data),
            #      'obs': copy.deepcopy(policy.states.data), 'rew': rew, 'done': done}
            # results.append(d)

    def save_recorders(self):
        """
        Save data from recorders.
        """
        for recorder in self.recorders:
            recorder.save()

    def reset_recorders(self):
        """
        Reset the recorders.
        """
        for recorder in self.recorders:
            recorder.reset()

    def add_data_row_in_recorder(self):
        """
        Add a new data row in the recorder's data "matrix".
        """
        for recorder in self.recorders:
            recorder.add_row()

    def discard_last_recorded_data(self):
        """
        Discard/remove the last recorded piece of data.
        """
        for recorder in self.recorders:
            recorder.remove_last_entry()

    def record_step(self):
        """Perform one step in the recording."""
        if self.recording_enabled:
            # perform a step with the recorder
            for recorder in self.recorders:
                recorder.record()

    def record(self, num_steps=None, dt=1./240, signal_from_interface=True):  # , render=True):
        """
        Record the states / actions using the recorders.

        Args:
            num_steps (int, None): total number of steps to take in the environment. If None, it will not end.
            dt (float, None): time to sleep before going to the next step. If None, it will be 1./240.
            render (bool): If True, it will render the environment.
            signal_from_interface (bool): If True, it is assumed that a signal is sent by the interface to start/stop
                the recording.
        """
        # check if the recording is enabled
        self.recording_enabled = not signal_from_interface
        self.bridge.enable_recording = signal_from_interface

        # check the number of steps
        if num_steps is None:
            num_steps = np.infty

        # check dt
        if dt is None or dt < 0.:
            dt = 1./240

        # run several steps in the world
        self.reset()
        for t in count():
            if t >= num_steps or self.end_recording:
                self.end_recording = False
                break

            # perform one step with the interface and bridge
            self.bridge.step(update_interface=True)

            # record if specified
            self.record_step()

            # perform a step in the world
            self.world.step(sleep_dt=dt)

    def train_step(self):
        """
        Perform one step in the training of the policy(ies) using the recorded data.
        """
        if self.training_enabled:
            # print("data: ", self.recorders[0].data)
            # print("shape: ", np.array(self.recorders[0].data).shape)

            # take data from recorder
            data = np.array([data[0] for data in self.recorders[0].data]).T

            # train policy
            for policy in self.policies:
                policy.imitate(data)

    def train(self, num_iters=1000, signal_from_interface=False):
        """
        Train the policy(ies) using the recorded data.

        Args:
            num_iters (int): number of iterations to train the model.
            signal_from_interface (bool): If True, it is assumed that a signal is sent by the interface to start/stop
                the training.
        """
        # check if the training is enabled
        self.training_enabled = not signal_from_interface
        self.bridge.enable_training = signal_from_interface

        # check the number of steps
        if num_iters is None or num_iters < 0:
            num_iters = 1000

        # run several steps in the environment
        for i in range(num_iters):
            if self.end_training:
                self.end_training = False
                break

            # perform one step with the interface and bridge
            self.bridge.step(update_interface=True)

            # perform a step in the training process
            self.train_step()

            # TODO one step learning
            if self.training_enabled:
                self.end_training = True
                self.training_enabled = False

    def test_step(self):
        """
        Perform one step in the test process.
        """
        if self.testing_enabled:
            # perform a step in the environment
            self.step(render=True)

    def test(self, num_steps=None, dt=1./240, signal_from_interface=False):
        """
        Test the policy(ies) in the environment.

        Args:
            num_steps (int, None): total number of steps to take in the environment. If None, it will not end.
            dt (float, None): time to sleep before going to the next step. If None, it will be 1./240.
            render (bool): If True, it will render the environment.
            signal_from_interface (bool): If True, it is assumed that a signal is sent by the interface to start/stop
                the testing.
        """
        # check if the recording is enabled
        self.testing_enabled = not signal_from_interface
        self.bridge.enable_testing = signal_from_interface

        # check the number of steps
        if num_steps is None:
            num_steps = np.infty

        # check dt
        if dt is None or dt < 0.:
            dt = 1. / 240

        # run several steps in the environment
        # print('Test: resetting...')
        self.reset()
        # time.sleep(10)
        for t in count():
            if t >= num_steps or self.end_testing:
                self.end_testing = False
                break

            # perform one step with the interface and bridge
            self.bridge.step(update_interface=True)

            # perform a step in the environment
            if self.testing_enabled:
                self.step(render=True)
            else:
                self.world.step()

            # sleep
            time.sleep(dt)

    def run(self, num_steps=None, dt=1./240, render=True):
        """
        Reset and run the task until it is done, or the current time step matches num_steps.
        It allows to record data, train, and test a policy using the bridge between the interface and this task.

        Warnings: It is assumed that the bridge will send the signals to record the data, train and test the policy,
        and end the task.

        Args:
            num_steps (int, None): total number of steps to take in the environment. If None, it will not end.
            dt (float, None): time to sleep before going to the next step. If None, it will be 1./240.
            render (bool): If True, it will render the environment. We always render in an imitation task, as it does
                not make sense to not render in this setting.
        """
        # We use the interface
        self.recording_enabled = False
        self.training_enabled = False
        self.testing_enabled = False
        self.bridge.enable_recording = True
        self.bridge.enable_training = True
        self.bridge.enable_testing = True

        # check the number of steps
        if num_steps is None:
            num_steps = np.infty

        # check dt
        if dt is None or dt < 0.:
            dt = 1. / 240

        self.reset()
        for t in count():
            if t >= num_steps or self.end_task:
                self.end_task = False
                break

            # perform one step with the interface and bridge
            self.bridge.step(update_interface=True)

            # record
            if self.recording_enabled:
                self.record_step()
                # perform a step forward in the simulation
                self.world.step(sleep_dt=dt)

            # train
            if self.training_enabled:
                self.train_step()
                # perform a step forward in the simulation (without sleeping?) TODO
                self.world.step()

            # test
            if self.testing_enabled:
                self.test_step()
                # perform a step in the environment
                self.step(render=True)
                time.sleep(dt)

            # if not recording, training or testing, just perform a step in the world
            if not (self.recording_enabled or self.training_enabled or self.testing_enabled):
                self.world.step(sleep_dt=dt)
