#!/usr/bin/env python
"""Define the recorder classes.

The recorders allow to record the data from a source at a certain rate.
For instance, it can record the robot states and actions. This can be useful for imitation learning tasks where
the user demonstrates a certain skill through teleoperation or kinesthetic teaching. Using the recorder, you can
record the data to be replayed later to the learning model.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

# Memory vs Storage vs Recorder vs Sampler

import pickle
import copy
import time

from pyrobolearn.states import State
from pyrobolearn.actions import Action

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Recorder(object):
    """Recorder

    This class allows to record the given data, and save it to a file.
    """

    def __init__(self, filename=None):
        """
        Initialize the recorder.

        Args:
            filename (str, None): file to save/load the data. If None, it will generate a filename based on the class
                name and the current local time.
        """
        if filename is None:
            filename = self.__class__.__name__ + time.strftime("_%d-%m-%Y_%Hh%Mm%Ss", time.localtime())
        self.filename = filename
        # data row
        self.data = []
        # data "matrix" (it is not really a matrix because each row may have different dimension)
        self.all_data = []

    def __repr__(self):
        return self.__class__.__name__

    def __len__(self):
        """
        Return the current number of data points recorded.
        """
        return len(self.data)

    def __iter__(self):
        """
        Return iterator over the data.

        Returns:
            iterator
        """
        return iter(self.data)

    def __getitem__(self, key):
        """
        Access the data from the recorder.

        Args:
            key (int, slice): data index

        Returns:
            np.array: data
        """
        return self.data[key]

    def __contains__(self, item):
        """
        Check if the given item is in the recorder.

        Warnings: Currently, this is an O(N) operation. Need to use OrderedSet.

        Args:
            item: item to check if it is in the recorder

        Returns:
            bool: True if the item is in the recorder
        """
        return item in self.data

    def add(self, data):
        """
        Add the given data to the recorder.

        Args:
            data: data to add to the recorder
        """
        self.data.append(data)

    # alias
    __lshift__ = add

    def add_row(self):
        """
        Add a new data row in the list of data.
        """
        self.all_data.append(self.data)
        self.data = []

    def remove(self, key=0):
        """
        Remove and return the specified data from the recorded data.

        Args:
            key (int): data index
        """
        return self.data.pop(key)

    def remove_last_entry(self):
        """
        Remove and return the last recorded datum.
        """
        return self.remove(key=-1)

    def remove_first_entry(self):
        """
        Remove and return the first recorded datum.
        """
        return self.remove(key=0)

    def remove_row(self, key=0):
        """
        Remove and return a data row in the list of all data.

        Args:
            key: data row index
        """
        return self.all_data.pop(key)

    def remove_last_row(self):
        """
        Remove and return the last data row from the data "matrix".
        """
        return self.remove_row(key=-1)

    def remove_first_row(self):
        """
        Remove and return the first data row from the data "matrix".
        """
        return self.remove_row(key=0)

    def save(self, filename=None, append=True):
        """
        Save the recorded data into the specified filename.

        Args:
            filename (str, None): filename to save the data. If None, use the default one provided at the beginning.
            append (bool): If True, it will append the data to the end of the file
        """
        if filename is None:
            filename = self.filename
        mode = "wba" if append else "wb"
        with open(filename, mode) as f:
            pickle.dump(self.data, f)

    def load(self, filename=None):
        """
        Load the recorded data from the specified filename.

        Args:
            filename (str, None): filename to load the data from. If None, use the default one provided at the
                beginning.
        """
        if filename is None:
            filename = self.filename
        with open(filename, "rb") as f:
            self.data = pickle.load(f)

    def generate(self):
        """
        Return a generator over the recorded data.

        Returns:
            generator
        """
        for data in self.data:
            yield data

    def reset(self):
        """
        Reset the recorder; empty it.
        """
        self.data = []


class DataRecorder(Recorder):
    r"""Data Recorder

    This class is useful to record data from a source. Basically, we specify what we wish to record.
    """

    def __init__(self, source, filename=None, rate=1, update=True):
        """Initialize the recorder.

        Args:
            source (object): instance that needs to have the property variable `data`. At each acquisition step,
                we append/save `source.data`
            filename (str, None): file to save/load the data. If None, it will generate a filename based on the class
                name and the current local time.
            rate (int): sampling rate
            update (bool): update the source by calling it if callable.
        """

        # Get world and simulator
        super(DataRecorder, self).__init__(filename)

        # useful variables
        if not hasattr(source, 'data'):
            raise AttributeError("The given source doesn't have the 'data' attribute")
        self.src = source
        self.rate = rate
        self.cnt = 0
        self.update = update

    def record(self):
        """
        Record the data.
        """
        if (self.cnt % self.rate) == 0:
            self.cnt = 0
            self.update_source()
            self.data.append(copy.deepcopy(self.src.merged_data))
        self.cnt += 1

    def __call__(self, *args, **kwargs):
        self.record()

    def update_source(self):
        if self.update and callable(self.src):
            self.src()

    def generate(self):
        """
       Return a generator over the recorded data.

       Returns:
           generator
       """
        for data in self.data:
            self.src.data = data
            yield data


class StateRecorder(DataRecorder):
    r"""State recorder

    Record the state.
    """

    def __init__(self, states, filename=None, rate=1, update=True):
        """
        Record the given states at each `rate` time steps.

        Args:
            states (State): states to save
            filename (str, None): file to save/load the states. If None, it will generate a filename based on the class
                name and the current local time.
            rate (int): sampling rate
            update (bool): update the states by calling it if callable.
        """
        if not isinstance(states, State):
            raise TypeError("Expecting 'states' to be an instance of State")
        super(StateRecorder, self).__init__(source=states, filename=filename, rate=rate, update=update)


class ActionRecorder(DataRecorder):
    r"""Action recorder

    Record the action.
    """

    def __init__(self, actions, filename=None, rate=1, update=True):
        """
        Record the given actions at each `rate` time steps.

        Args:
            actions (Action): actions to save
            filename (str, None): file to save/load the states. If None, it will generate a filename based on the class
                name and the current local time.
            rate (int): sampling rate
            update (bool): update the actions by calling it if callable.
        """
        if not isinstance(actions, Action):
            raise TypeError("Expecting 'actions' to be an instance of Action")
        super(ActionRecorder, self).__init__(source=actions, filename=filename, rate=rate, update=update)
