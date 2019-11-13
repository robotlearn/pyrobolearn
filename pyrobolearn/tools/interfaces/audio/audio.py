#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the main basic audio interface.
"""

# TODO

from pyrobolearn.tools.interfaces.interface import Interface, InputInterface, OutputInterface, InputOutputInterface

# To use microphone (using PyAudio). This also needed for the 'speech_recognition' module.
try:
    import pyaudio
except ImportError as e:
    # `pip install --allow-external pyaudio --allow-unverified pyaudio pyaudio` ??
    string = "\nHint: try to install pyaudio by typing the following lines in the terminal: \n" \
             "sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0\n" \
             "sudo apt-get install ffmpeg libav-tools\n" \
             "pip install pyaudio\n"
    raise ImportError(e.__str__() + string)


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class AudioInterface(Interface):
    r"""Audio Interface

    References:
        [1] https://gist.github.com/mabdrabo/8678538
        [2] https://raspberrypi.stackexchange.com/questions/59852/pyaudio-does-not-detect-my-microphone-connected-via-usb-audio-adapter
        [3] https://www.swharden.com/wp/2016-07-19-realtime-audio-visualization-in-python/
        [4] https://www.programcreek.com/python/example/52624/pyaudio.PyAudio
        [5] https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio

        [6] https://realpython.com/python-speech-recognition/#working-with-microphones
    """

    def __init__(self):
        """Initialize the audio interface."""
        super(AudioInterface, self).__init__()
        self.port = pyaudio.PyAudio()
        self.stream = self.port.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True,
                                     frames_per_buffer=1024)  # input_device_index=)

    def print_info(self):
        """Print information about the audio interface."""
        for i in range(self.port.get_device_count()):
            info = self.port.get_device_info_by_index(i)
            print("###############################################################")
            print("Index: {} - name: {} - rate: {} ".format(i, info['name'], info['defaultSampleRate']))
            print("Max input/output channels: {}, {}".format(info['maxInputChannels'], info['maxOutputChannels']))
            print("Input latency (low, high): {}, {}".format(info['defaultLowInputLatency'],
                                                             info['defaultHighInputLatency']))
            print("Output latency (low, high): {}, {}".format(info['defaultLowOutputLatency'],
                                                              info['defaultHighOutputLatency']))
            print("Is an input device? {}".format(self.is_input_device(info)))
            print("Is an output device? {}".format(self.is_output_device(info)))

    @staticmethod
    def is_input_device(info):
        return info['maxInputChannels'] != 0

    @staticmethod
    def is_output_device(info):
        return info['maxOutputChannels'] != 0

    def step(self):
        """Perform a step with the interface."""
        data = self.stream.read()

    def __del__(self):
        """Delete the audio interface."""
        self.stream.stop_stream()
        self.stream.close()


class InputAudioInterface(InputInterface):
    r"""Input Audio Interface.

    See `pyAudio`: https://people.csail.mit.edu/hubert/pyaudio/

    In pyAudio:
    * 'Rate': sampling rate, i.e. the number of frames per second
    * 'chunk': arbitrary chosen number of frames the signals are split into
    """
    pass


class OutputAudioInterface(OutputInterface):
    r"""Output Audio Interface

    See `pyAudio`: https://people.csail.mit.edu/hubert/pyaudio/
    """
    pass


class InputOutputAudioInterface(InputOutputInterface):
    r"""Input and Output Audio Interface.

    See `pyAudio`: https://people.csail.mit.edu/hubert/pyaudio/
    """
    pass


# Tests
if __name__ == '__main__':
    interface = AudioInterface()
    interface.print_info()
