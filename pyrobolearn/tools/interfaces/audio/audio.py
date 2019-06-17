#!/usr/bin/env python
"""Define the main basic Camera interface

This defines the main basic camera interface from which all other interfaces which uses a camera inherit from.
"""

import os

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

# Speech recognition
# Good tutorial: https://realpython.com/python-speech-recognition/#working-with-microphones
try:
    import speech_recognition as sr
except ImportError as e:
    string = "\nHint: try to install speech_recognition by typing the following lines in the terminal: \n" \
             "sudo apt-get install libpulse-dev" \
             "pip install pocketsphinx\n" \
             "pip install google-cloud-speech" \
             "pip install SpeechRecognition"
    raise ImportError(e.__str__() + string)

# Speech synthesis
# Good tutorial: https://pythonprogramminglanguage.com/text-to-speech/
# If Python3.3 or higher: https://pypi.org/project/google_speech/
try:
    from gtts import gTTS
except ImportError as e:
    string = "\nHint: try to install gTTS by typing: pip install gTTS"
    raise ImportError(e.__str__() + string)

# # Another one is `pyttsx3`, which is the best offline module (the problem is that it only supports english)
# # Documentation (with examples): pyttsx3.readthedocs.io/en/latest/
# try:
#     import pyttsx3
# except ImportError as e:
#     string = "\nHint: try to install pyttsx3 by typing: pip install pyttsx3"
#     raise ImportError(e.__str__() + string)

# what I also checked: `pyttsx` and `pyvona`
# import pyttsx
# import pyvona


# Translation
# Github repo: https://github.com/ssut/py-googletrans
# Tutorial: https://www.codeproject.com/Tips/1236705/How-to-Use-Google-Translator-in-Python
try:
    from googletrans import Translator
except ImportError as e:
    string = "\nHint: try to install googletrans by typing: pip install googletrans"
    raise ImportError(e.__str__() + string)


# ChatterBot
# Github repo: https://github.com/gunthercox/ChatterBot
# Documentation: https://chatterbot.readthedocs.io/en/stable
# try:
#     from chatterbot import ChatBot
# except ImportError as e:
#     string = "\nHint: try to install chatterbot by typing: pip install chatterbot"
#     raise ImportError(e.__str__() + string)


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


class SpeechRecognizerInterface(InputInterface):
    r"""Speech Recognizer Interface

    References:
        [1] Tutorial: https://realpython.com/python-speech-recognition/#working-with-microphones
    """

    def __init__(self, use_thread=False, lang='english', verbose=False):
        """
        Initialize the speech recognizer input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            lang (str): language to recognize
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # device_index=-1

        languages = {'french': 'fr', 'english': 'en', 'american english': 'en-US', 'british english': 'en-GB',
                     'indian english': 'en-IN', 'italian': 'it', 'japanese': 'ja', 'korean': 'ko', 'german': 'de',
                     'dutch': 'nl', 'spanish': 'es', 'spanish (peru)': 'es-PE', 'chinese': 'zh-CN',
                     'mandarin': 'zh-CN', 'polish': 'pl', 'portuguese': 'pt', 'russian': 'ru', 'greek': 'el-GR'}
        # Check < https://gist.github.com/traysr/2001377 > for more
        self.lang = languages[lang]

        # string that is being said
        self.data = ''

        super(SpeechRecognizerInterface, self).__init__(use_thread=use_thread, verbose=verbose)

    def run(self):
        """Run the interface."""
        # listen to speech through the microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("I am listening...")
            audio = self.recognizer.listen(source)  # listen to what is being said

        # recognize speech (get the string from audio)
        try:
            print('Trying to understand what you just said...')
            self.data = self.recognizer.recognize_google(audio, language=self.lang)
        except sr.UnknownValueError:
            print("Unable to recognize speech")
        except sr.RequestError as e:
            print("API unavailable".format(e))

        if self.verbose:
            print("You said: {}".format(self.data))


class SpeechSynthesizerInterface(OutputInterface):
    r"""Speech Synthesizer Interface

    References:
        [1] tutorial: https://pythonprogramminglanguage.com/text-to-speech/
        [2] If Python3.3 or higher: https://pypi.org/project/google_speech/
    """

    def __init__(self, use_thread=False, lang='english', verbose=False):
        """
        Initialize the speech synthesizer output interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            lang (str): language to recognize
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        languages = {'french': 'fr', 'english': 'en', 'american english': 'en-US', 'british english': 'en-GB',
                     'indian english': 'en-IN', 'italian': 'it', 'japanese': 'ja', 'korean': 'ko', 'german': 'de',
                     'dutch': 'nl', 'spanish': 'es', 'spanish (peru)': 'es-PE', 'chinese': 'zh-CN',
                     'mandarin': 'zh-CN', 'polish': 'pl', 'portuguese': 'pt', 'russian': 'ru', 'greek': 'el-GR'}
        # Check < https://gist.github.com/traysr/2001377 > for more
        self.lang = languages[lang]

        self.updated = False
        self.data = ''

        super(SpeechSynthesizerInterface, self).__init__(use_thread=use_thread, verbose=verbose)

    def run(self):
        """Run the interface."""
        if self.updated:
            # tts = text-to-speech
            tts = gTTS(text=self.data, lang=self.lang)
            tts.save('tmp.mp3')
            os.system('mpg321 tmp.mp3')
            os.system('rm tmp.mp3')
            self.updated = False

    def update(self, data):
        """Update the data."""
        self.data = data
        self.updated = True


class SpeechTranslatorInterface(InputOutputInterface):
    r"""Speech Translator Interface
    """

    def __init__(self, use_thread=False, target_lang='english', from_lang='auto', verbose=False):
        """
        Initialize the speech translator input/output interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            target_lang (str): language to translate to.
            from_lang (str): language to translate from.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        self.translator = Translator()

        languages = {'french': 'fr', 'english': 'en', 'american english': 'en-US', 'british english': 'en-GB',
                     'indian english': 'en-IN', 'italian': 'it', 'japanese': 'ja', 'korean': 'ko', 'german': 'de',
                     'dutch': 'nl', 'spanish': 'es', 'spanish (peru)': 'es-PE', 'chinese': 'zh-CN',
                     'mandarin': 'zh-CN', 'polish': 'pl', 'portuguese': 'pt', 'russian': 'ru', 'greek': 'el-GR',
                     'auto': 'auto'}
        # Check < https://gist.github.com/traysr/2001377 > for more
        self.target_lang = languages[target_lang]
        self.from_lang = languages[from_lang]

        self.updated = False
        self.input_data = ''
        self.data = ''

        super(SpeechTranslatorInterface, self).__init__(use_thread, verbose=verbose)

    def run(self):
        """Run the interface."""
        # translate
        if self.updated and self.target_lang != self.from_lang:
            translated = self.translator.translate(self.input_data, dest=self.target_lang, src=self.from_lang)
            self.data = translated.text
            self.updated = False

    def update(self, data):
        """Update the data."""
        self.input_data = data
        self.updated = True


class SpeechInterface(InputOutputInterface):
    r"""Speech Interface

    This class performs speech recognition, translation, and synthesization.
    """

    def __init__(self, use_thread=False, target_lang='english', from_lang='english', verbose=False):
        """
        Initialize the speech (recognizer, translator, and synthesizer) interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            target_lang (str): language to translate to.
            from_lang (str): language to translate from.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        self.recognizer = SpeechRecognizerInterface(use_thread=False, lang=from_lang)
        self.translator = None
        if target_lang != from_lang:
            self.translator = SpeechTranslatorInterface(use_thread=False, target_lang=target_lang,
                                                        from_lang=from_lang)
        self.synthesizer = SpeechSynthesizerInterface(use_thread=False, lang=target_lang)

        self.updated = False
        self.input_data = ''
        self.output_data = ''

        super(SpeechInterface, self).__init__(use_thread, verbose=verbose)

    def run(self):
        """Run the interface."""
        pass

    def update(self, data):
        """Update the interface."""
        pass


# Tests
if __name__ == '__main__':
    interface = AudioInterface()
    interface.print_info()

    # recognize, translate and synthesize speech
    english = set(['en', 'en-US', 'en-GB'])
    languages = {'french': 'fr', 'english': 'en', 'american english': 'en-US', 'british english': 'en-GB',
                 'indian english': 'en-IN', 'italian': 'it', 'japanese': 'ja', 'korean': 'ko', 'german': 'de',
                 'dutch': 'nl', 'spanish': 'es', 'spanish (peru)': 'es-PE', 'chinese': 'zh-CN',
                 'mandarin': 'zh-CN', 'polish': 'pl', 'portuguese': 'pt', 'russian': 'ru', 'greek': 'el-GR'}
    # Check < https://gist.github.com/traysr/2001377 > for more
    lang = languages['english']

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()  # device_index=-1

    # print(microphone.list_microphone_names())

    # listen to speech through the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = recognizer.listen(source)

    # recognize speech
    print('processing...')
    string = ''
    try:
        string = recognizer.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    except sr.RequestError as e:
        print("API unavailable".format(e))

    print("You said: " + string)

    # translate it if other language than english
    if lang not in english:
        translator = Translator()
        translated = translator.translate(string)  # dest='en', src='auto')
        print("which translates to: " + translated.text)

    # produce speech
    print('Let me try to repeat what you just said:')
    tts = gTTS(text=string, lang=lang)
    tts.save('tmp.mp3')
    os.system('mpg321 tmp.mp3')
    os.system('rm tmp.mp3')

    # recognize speech using Sphinx
    # try:
    #     print("Sphinx thinks you said '" + recognizer.recognize_sphinx(audio) + "'")
    # except sr.UnknownValueError:
    #     print("Sphinx could not understand audio")
    # except sr.RequestError as e:
    #     print("Sphinx error; {0}".format(e))
