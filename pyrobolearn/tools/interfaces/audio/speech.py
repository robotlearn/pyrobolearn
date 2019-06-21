#!/usr/bin/env python
"""Define the various speech interfaces allowing to perform speech recognition, translation, and synthesization.
"""
# TODO: chatbox, google assistant, alexa

import os  # TODO: use subprocess instead
# import subprocess

from pyrobolearn.tools.interfaces.interface import InputInterface, OutputInterface, InputOutputInterface


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
    string = "\nHint: try to install gTTS by typing: `pip install gTTS`." \
             "Also install `mpg321` using `sudo apt-get install mpg321`."
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


class SpeechRecognizerInterface(InputInterface):
    r"""Speech Recognizer Interface

    References:
        [1] Tutorial: https://realpython.com/python-speech-recognition/#working-with-microphones
    """

    available_languages = {'french', 'english', 'american english',  'british english', 'indian english', 'italian',
                           'japanese', 'korean', 'german', 'dutch', 'spanish', 'spanish (peru)', 'chinese',
                           'mandarin', 'polish', 'portuguese', 'russian', 'greek'}

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
            if self.verbose:
                print("Say something, I am listening!")
            audio = self.recognizer.listen(source)  # listen to what is being said

        # recognize speech (get the string from audio)
        try:
            if self.verbose:
                print('Please wait, trying to understand what you just said...')
            self.data = self.recognizer.recognize_google(audio, language=self.lang)

            if self.verbose:
                print("You said: {}".format(''.join(self.data).encode('utf-8')))
            return self.data
        except sr.UnknownValueError:
            print("Unable to recognize speech")
        except sr.RequestError as e:
            print("API unavailable".format(e))

        return None


class SpeechSynthesizerInterface(OutputInterface):
    r"""Speech Synthesizer Interface

    References:
        [1] tutorial: https://pythonprogramminglanguage.com/text-to-speech/
        [2] If Python3.3 or higher: https://pypi.org/project/google_speech/
    """

    available_languages = {'french', 'english', 'american english', 'british english', 'indian english', 'italian',
                           'japanese', 'korean', 'german', 'dutch', 'spanish', 'spanish (peru)', 'chinese',
                           'mandarin', 'polish', 'portuguese', 'russian', 'greek'}

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
        self._data = ''

        super(SpeechSynthesizerInterface, self).__init__(use_thread=use_thread, verbose=verbose)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, (str, unicode)):
            self._data = data
            self.updated = True

    def run(self):
        """Run the interface."""
        if self.updated:
            if self.verbose:
                print("Generating speech...")
            # tts = text-to-speech
            tts = gTTS(text=self.data, lang=self.lang)
            tts.save('tmp.mp3')
            os.system('mpg321 tmp.mp3 > /dev/null 2>&1')  # TODO: use subprocess instead
            os.system('rm tmp.mp3')
            # subprocess.call(['mpg321 tmp.mp3'])
            # subprocess.call(['rm tmp.mp3'])
            self.updated = False
            print("Speech generated!")
            return self.data

    def update(self, data):
        """Update the data."""
        self.data = data


class TranslatorInterface(InputOutputInterface):
    r"""Text Translator Interface
    """

    available_languages = {'french', 'english', 'american english', 'british english', 'indian english', 'italian',
                           'japanese', 'korean', 'german', 'dutch', 'spanish', 'spanish (peru)', 'chinese',
                           'mandarin', 'polish', 'portuguese', 'russian', 'greek'}

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
        self._input_data = ''
        self.data = ''

        super(TranslatorInterface, self).__init__(use_thread, verbose=verbose)

    @property
    def input_data(self):
        return self._input_data

    @input_data.setter
    def input_data(self, data):
        if isinstance(data, (str, unicode)):
            self._input_data = data
            self.updated = True

    def run(self):
        """Run the interface."""
        # translate
        if self.updated and self.target_lang != self.from_lang:
            if self.verbose:
                print("Translating: {}".format(''.join(self.input_data).encode('utf-8')))
            translated = self.translator.translate(self.input_data, dest=self.target_lang, src=self.from_lang)
            self.data = translated.text
            if self.verbose:
                print("Translated text: {}".format(''.join(self.data).encode('utf-8')))
            self.updated = False
            return self.data

    def update(self, data):
        """Update the data."""
        self.input_data = data


class SpeechTranslatorInterface(InputOutputInterface):
    r"""Speech Translator Interface

    This class performs speech recognition, translation, and synthesization.
    """

    available_languages = {'french', 'english', 'american english', 'british english', 'indian english', 'italian',
                           'japanese', 'korean', 'german', 'dutch', 'spanish', 'spanish (peru)', 'chinese',
                           'mandarin', 'polish', 'portuguese', 'russian', 'greek'}

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
        self.recognizer = SpeechRecognizerInterface(use_thread=False, lang=from_lang, verbose=verbose)
        self.translator = None
        if target_lang != from_lang:
            self.translator = TranslatorInterface(use_thread=False, target_lang=target_lang,
                                                  from_lang=from_lang, verbose=verbose)
        self.synthesizer = SpeechSynthesizerInterface(use_thread=False, lang=target_lang, verbose=verbose)

        self.updated = False
        self.input_data = ''
        self.output_data = ''

        super(SpeechTranslatorInterface, self).__init__(use_thread, verbose=verbose)

    def run(self):
        """Run the interface."""

        # run the recognizer
        self.input_data = self.recognizer.run()

        # run the translator
        if self.translator is not None:
            self.translator.input_data = self.input_data
            self.output_data = self.translator.run()
        else:
            self.output_data = self.input_data

        # run the synthesizer
        if self.input_data is not None:
            self.synthesizer.data = self.output_data
            self.synthesizer.run()

        return self.output_data


# Tests
if __name__ == '__main__':

    print("Available languages are: {}".format(SpeechRecognizerInterface.available_languages))

    # interface = SpeechRecognizerInterface(verbose=True, lang='english')
    interface = SpeechTranslatorInterface(verbose=True, from_lang='french', target_lang='english')

    while True:
        data = interface.run()
