#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the speaker audio interface.
"""

import os

# Speech synthesis
# Good tutorial: https://pythonprogramminglanguage.com/text-to-speech/
# If Python3.3 or higher: https://pypi.org/project/google_speech/
try:
    from gtts import gTTS
except ImportError as e:
    string = "\nHint: try to install gTTS by typing: `pip install gTTS`." \
             "Also install `mpg321` using `sudo apt-get install mpg321`."
    raise ImportError(e.__str__() + string)

from pyrobolearn.tools.interfaces import OutputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SpeakerInterface(OutputInterface):
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

        super(SpeakerInterface, self).__init__(use_thread=use_thread, verbose=verbose)

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
