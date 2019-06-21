#!/usr/bin/env python
"""Run the speech interface.

This will perform speech recognition, translation, and synthesization. Make sure that your computer has a microphone
connected.

In the future, interfaces using the Google assistant, Alexa, or a similar tool will be implemented.
"""

import argparse

from pyrobolearn.tools.interfaces.audio.speech import SpeechRecognizerInterface, SpeechTranslatorInterface


# get the available languages.
languages = SpeechRecognizerInterface.available_languages
print("Available languages are: {}".format(languages))

# create parser
parser = argparse.ArgumentParser()
# parser.add_argument('-t', '--use_thread', help='If we should run the webcam interface in a thread.', type=bool,
#                     default=False)
parser.add_argument('-l', '--lang', help='The language that needs to be recognized.', type=str, choices=languages,
                    default='english')
parser.add_argument('-a', '--target_lang', help='If we should get depth images.', type=str, choices=languages,
                    default='english')
args = parser.parse_args()


# create speech recognizer/translator interface
# interface = SpeechRecognizerInterface(verbose=True, lang=args.lang)
interface = SpeechTranslatorInterface(verbose=True, from_lang=args.lang, target_lang=args.target_lang)

# run the interface
while True:
    data = interface.run()
