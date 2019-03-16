
from audio import InputAudioInterface


class MicrophoneInterface(InputAudioInterface):
    r"""Microphone Interface.

    References:
        [1] https://github.com/castorini/honk
        [2] https://github.com/llSourcell/tensorflow_speech_recognition_demo/blob/master/speech_data.py
        [3] https://github.com/SeanNaren/deepspeech.pytorch
        [4] https://github.com/awni/speech
        [5] https://github.com/tugstugi/pytorch-speech-commands
        [6] https://cmusphinx.github.io/
        [7] https://realpython.com/python-speech-recognition/
    """
    def __init__(self):
        super(MicrophoneInterface, self).__init__()


if __name__ == '__main__':
    pass
