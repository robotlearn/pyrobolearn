
from camera import CameraInterface


class FERInterface(CameraInterface):
    r"""Facial Expression Recognition (FER) Interface

    References:
        [1] EmoPy - A deep neural net toolkit for emotion analysis via Facial Expression Recognition:
            https://github.com/thoughtworksarts/EmoPy
        [2] http://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/
        [3] https://github.com/a514514772/Real-Time-Facial-Expression-Recognition-with-DeepLearning
    """

    def __init__(self):
        super(FERInterface, self).__init__()
