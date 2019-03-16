
from pyrobolearn.tools.interfaces.interface import InputOutputInterface


class GameControllerInterface(InputOutputInterface):
    r"""Game Controller Interface

    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        super(GameControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
