
from pyrobolearn.tools.interfaces.vr import VRInterface


class HTCViveInterface(VRInterface):
    r"""HTC Vive Interface

    To install:
    * SteamVR
    * OpenVR
    * PyOpenVR

    References:
        [1] OpenVR: https://github.com/ValveSoftware/openvr
        [2] PyOpenVR: https://github.com/cmbruns/pyopenvr
            with wiki: https://github.com/cmbruns/pyopenvr/wiki/API-Documentation
        [3] https://github.com/osudrl/CassieVrControls/wiki/OpenVR-Quick-Start
        [4] https://github.com/osudrl/OpenVR-Tracking-Example
        [5] How to use the HTC Vive Trackers in Ubuntu using Python 3.6:
            https://gist.github.com/DanielArnett/c9a56c9c7cc0def20648480bca1f6772
    """
    def __init__(self, world, use_controllers=True, use_headset=True):
        super(HTCViveInterface, self).__init__()