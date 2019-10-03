# -*- coding: utf-8 -*-
# a bridge links an interface with something else

# general bridge import
from .bridge import Bridge

# Bridge for mouse-keyboard interfaces
from .mouse_keyboard import *

# Bridge for audio interfaces
from . import audio

# Bridge for camera interfaces
from . import camera

# Bridge for controller interfaces
from . import controllers

# Bridge for VR interfaces
from . import vr
