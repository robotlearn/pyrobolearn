# -*- coding: utf-8 -*-

import logging

# import exploration
from .exploration import *

# import action exploration
from .actions import *

# import parameter exploration
from .parameters import *

# create logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
