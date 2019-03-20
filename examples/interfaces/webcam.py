#!/usr/bin/env python
"""Load the Webcam interface.
"""

from itertools import count
import matplotlib.pyplot as plt
from pyrobolearn.tools.interfaces.camera.webcam import WebcamInterface

# create interface
interface = WebcamInterface(use_thread=True, sleep_dt=1./10, verbose=False)

# plotting using matplotlib in interactive mode
fig = plt.figure()
plot = None
plt.ion()   # interactive mode on

for _ in count():
    # # if don't use thread call `step` or `run` (note that `run` returns the frame but not
    # interface.step()

    # get the frame and plot it with matplotlib
    frame = interface.frame
    if plot is None:
        plot = plt.imshow(frame)
    else:
        plot.set_data(frame)
        plt.pause(0.01)

    # check if the figure is closed, and if so, get out of the loop
    if not plt.fignum_exists(fig.number):
        break

plt.ioff()  # interactive mode off
plt.show()
