# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Run the Webcam interface.

Make sure that the webcam is connected before running this code. Note that it can take some time to initialize
the interface.
"""

import argparse
import matplotlib.pyplot as plt

from pyrobolearn.tools.interfaces.camera.webcam import WebcamInterface


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--use_thread', help='If we should run the webcam interface in a thread.', type=bool,
                    default=False)
args = parser.parse_args()


# create webcam interface
if args.use_thread:
    # create and run interface in a thread
    interface = WebcamInterface(use_thread=True, sleep_dt=1./10, verbose=True)
    raw_input('Press key to stop the webcam interface')
else:
    # create interface
    interface = WebcamInterface()

    # plotting using matplotlib in interactive mode
    fig = plt.figure()
    plot = None
    plt.ion()   # interactive mode on

    while True:
        # perform a `step` with the interface
        interface.step()

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
