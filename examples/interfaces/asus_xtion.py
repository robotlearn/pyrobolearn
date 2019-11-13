#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the Asus Xtion interface.

Make sure that the `openni` library is installed with all the correct environment variables set, and that the Asus
Xtion is connected before running this code. Note that it can take some time to initialize the interface.
"""

import argparse
import matplotlib.pyplot as plt

from pyrobolearn.tools.interfaces.camera.asus_xtion import AsusXtionInterface


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--use_thread', help='If we should run the webcam interface in a thread.', type=bool,
                    default=False)
parser.add_argument('-r', '--use_rgb', help='If we should get RGB images. Note that RGB and IR images can not be '
                                            'captured at the same time.', type=bool,
                    default=True)
parser.add_argument('-d', '--use_depth', help='If we should get depth images.', type=bool,
                    default=True)
parser.add_argument('-i', '--use_ir', help='If we should get IR images. Note that RGB and IR images can not be '
                                           'captured at the same time.', type=bool,
                    default=False)
args = parser.parse_args()


# get which pictures to capture
use_rgb, use_depth, use_ir = args.use_rgb, args.use_depth, args.use_ir
if use_rgb and use_ir:
    use_ir = False


# create Asus Xtion interface
interface = AsusXtionInterface(use_rgb=use_rgb, use_depth=use_depth, use_ir=use_ir)


# plotting using matplotlib in interactive mode
fig, axes = plt.subplots(1, 2)
plots = [None]*2
titles = []
if use_rgb:
    titles.append('RGB')
if use_ir:
    titles.append('IR')
if use_depth:
    titles.append('Depth')

plt.ion()  # interactive mode on

while True:
    # if don't use thread call `step` or `run`
    data = interface.run()

    # get the frame and plot it with matplotlib
    if plots[0] is None:
        for i in range(len(plots)):
            plots[i] = axes[i].imshow(data[i])
            axes[i].set_title(titles[i])
    else:
        for plot, img in zip(plots, data):
            plot.set_data(img)

        # pause a bit
        plt.pause(0.01)

    # check if the figure is closed, and if so, get out of the loop
    if not plt.fignum_exists(fig.number):
        break

plt.ioff()  # interactive mode off
plt.show()
