#!/usr/bin/env python
"""Run the Openpose interface.

Make sure that the webcam is connected, and that the openpose framework has been installed before running this code.
For this code to work, you have to specify the path to the openpose framework, or set the `OPENPOSE_PATH` environment
variable. Note that it can take some time to initialize the interface.
"""

import cv2
import argparse

from pyrobolearn.tools.interfaces.camera.openpose import OpenPoseInterface


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Absolute path to the openpose framework. If not specified, it will check '
                                         'for the environment variable `OPENPOSE_PATH`.', type=str, default='')
parser.add_argument('-t', '--use_thread', help='If we should run the openpose interface in a thread.', type=bool,
                    default=False)
parser.add_argument('-f', '--detect_face', help='If we should detect the face with openpose.', type=bool,
                    default=True)
parser.add_argument('-a', '--detect_hands', help='If we should detect the hands with openpose.', type=bool,
                    default=False)
args = parser.parse_args()
path = None if args.path == '' else args.path


# create openpose interface
if args.use_thread:
    # create and run interface in a thread
    interface = OpenPoseInterface(openpose_path=path, detect_face=args.detect_face, detect_hands=args.detect_hands,
                                  use_thread=True, sleep_dt=1. / 10, verbose=True)
    raw_input('Press key to stop the openpose interface')
else:
    # create interface
    interface = OpenPoseInterface(openpose_path=path, detect_face=args.detect_face, detect_hands=args.detect_hands)

    # run interface
    while True:
        frame, keypoints = interface.run()
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", frame)

        # quit display if 'esc' button is pressed
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            cv2.destroyWindow('frame')
