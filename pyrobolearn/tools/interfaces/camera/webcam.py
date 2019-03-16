#!/usr/bin/env python
"""Define the Webcam Interface

This provides the main interface to get pictures from the specified webcam.
"""


import cv2  # OpenCV to capture image from webcam
import os

from camera import CameraInterface

# to close correctly the webcam once we run
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# For multithreading, check also:
# - http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/
# - https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WebcamInterface(CameraInterface):
    r"""Webcam Interface

    This class defines the webcam interface. It gets pictures from the webcam, which can then be processed by another
    tool (such as openpose).

    References:
        [1] https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    """

    def __init__(self, webcamId=0, saveVideo=False, filename='output.avi', fps=20, frameSize=(640,480), codec='XVID',
                 convertTo=cv2.COLOR_BGR2RGB, use_thread=False, sleep_dt=0, verbose=False):

        # create video capture object
        self.capture = cv2.VideoCapture(webcamId)

        # create video writer
        if saveVideo:
            # define codec (DIVX, XVID, MJPG, X264, WMV1, WMV2)
            codec = cv2.VideoWriter_fourcc(*codec)
            # define video writer
            self.writer = cv2.VideoWriter(filename, codec, fps, frameSize)
        else:
            self.writer = None

        # variables
        self.convertTo = convertTo
        self.verbose = False

        # camera image
        self.frame = self.run()

        # call superclass constructor
        super(WebcamInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    def run(self):  # , display=True, convertToGray=False):
        # get the frame from the webcam
        return_code, frame = self.capture.read()

        if not return_code:
            raise ValueError("There was an error when reading the frame from the webcam")

        # convert to gray (if specified)
        if self.convertTo == cv2.COLOR_BGR2GRAY:
            frame = cv2.cvtColor(frame, self.convertTo)

        # display image if specified
        if self.verbose:
            cv2.imshow('frame', frame)

            # quit display if 'esc' button is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.verbose = False
                cv2.destroyWindow('frame')

        # convert to the specified format
        if self.convertTo is not None:
            frame = cv2.cvtColor(frame, self.convertTo)

        # save the frame for the video
        if self.writer is not None:
            self.writer.write(frame)

        # set and return the frame
        self.frame = frame
        return frame

    def __del__(self):
        self.capture.release()
        if self.writer is not None:
            self.writer.release()
        # cv2.destroyAllWindows()


# Test
if __name__ == '__main__':
    from itertools import count
    import matplotlib.pyplot as plt

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
