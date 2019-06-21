#!/usr/bin/env python
"""Define the Asus Xtion input interface.
"""

import numpy as np

try:
    # References:
    # - Installation: https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-openni-nite.md
    # - OpenNI: https://structure.io/openni
    # - Github repo: https://github.com/occipital/openni2
    # - Python wrappers: https://github.com/severin-lemaignan/openni-python
    # - OpenNI2-FreenectDriver: https://github.com/OpenKinect/libfreenect/tree/master/OpenNI2-FreenectDriver

    # from primesense import openni2, nite2
    from openni import openni2, nite2
    from openni.utils import InitializationError

    try:
        openni2.initialize()
        openni2.unload()
    except InitializationError as e:
        raise InitializationError(repr(e) + '\nYou have to export the path to the folder which contains '
                                            'the library libOpenNI2.so. Depending on your architecture, try '
                                            'to type one of the following command in the terminal: '
                                            '\nexport OPENNI2_REDIST=<path/to/libOpenNI2.so/folder>'
                                            '\nexport OPENNI2_REDIST64=<path/to/libOpenNI2.so/folder>')

except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `openni`: pip install openni'
                                '\nTo install the OpenNI2-Freenect driver, install libfreenect manually, and '
                                'check the README in the `libfreenect/OpenNI2-FreenectDriver` folder')


from pyrobolearn.tools.interfaces.camera import CameraInterface

# check https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-openni-nite.md

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class AsusXtionInterface(CameraInterface):
    r"""Asus Xtion Interface

    Check OpenNI. Check also ROS.

    If using `openni`, connect the Asus Xtion to your computer, and type the following in the terminal:
    $ NiViewer
    to check if it is correctly detected and working properly.

    References:
        [1] https://github.com/danielelic/PyOpenNI2-Utility
        [2] https://github.com/kanishkaganguly/OpenNIMultiSensorCapture
        [3] https://docs.opencv.org/3.1.0/d7/d6f/tutorial_kinect_openni.html
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False, use_rgb=True, use_depth=True, use_ir=False):
        """
        Initialize the Asus Xtion input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
            use_rgb (bool): If True, it will get the RGB camera image. Note that if this is enabled, you can not
                 get IR images at the same time.
            use_depth (bool): If True, it will get the depth camera image.
            use_ir (bool): If True, it will get the Infrared image. Note that if this is enabled, you can not get
                RGB images at the same time.
        """

        # quick check
        if use_rgb and use_ir:
            raise ValueError("It is not possible to stream RGB images at the same as IR images, set one to False")

        # set variables
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_ir = use_ir

        # initialize openni2; you can give the path to the library as an argument. Otherwise, it will look for
        # OPENNI2_REDIST and OPENNI2_REDIST64 environment variables.
        openni2.initialize()

        # open all the devices
        devices = openni2.Device.open_all()

        # get the correct device (PrimeSense)
        self.device = None
        for device in devices:
            info = device.get_device_info()
            if info.vendor == 'PrimeSense': # Asus Xtion Interface
                self.device = device
                break

        # If didn't find it, return an error
        if self.device is None:
            devices = [device.get_device_info() for device in devices]
            raise ValueError("No Asus devices were detected; we found these devices instead: {}".format(devices))

        if verbose:
            print(self.device.get_device_info())

        # create RGB, IR, and depth streams
        self.streams = []
        if self.use_rgb:
            self.rgb_stream = self.device.create_color_stream()
            self.streams.append(self.rgb_stream)

        if self.use_ir:
            self.ir_stream = self.device.create_ir_stream()
            self.streams.append(self.ir_stream)

        if self.use_depth:
            self.depth_stream = self.device.create_depth_stream()
            self.streams.append(self.depth_stream)

        # start each stream
        for stream in self.streams:
            stream.start()

        # data
        self.rgb = None
        self.ir = None
        self.depth = None

        super(AsusXtionInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    def run(self):
        data = []
        if self.use_rgb:
            # read frame
            frame = self.rgb_stream.read_frame()
            buffer = frame.get_buffer_as_uint8()
            img = np.frombuffer(buffer, dtype=np.uint8)
            img = img.reshape(frame.height, frame.width, 3)
            self.rgb = img
            data.append(self.rgb)

        if self.use_ir:
            # read frame
            frame = self.ir_stream.read_frame()
            buffer = frame.get_buffer_as_uint16()
            img = np.frombuffer(buffer, dtype=np.uint16)
            img = img.reshape(frame.height, frame.width)
            self.ir = img
            data.append(self.ir)

        if self.use_depth:
            frame = self.depth_stream.read_frame()
            buffer = frame.get_buffer_as_uint16()
            img = np.frombuffer(buffer, dtype=np.uint16)
            img = img.reshape(frame.height, frame.width)
            self.depth = img
            data.append(self.depth)

        return data

    def __del__(self):
        # close all the streams
        for stream in self.streams:
            stream.close()

        # unload openni2
        openni2.unload()


if __name__ == '__main__':
    # https://structure.io/openni
    # https://github.com/occipital/OpenNI2
    # https://www.reddit.com/r/ROS/comments/6qejy0/openni_kinect_installation_on_kinetic_indigo/
    # https://github.com/cjcase/openGeppetto/wiki/Installing-OpenNI-on-Ubuntu
    # https://github.com/jmendeth/PyOpenNI/blob/c7fa4fa01de3bb717ece5d036daaa343fe1c2ca9/examples/record.py
    # https://pypi.org/project/openni/
    # https://roboram.wordpress.com/asus-xtion-pro-live-ubuntu-14-04-installation/
    # https://docs.opencv.org/3.1.0/d7/d6f/tutorial_kinect_openni.html
    # http://euanfreeman.co.uk/pyopenni-and-opencv/
    # WARNINGS: THERE ARE 2 OPENNI: One is pyopenni and the other one is openni
    # Use opencv with openni: https://gist.github.com/joinAero/1f76844278f141cea8338d1118423648
    # `sudo apt-get install libopenni-dev libopenni2-dev libopenni-sensor-primesense-dev`
    # https://github.com/jmendeth/PyOpenNI/wiki/Building-on-Linux

    # capture = cv2.VideoCapture(cv2.CAP_OPENNI)
    # print(capture.get(cv2.CAP_PROP_OPENNI_GENERATOR_PRESENT))
    # capture.set(cv2.CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv2.CAP_OPENNI_VGA_30HZ)
    # print(type(capture))
    # time.sleep(1.)
    # ret = capture.grab()
    # print(ret)
    # ret, frame = capture.read()
    # print(ret)
    # ret, depth = capture.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
    #
    # print(depth.shape)
    # plt.imshow(depth)
    # plt.show()

    from itertools import count
    import matplotlib.pyplot as plt

    # set what we want to use (note you can not get RGB and IR images at the same time)
    use_rgb, use_depth, use_ir = False, True, True

    # create interface
    interface = AsusXtionInterface(use_thread=False, sleep_dt=1. / 10, verbose=True, use_rgb=use_rgb,
                                   use_depth=use_depth, use_ir=use_ir)

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

    for _ in count():
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
