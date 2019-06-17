#!/usr/bin/env python
"""Provide the Kinect input interface.
"""

# import kinect library

# select which library we want from {freenect, openni, pykinect}
# - freenect is a lower-level library which, in addition to access depth and color images, it also allows you to
#   access to various the Kinect hardware (such as the sensors (e.g. accelerometer) and actuators (e.g. LED control)
#   on the Kinect).
# - openni is a higher-level library which, in addition to depth and color images, it allows you to perform high-level
#   tasks such as skeleton tracking, segmentation, gesture recognition, and others. It also allows you to use other
#   devices such as the asus xtion as well.
#   --> Note that there is an `OpenNI2-FreenectDriver` in the libfreenect repo, which is a bridge to libfreenect
#       implemented as an OpenNI2 driver. It allows OpenNI2 to use Kinect hardware on Linux and OSX.
# - pykinect is a Windows library that allows the user to access the kinect (but cannot be used on Unix systems)
#
# For more info, check the following links:
# - https://robotics.stackexchange.com/questions/565/kinect-libfreenect-vs-opennisensorkinect
# - https://stackoverflow.com/questions/19181332/libfreenect-vs-openni

import numpy as np
import cv2

# by default, use `openni` (optionally with the freenect driver) as it seems to be the most complete library
KINECT_LIBRARY = 'freenect'

if KINECT_LIBRARY[-8:] == 'freenect':  # 'libfreenect' or 'freenect'
    # References:
    # - OpenKinect: https://openkinect.org/wiki/Getting_Started
    # - tuto: naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python-and-open-kinect-libfreenect/

    try:
        import freenect
    except ImportError as e:
        raise ImportError(repr(e) + '\nTry to install `libfreenect` (manually in order to have the python wrappers):'
                                    '\n# install dependencies'
                                    '\nsudo apt-get install git-core cmake libglut3-dev pkg-config build-essential '
                                    'libxmu-dev libxi-dev libusb-1.0-0-dev'
                                    '\n# clone the repo'
                                    '\ngit clone git://github.com/OpenKinect/libfreenect.git'
                                    '\nsudo python setup.py install'
                                    '\n# build the repo and install it'
                                    '\ncd libfreenect; mkdir build; cd build;'
                                    '\ncmake ..; make'
                                    '\nsudo make install'
                                    '\n\n# Install Python wrappers'
                                    '\ncd ../wrappers/python'
                                    '\nsudo python setup.py install')

    # If using `freenect`, connect the kinect to your computer, and type the following in the terminal:
    # $ freenect-glview
    # to check if it is correctly detected and working properly.

elif KINECT_LIBRARY == 'openni':
    # References:
    # - Installation: https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-openni-nite.md
    # - OpenNI: https://structure.io/openni
    # - Github repo: https://github.com/occipital/openni2
    # - Python wrappers: https://github.com/severin-lemaignan/openni-python
    # - OpenNI2-FreenectDriver: https://github.com/OpenKinect/libfreenect/tree/master/OpenNI2-FreenectDriver
    if __name__ == '__main__':
        try:
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

    # Checks:
    # - To check if `OpenNI` is correctly installed and work, check `NiViewer` binary application in the `OpenNI`
    #   package
    # - To check if `Nite` is correctly installed and work, check `UserViewer` and `HandViewer` binaries in
    #   the `Nite2` package

    # Troubleshootings:
    # - sudo ln -s /lib/x86_64 ..
    # - Not detected


elif KINECT_LIBRARY == 'pykinect':  # WARNING: ONLY WORKS ON WINDOWS
    # Reference:
    # - https://github.com/Microsoft/PTVS/wiki/PyKinect
    # - https://possiblywrong.wordpress.com/2012/11/04/kinect-skeleton-tracking-with-visual-python/
    try:
        import pykinect
    except ImportError as e:
        raise ImportError(repr(e) + '\nTry to install `pykinect`: pip install pykinect')

else:
    raise ValueError("The given KINECT_LIBRARY variable is not known, please select between {freenect, openni, "
                     "pykinect}")


# import interface
from pyrobolearn.tools.interfaces.camera import CameraInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KinectInterface(CameraInterface):
    r"""Kinect Interface

    This defines the kinect interface class.

    There are 3 different libraries `freenect`, `openni`, and `pykinect` that can be used.
    - `freenect`: it is a lower-level library which, in addition to access depth and color images, it also allows you
    to access to various the Kinect hardware (such as the sensors (e.g. accelerometer) and actuators (e.g. LED control)
    on the Kinect).
    - `openni`: it is a higher-level library which, in addition to depth and color images, it allows you to perform
    high-level tasks such as skeleton tracking, segmentation, gesture recognition, and others. Note that there is an
    `OpenNI2-FreenectDriver` in the libfreenect repo, which is a bridge to libfreenect implemented as an OpenNI2
    driver. It allows OpenNI2 to use Kinect hardware on Linux and OSX.
    - `pykinect`: it is a Windows library that allows the user to access the kinect (but cannot be used on Unix
    systems)

    References:
        [1] libfreenect:
            - Wiki: https://openkinect.org/wiki/Getting_Started
            - Github repo: https://github.com/OpenKinect/libfreenect
        [2] OpenNI:
            - Homepage: https://structure.io/openni
            - Github repo: https://github.com/occipital/openni2
            - Openni-Python: https://github.com/severin-lemaignan/openni-python
            - OpenNI2-FreenectDriver: https://github.com/OpenKinect/libfreenect/tree/master/OpenNI2-FreenectDriver
        [3] PyKinect: https://github.com/Microsoft/PTVS/wiki/PyKinect
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False):
        """
        Initialize the Kinect input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring
                the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(KinectInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)


class FreenectKinectInterface(KinectInterface):
    r"""Freenect Kinect Interface

    Kinect interface using the `freenect` library.

    References:
        [1] libfreenect:
            - Wiki: https://openkinect.org/wiki/Getting_Started
            - Github repo: https://github.com/OpenKinect/libfreenect
        [2] Tutorial: naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python\
                       -and-open-kinect-libfreenect/
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False):
        """
        Initialize the Kinect input interface using the `freenect` library.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring
                the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        # data
        self.rgb = None
        self.depth = None

        super(FreenectKinectInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    def get_image(self, convert_to=None):  # cv2.COLOR_RGB2BGR):
        """Get the RGB image.

        Args:
            convert_to (int): if the picture must be converted to another format using `cv2.COLOR`

        Returns:
            np.array[width, height, 3]: RGB image.
        """
        array, _ = freenect.sync_get_video()
        if convert_to is not None:
            array = cv2.cvtColor(array, convert_to)
        return array

    def get_depth(self):
        """Get the depth image.

        Returns:
            np.array[width, height]: depth image.
        """
        array, _ = freenect.sync_get_depth()
        array = array.astype(np.uint8)
        return array

    def run(self):
        """Run the interface; get the RGB and depth images.

        Returns:
            np.array[width, height, 3]: RGB image
            np.array[width, height]: depth image
        """
        self.rgb = self.get_image()
        self.depth = self.get_depth()

        return self.rgb, self.depth


class OpenNIKinectInterface(KinectInterface):
    r"""OpenNI Kinect Interface

    Kinect interface using the `openni` library.

    References:
        [1] https://github.com/danielelic/PyOpenNI2-Utility
        [2] https://github.com/kanishkaganguly/OpenNIMultiSensorCapture
        [3] https://docs.opencv.org/3.1.0/d7/d6f/tutorial_kinect_openni.html
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False):
        """
        Initialize the Kinect input interface using the `openni` library.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring
                the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        # initialize openni2; you can give the path to the library as an argument. Otherwise, it will look for
        # OPENNI2_REDIST and OPENNI2_REDIST64 environment variables.
        openni2.initialize()

        # open all the devices
        devices = openni2.Device.open_all()

        # get the correct device (Microsoft Kinect)
        self.device = None
        for device in devices:
            info = device.get_device_info()
            if info.vendor == 'Microsoft' and info.name == 'Kinect':  # Kinect Interface
                self.device = device
                break

        # If didn't find it, return an error
        if self.device is None:
            devices = [device.get_device_info() for device in devices]
            raise ValueError("No Asus devices were detected; we found these devices instead: {}".format(devices))

        if verbose:
            print(self.device.get_device_info())

        # create RGB and depth streams
        self.rgb_stream = self.device.create_color_stream()
        self.depth_stream = self.device.create_depth_stream()

        # start the streams
        self.rgb_stream.start()
        self.depth_stream.start()

        # data
        self.rgb = None
        self.depth = None

        super(OpenNIKinectInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    def run(self):
        """Run the interface; get the RGB and depth images.

        Returns:
            np.array[width, height, 3]: RGB image
            np.array[width, height]: depth image
        """
        # read frames
        rgb_frame = self.rgb_stream.read_frame()
        depth_frame = self.depth_stream.read_frame()

        # get buffers
        rgb = rgb_frame.get_buffer_as_uint8()
        depth = depth_frame.get_buffer_as_uint16()

        # convert from buffers to images and reshape them
        rgb = np.frombuffer(rgb, dtype=np.uint8)
        rgb = rgb.reshape(rgb_frame.height, rgb_frame.width, 3)
        depth = np.frombuffer(depth, dtype=np.uint16)
        depth = depth.reshape(depth_frame.height, depth_frame.width)

        # save images and return them
        self.rgb, self.depth = rgb, depth
        return self.rgb, self.depth

    def __del__(self):
        """Delete the Kinect interface."""
        # close all the streams
        self.rgb_stream.close()
        self.depth_stream.close()

        # unload openni2
        openni2.unload()


# TODO: add pose and gesture types
class KinectSkeletonTrackingInterface(KinectInterface):
    r"""Skeleton tracking

    Skeleton tracking using the openni and nite libraries.

    References:
        [1] https://github.com/severin-lemaignan/openni-python
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False, track_hand=False):
        """
        Initialize the Kinect input interface using the `openni` library.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring
                the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
            track_hand (bool): If True, it will track the hands.
        """
        # initialize openni2 and nite2; you can give the path to the library as an argument.
        # Otherwise, it will look for OPENNI2_REDIST / OPENNI2_REDIST64 and NITE2_REDIST / NITE2_REDIST64 environment
        # variables.
        openni2.initialize()
        nite2.initialize()

        # open all the devices
        devices = openni2.Device.open_all()

        # get the correct device (Microsoft Kinect)
        self.device = None
        for device in devices:
            info = device.get_device_info()
            if info.vendor == 'Microsoft' and info.name == 'Kinect':  # Kinect Interface
                self.device = device
                break

        # If didn't find it, return an error
        if self.device is None:
            devices = [device.get_device_info() for device in devices]
            raise ValueError("No Asus devices were detected; we found these devices instead: {}".format(devices))

        if verbose:
            print(self.device.get_device_info())

        # create tracker for the hand or user depending on the given parameter
        if track_hand:
            self.tracker = nite2.HandTracker(self.device)
        else:
            self.tracker = nite2.UserTracker(self.device)

        # data
        self.joints = ['head', 'neck', 'torso', 'left_shoulder', 'left_elbow', 'left_hand', 'left_hip', 'left_knee',
                       'left_foot', 'right_shoulder', 'right_elbow', 'right_hand', 'right_hip', 'right_knee',
                       'right_foot']
        joint = nite2.JointType
        self.nite_joints = [joint.NITE_JOINT_HEAD, joint.NITE_JOINT_NECK, joint.NITE_JOINT_TORSO,
                            joint.NITE_JOINT_LEFT_SHOULDER, joint.NITE_JOINT_LEFT_ELBOW, joint.NITE_JOINT_LEFT_HAND,
                            joint.NITE_JOINT_LEFT_HIP, joint.NITE_JOINT_LEFT_KNEE, joint.NITE_JOINT_LEFT_FOOT,
                            joint.NITE_JOINT_RIGHT_SHOULDER, joint.NITE_JOINT_RIGHT_ELBOW, joint.NITE_JOINT_RIGHT_HAND,
                            joint.NITE_JOINT_RIGHT_HIP, joint.NITE_JOINT_RIGHT_KNEE, joint.NITE_JOINT_RIGHT_FOOT]

        self.data = {}

        super(KinectSkeletonTrackingInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt,
                                                              verbose=verbose)

    def run(self):
        """Run the interface; get the skeleton data.

        Returns:
            dict: skeleton data
        """
        # read frame
        frame = self.tracker.read_frame()

        # check if users in the frame
        if frame.users:
            # for each user in the frame
            for user in frame.users:
                # check if it is a new one
                if user.is_new():
                    if self.verbose:
                        print("New user detected! Calibrating...")
                    self.tracker.start_skeleton_tracking(user.id)
                    self.data = {user.id: {}}

                # check that the state has been correctly tracked
                elif user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED:

                    # go through each joint and update the user data
                    for joint, nite_joint in zip(self.joints, self.nite_joints):
                        j = user.skeleton.joints[nite_joint]
                        self.data[user.id][joint] = (j.position.x, j.position.y, j.position.z, j.positionConfidence)

                    # self.data[user.id]['updated'] = True

        return self.data

    def __del__(self):
        """Delete the Kinect interface."""
        # unload nite2 and openni2
        nite2.unload()
        openni2.unload()


class ROSKinectInterface(KinectInterface):
    r"""ROS Kinect Interface

    References:
        [1] http://wiki.ros.org/openni_camera
    """
    pass


# Tests
if __name__ == '__main__':
    pass
