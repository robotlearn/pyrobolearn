#!/usr/bin/env python
"""Define the Openpose Interface

This extracts the human skeleton from an image or stream of images (from a webcam for instance) using the openpose
library. See `https://github.com/CMU-Perceptual-Computing-Lab/openpose` for more info about openpose.
"""

import os
import cv2
# import PyOpenPose as OP
try:
    import pyopenpose
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `pyopenpose` by installing the openpose library. Check the script '
                                '`pyrobolearn/scripts/install_openpose.sh` to install the library and the associated '
                                'python wrapper.')

from pyrobolearn.tools.interfaces.camera import CameraInterface
from pyrobolearn.tools.interfaces.camera.webcam import WebcamInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class OpenPoseInterface(CameraInterface):
    r"""OpenPose interface

    This class defines the OpenPose interface which uses a camera interface like a webcam or kinect to get the (2D or
    3D) pictures and map them to the human kinematic skeleton.

    References:
        [1] OpenPose: github.com/CMU-Perceptual-Computing-Lab/openpose
        [2] PyOpenPose (official): github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md
        [3] OpenPose output format: github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        [4] PyOpenPose (python wrappers): github.com/FORTH-ModelBasedTracker/PyOpenPose
    """

    def __init__(self, camera=None, detect_face=False, detect_hands=False, openpose_path=None,
                 use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the openpose camera input interface.

        Args:
            camera (None, CameraInterface): camera interface to get the images from.
            detect_face (bool): if True, it will also detect the face keypoints.
            detect_hands (bool): if True, it will also detect the hand keypoints.
            openpose_path (str, None): path to the Openpose folder. If None, it will get the `OPENPOSE_PATH` bash
                environment variable.
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring the
                next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """

        # save variables
        self.detect_face = detect_face
        self.detect_hands = detect_hands

        # Check the given camera
        if camera is None:
            # If None, get pictures from a webcam
            camera = WebcamInterface(use_thread=False, convert_to=None, verbose=False)
            self.camera_in_openpose = True
        else:
            self.camera_in_openpose = False

        self.camera = camera

        # Define the JOINTS
        # define the 25 BODY joints
        self.body_joints = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip',
                            'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar',
                            'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel', 'Background']
        self.body_joint_names_to_ids = dict(zip(self.body_joints, range(len(self.body_joints))))

        # define the 21 HAND joints
        # for each of the 4 main finger(s) (without thumb), there are 4 joints (proximal, middle, distal, tip)
        self.hand_joints = ['Palm'] + ['Thumb' + str(i) for i in range(4)] + ['Index' + str(i) for i in range(4)] + \
                           ['Middle' + str(i) for i in range(4)] + ['Ring' + str(i) for i in range(4)] + \
                           ['Little' + str(i) for i in range(4)]
        self.hand_joint_names_to_ids = dict(zip(self.hand_joints, range(len(self.hand_joints))))

        # define the 70 FACE joints
        # Note that in github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_face.png,
        # the joints are not symmetric but are read from left to right.
        # The joints are defined as:
        # face (beard): [0, 16]
        # right eyebrow: [17, 21]
        # left eyebrow: [22,26]
        # nose: [27, 35]
        # right eye: [36, 41] + 68
        # left eye: [42, 47] + 69
        # mouth: [48, 67]
        self.face_joints = ['Face' + str(i) for i in range(17)] + ['REyebrow' + str(i-17) for i in range(17, 22)] + \
                           ['LEyebrow' + str(i-22) for i in range(22, 27)] + \
                           ['Nose' + str(i-27) for i in range(27, 36)] + \
                           ['REye' + str(i-36) for i in range(36, 42)] + \
                           ['LEye' + str(i-42) for i in range(42, 48)] + \
                           ['Mouth' + str(i-48) for i in range(48, 68)] + ['REye6'] + ['LEye6']
        self.face_joint_names_to_ids = dict(zip(self.face_joints, range(len(self.face_joints))))

        # Check the path to the openpose folder (which contains various models and test images)
        if openpose_path is None:
            if 'OPENPOSE_PATH' not in os.environ:
                raise ValueError("The OPENPOSE_PATH environment variable has not been set properly. Please "
                                 "then provide the path to the openpose folder by specifying the `openpose_path` "
                                 "argument")
            openpose_path = os.environ['OPENPOSE_PATH']
        self.openpose_path = openpose_path

        # specify the parameters
        params = dict()
        params["model_folder"] = path + "models/"
        if detect_face:
            params["face"] = True
        if detect_hands:
            params["hands"] = True

        # configure openpose
        self.openpose = pyopenpose.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()

        # define data holder
        self.datum = pyopenpose.Datum()

        # define frame
        self.frame = None

        # call superclass constructor
        super(OpenPoseInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    @property
    def num_bodies(self):
        """Get the number of detected bodies."""
        return len(self.datum.poseKeypoints)

    @property
    def body_keypoints(self):
        """Return the keypoints (x,y,confidence) for each body part for each person"""
        return self.datum.poseKeypoints

    @property
    def left_hand_keypoints(self):
        """Get the left hand keypoints."""
        return self.datum.handKeypoints[0]

    @property
    def right_hand_keypoints(self):
        """Get the right hand keypoints."""
        return self.datum.handKeypoints[1]

    @property
    def hand_keypoints(self):
        """Get the hand keypoints."""
        return self.left_hand_keypoints, self.right_hand_keypoints

    @property
    def face_keypoints(self):
        """Get the face keypoints."""
        return self.datum.faceKeypoints

    @property
    def keypoints(self):
        """Get the keypoints."""
        return self.body_keypoints, self.face_keypoints, self.left_hand_keypoints, self.right_hand_keypoints

    @property
    def heatmap(self):
        """Get the heatmap."""
        return None

    @property
    def input_image(self):
        """Get the input image."""
        return self.datum.cvInputData

    @property
    def output_image(self):
        """Get the output image."""
        return self.datum.cvOutputData

    @property
    def num_gpus(self):
        """Return the number of GPUs."""
        return pyopenpose.get_gpu_number()

    def run(self, input_frame=None):
        """Run the interface."""
        if input_frame is None:
            # read image
            if self.camera_in_openpose:
                self.camera.run()
            img = self.camera.frame
        else:
            if isinstance(input_frame, str):
                img = cv2.imread(input_frame)
            else:
                img = input_frame

        # define data holder
        self.datum = pyopenpose.Datum()

        # process image
        self.datum.cvInputData = img
        self.openpose.emplaceAndPop([self.datum])

        # save frame
        self.frame = self.datum.cvOutputData

        # define dictionary of keypoints
        keypoints = dict()
        keypoints['body'] = self.body_keypoints
        if self.detect_face:
            keypoints['face'] = self.face_keypoints
        if self.detect_hands:
            keypoints['left_hand'] = self.left_hand_keypoints
            keypoints['right_hand'] = self.right_hand_keypoints

        # display image if specified
        if self.verbose:
            cv2.imshow('OpenPose frame', self.frame)

            # quit display if 'esc' button is pressed
            key = cv2.waitKey(15) & 0xFF
            if key == 27:
                self.verbose = False
                cv2.destroyWindow('frame')

        # return frame and the associated keypoints
        return self.frame, keypoints


# Tests
if __name__ == '__main__':
    path = '/home/brian/repos/openpose/'
    interface = OpenPoseInterface(openpose_path=path)  # , use_thread=False, sleep_dt=1./10, verbose=True)

    while True:
        frame, keypoints = interface.run()
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", frame)
        cv2.waitKey(15)
