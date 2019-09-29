# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the the `WorldCamera` class.

Get the camera that looks at the world (only available in the simulator).

Dependencies:
- `pyrobolearn.simulators`
"""

import sys
import numpy as np

from pyrobolearn.utils.transformation import get_quaternion_from_matrix, get_rpy_from_matrix, get_rpy_from_quaternion
from pyrobolearn.simulators import Simulator

# define long for Python 3.x
if int(sys.version[0]) == 3:
    long = int

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WorldCamera(object):
    r"""World camera.

    Camera that looks at the world (only available in the simulator).

    The following operations carried out (in the given order) by OpenGL in order to display images seen by the
    camera are:
    * M: Model space --> World space. This transforms the coordinates of each model described in their own local
     frame :math:`[x_{l}, y_{l}, z_{l}, 1]` to world coordinates :math:`[x_{w}, y_{w}, z_{w}, 1]`.
    * V: World space --> View space. This transforms world coordinates :math:`[x_{w}, y_{w}, z_{w}, 1]` into eye
     coordinates :math:`[x_{e}, y_{e}, z_{e}, 1]`. That is, it rotates and translates the world such that it is in
     front of the camera.
    * P: View space --> Projection space. Transforms the eye coordinates into clip coordinates using an orthographic
     or perspective projection. The new coordinates are given by :math:`[x_{c}, y_{c}, z_{c}, w_{c}]`. This is not
     normalized, i.e. w_{c} is not equal to 1. See next operation.
    * norm: Screen space --> NDC space. This normalizes the previous clipped coordinates into
     Normalized Device Coordinates (NDC) where each coordinate is normalized and is between -1 and 1. That is,
     we now have :math:`[x_{n}, y_{n}, z_{n}, 1] = [x_c/w_c, y_c/w_c, z_c, w_c/w_c]`
    * Vp: NDC space --> Screen space. Finally, this maps the previous normalized clip coordinates to pixel
     coordinates :math:`[x_{s}, y_{s}, z_{s}, 1]` where :math:`x_s` (:math:`y_s`) is between 0 and the width
     (height) of the screen respectively, and :math:`z_s` represents the depth which is between 0 and 1.

    References:
        [1] http://www.codinglabs.net/article_world_view_projection_matrix.aspx
        [2] https://learnopengl.com/Getting-started/Coordinate-Systems
        [3] http://www.thecodecrate.com/opengl-es/opengl-transformation-matrices/
        [4] http://learnwebgl.brown37.net/08_projections/projections_perspective.html
    """

    def __init__(self, simulator):
        """
        Initialize the world camera.

        Args:
            simulator:
        """
        self.simulator = simulator

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.sim

    @simulator.setter
    def simulator(self, simulator):
        """Set the simulator instance."""
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given simulator to be an instance of `Simulator`, instead got: "
                            "{}".format(type(simulator)))
        self.sim = simulator

    @property
    def info(self):
        """
        Return all the information about the camera.
        """
        return self.get_debug_visualizer_camera()

    @property
    def width(self):
        """
        Return the width of the pictures (in pixel)
        """
        return self.sim.get_debug_visualizer()[0]

    @property
    def height(self):
        """
        Return the height of the pictures (in pixel)
        """
        return self.sim.get_debug_visualizer()[1]

    @property
    def V(self):
        """
        Return the view matrix, which maps from the world to the view space.
        """
        return self.sim.get_debug_visualizer()[2]

    # alias
    view_matrix = V

    @property
    def Vinv(self):
        """
        Return the inverse of the view matrix
        """
        return np.linalg.inv(self.V)

    @property
    def P(self):
        """
        Return the projection matrix, which maps from the view to the projected/clipped space.
        """
        return self.sim.get_debug_visualizer()[3]

    # alias
    projection_matrix = P

    @property
    def Pinv(self):
        """
        Return the inverse of the projection matrix
        """
        return np.linalg.inv(self.P)

    @property
    def Vp(self):
        """
        Return the viewport matrix, which maps from the normalized clip coordinates to pixel coordinates.
        """
        width, height = self.sim.get_debug_visualizer()[:2]
        return np.array([[width / 2., 0, 0, width / 2.],
                         [0, height / 2., 0, height / 2.],
                         [0, 0, 0.5, 0.5],
                         [0, 0, 0, 1]])

    viewport_matrix = Vp

    @property
    def Vp_inv(self):
        """
        Return the inverse of the viewport matrix.
        """
        return np.linalg.inv(self.Vp)

    @property
    def up_vector(self):
        """
        Return the up axis of the camera in the Cartesian world space coordinates
        """
        return self.sim.get_debug_visualizer()[4]

    @property
    def forward_vector(self):
        """
        Return the forward axis of the camera in the Cartesian world space coordinates.
        """
        return self.sim.get_debug_visualizer()[5]

    @property
    def lateral_vector(self):
        """
        Return the lateral axis of the camera (=cross product between forward and up vectors)
        """
        up_vector, forward_vector = self.sim.get_debug_visualizer()[4:6]
        return np.cross(forward_vector, up_vector)

    @property
    def yaw(self):
        """
        Return the yaw angle of the camera in radian
        """
        return self.sim.get_debug_visualizer()[8]

    @yaw.setter
    def yaw(self, yaw):
        """
        Set the yaw angle of the camera in radian. The yaw angle is positive when looking on the left and negative
        when looking on the right.
        """
        pitch, distance, target_position = self.sim.get_debug_visualizer()[-3:]
        self.reset(distance, yaw, pitch, target_position)

    @property
    def pitch(self):
        """
        Return the pitch angle of the camera in radian
        """
        return self.sim.get_debug_visualizer()[9]

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch angle of the camera in radian. The pitch angle is negative when looking down and positive when
        looking up.
        """
        yaw, _, distance, target_position = self.sim.get_debug_visualizer()[-4:]
        self.reset(distance, yaw, pitch, target_position)

    @property
    def distance(self):
        """
        Return the distance between the camera and the camera target.
        """
        return self.sim.get_debug_visualizer()[10]

    @distance.setter
    def distance(self, distance):
        """
        Set the distance of the camera (in meter) with respect to the target position.
        """
        yaw, pitch, _, target_position = self.sim.get_debug_visualizer()[-4:]
        self.reset(distance, yaw, pitch, target_position)

    @property
    def target_position(self):
        """
        Return the target position of the camera in the Cartesian world space coordinates.
        """
        return self.sim.get_debug_visualizer()[11]

    @target_position.setter
    def target_position(self, position):
        """
        Set the target position of the camera in the Cartesian world space coordinates.
        """
        yaw, pitch, distance = self.sim.get_debug_visualizer()[-4:-1]
        self.reset(distance, yaw, pitch, position)

    @property
    def position(self):
        """
        Return the current position of the camera in the Cartesian world space coordinates.
        """
        Vinv = np.linalg.inv(self.V)  # compute inverse of the view matrix
        position = Vinv[:3, 3]  # the last column is the current position of the camera
        return position

    @position.setter
    def position(self, position):
        """
        Set the position of the camera in the world.
        """
        target = self.target_position
        vector = (target - position)
        distance = np.sqrt(np.sum(vector**2))
        vector = vector / distance
        pitch = np.arcsin(vector[2])  # [-pi/2, pi/2]
        # pitch = np.arctan2(vector[2], vector[1])
        yaw = np.arctan2(vector[1], vector[0])  # [-pi, pi]
        self.reset(distance, yaw, pitch, target)

    @property
    def orientation(self):
        """
        Return the orientation (as a quaternion).
        """
        # based on forward_vector and up_vector
        Vinv = np.linalg.inv(self.V)  # compute inverse of the view matrix
        orientation = get_quaternion_from_matrix(Vinv[:3, :3])
        return orientation

    @orientation.setter
    def orientation(self, orientation):
        """
        Set the orientation (expressed as a quaternion, rotation matrix, or roll-pitch-yaw angles) of the camera.
        """
        # convert the orientation to roll-pitch-yaw angle
        if orientation.shape == (4,):  # quaternion (x,y,z,w)
            rpy = get_rpy_from_quaternion(orientation)
        elif orientation.shape == (3, 3):  # rotation matrix
            rpy = get_rpy_from_matrix(orientation)
        elif orientation.shape == (3,):  # roll-pitch-yaw angle
            rpy = orientation
        else:
            raise ValueError("Expecting the given orientation to be a quaternion, rotation matrix, or Roll-Pitch-Yaw "
                             "angles, instead got: {}".format(orientation))

        # reset the camera
        distance, target_position = self.sim.get_debug_visualizer()[-2:]
        _, pitch, yaw = rpy
        self.reset(distance, yaw, pitch, target_position)

    ###########
    # Methods #
    ###########

    def get_debug_visualizer_camera(self):
        """
        Return all the information provided by the camera.

        Returns:
            int: width of the visualizer camera (in pixel)
            int: height of the visualizer camera (in pixel)
            np.array[float[4,4]]: view matrix [4,4]
            np.array[float[4,4]]: perspective projection matrix [4,4]
            np.array[float[3]]: camera up vector expressed in the Cartesian world space
            np.array[float[3]]: forward axis of the camera expressed in the Cartesian world space
            np.array[float[3]]: This is a horizontal vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            np.array[float[3]]: This is a vertical vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            float: yaw angle (in radians) of the camera, in Cartesian local space coordinates
            float: pitch angle (in radians) of the camera, in Cartesian local space coordinates
            float: distance between the camera and the camera target
            np.array[float[3]]: target of the camera, in Cartesian world space coordinates
        """
        return self.sim.get_debug_visualizer()

    def reset(self, distance=None, yaw=None, pitch=None, target_position=None):
        """Reset the debug visualizer camera.

        Reset the 3D OpenGL debug visualizer camera distance (between eye and camera target position), camera yaw and
        pitch and camera target position

        Args:
            distance (float, None): distance from eye to camera target position. If None, it will take the current
                distance.
            yaw (float, None): camera yaw angle (in radians) left/right. If None, it will take the current yaw angle.
            pitch (float, None): camera pitch angle (in radians) up/down. If None, it will take the current pitch angle.
            target_position (np.array[float[3]], None): target focus point of the camera. If None, it will take the
                current target position.
        """
        y, p, d, t = self.sim.get_debug_visualizer()[-4:]
        if distance is None:
            distance = d
        if yaw is None:
            yaw = y
        if pitch is None:
            pitch = p
        if target_position is None:
            target_position = t
        self.sim.reset_debug_visualizer(distance, yaw, pitch, target_position)

    def get_matrices(self, inverse=False):
        """
        Return the view, projection, and viewport matrices.

        Args:
            inverse (bool): if True, it will also compute the inverse of the view, projection, and viewport matrices.

        Returns:
            if inverse:
                np.array[float[4,4]]: view matrix
                np.array[float[4,4]]: projection matrix
                np.array[float[4,4]]: viewport matrix
                np.array[float[4,4]]: inverse of the view matrix
                np.array[float[4,4]]: inverse of the projection matrix
                np.array[float[4,4]]: inverse o the viewport matrix
            else:
                np.array[float[4,4]]: view matrix
                np.array[float[4,4]]: projection matrix
                np.array[float[4,4]]: viewport matrix
        """
        width, height, V, P = self.sim.get_debug_visualizer()[:4]
        Vp = np.array([[width / 2., 0, 0, width / 2.],
                       [0, height / 2., 0, height / 2.],
                       [0, 0, 0.5, 0.5],
                       [0, 0, 0, 1]])
        if inverse:
            Vinv = np.linalg.inv(V)
            Pinv = np.linalg.inv(P)
            Vpinv = np.linalg.inv(Vp)
            return V, P, Vp, Vinv, Pinv, Vpinv
        return V, P, Vp

    def get_vectors(self):
        """
        Return the forward, up, and lateral vectors of the camera.

        Returns:
            np.array[float[3]]: forward vector
            np.array[float[3]]: up vector
            np.array[float[3]]: lateral vector (=cross product between forward and up vectors)
        """
        up_vector, forward_vector = self.sim.get_debug_visualizer()[4:6]
        lateral_vector = np.cross(forward_vector, up_vector)
        return forward_vector, up_vector, lateral_vector

    def compute_view_matrix_from_ypr(self, target_position, distance, yaw, pitch, roll, up_axis_index):
        """Compute the view matrix from the yaw, pitch, and roll angles.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            target_position (np.array[float[3]]): target focus point in Cartesian world coordinates
            distance (float): distance from eye to focus point
            yaw (float): yaw angle in radians left/right around up-axis
            pitch (float): pitch in radians up/down.
            roll (float): roll in radians around forward vector
            up_axis_index (int): either 1 for Y or 2 for Z axis up.

        Returns:
            np.array[float[4,4]]: the view matrix

        More info:
            [1] http://www.codinglabs.net/article_world_view_projection_matrix.aspx
            [2] http://www.thecodecrate.com/opengl-es/opengl-transformation-matrices/
        """
        return self.sim.compute_view_matrix_from_ypr(self, target_position=target_position, distance=distance, yaw=yaw,
                                                     pitch=pitch, roll=roll, up_axis_index=up_axis_index)

    def compute_view_matrix(self, eye_position, target_position, up_vector):
        """Compute the view matrix.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            eye_position (np.array[float[3]]): eye position in Cartesian world coordinates
            target_position (np.array[float[3]]): position of the target (focus) point in Cartesian world coordinates
            up_vector (np.array[float[3]]): up vector of the camera in Cartesian world coordinates

        Returns:
            np.array[float[4,4]]: the view matrix

        More info:
            [1] http://www.codinglabs.net/article_world_view_projection_matrix.aspx
            [2] http://www.thecodecrate.com/opengl-es/opengl-transformation-matrices/
        """
        return self.sim.compute_view_matrix(eye_position, target_position, up_vector)

    def set_yaw_pitch(self, yaw, pitch, radian=True):
        """
        Set the yaw and pitch angles.

        Args:
            yaw (float): yaw angle.
            pitch (float): pitch angle.
            radian (bool): If the given pitch and yaw angles are in radian.
        """
        if radian:
            yaw, pitch = np.rad2deg(yaw), np.rad2deg(pitch)
        distance, target_pos = self.sim.get_debug_visualizer()[-2:]
        self.sim.reset_debug_visualizer(distance, yaw, pitch, target_pos)

    def add_yaw_pitch(self, dyaw, dpitch, radian=True):
        """
        Add a small amount `dyaw` and `dpitch` to the camera's current yaw and pitch angles.

        Args:
            dyaw (float): small amount to add to the camera's current yaw angle
            dpitch (float): small amount to add to the camera's current pitch angle
            radian (bool): If the given pitch and yaw angles are in radian.
        """
        yaw, pitch, distance, target_pos = self.sim.get_debug_visualizer()[-4:]
        if radian:
            dyaw, dpitch = np.rad2deg(dyaw), np.rad2deg(dpitch)
        yaw += dyaw
        pitch += dpitch
        self.sim.reset_debug_visualizer(distance, yaw, pitch, target_pos)

    def get_rgb_image(self):
        """
        Return the captured RGB image.

        Returns:
            np.array[int[W,H,C]]: RGB image (width, height, RGB channels)
        """
        return self.get_rgba_image()[:, :, :3]

    def get_rgba_image(self):
        """
        Return the captured RGBA image. 'A' stands for alpha channel (for opacity/transparency)

        Returns:
            np.array[int[W,H,C]]: RGBA image (width, height, RGBA channels)
        """
        width, height, view_matrix, projection_matrix = self.sim.get_debug_visualizer()[:4]
        img = np.array(self.sim.get_camera_image(width, height, view_matrix, projection_matrix)[2])
        img = img.reshape(width, height, 4)  # RGBA
        return img

    def get_depth_image(self):
        """
        Return the depth image.

        Returns:
            np.array[float[W,H,C]]: depth image (width, height)
        """
        width, height, view_matrix, projection_matrix = self.sim.get_debug_visualizer()[:4]
        img = np.array(self.sim.get_camera_image(width, height, view_matrix, projection_matrix)[3])
        img = img.reshape(width, height)
        return img

    def get_rgbad_image(self, concatenate=True):
        """
        Return the RGBA and depth images.

        Args:
            concatenate (bool): If True, it will concatenate the RGBA and depth images such that it has a shape of
                (width, height, 5).

        Returns:
            if concatenate:
                np.array[int[W,H,C]]: RGBAD image (width, height, RGBAD channels)
            else:
                np.array[int[W,H,C]]: RGBA image (width, height, RGBA channels)
                np.array[float[W,H,C]]: depth image (width, height)
        """
        width, height, view_matrix, projection_matrix = self.sim.get_debug_visualizer()[:4]
        rgba, depth = self.sim.get_camera_image(width, height, view_matrix, projection_matrix)[2:4]
        rgba = np.array(rgba).reshape(width, height, 4)
        depth = np.array(depth).reshape(width, height)
        if concatenate:
            return np.dstack((rgba, depth))
        return rgba, depth

    def screen_to_world(self, x_screen, Vp_inv=None, P_inv=None, V_inv=None):
        """
        Return the corresponding coordinates in the Cartesian world space from the coordinates of a point
        on the screen.

        Args:
            x_screen (np.array[float[4]]): augmented vector coordinates of a point on the screen
            Vp_inv (np.array[float[4,4]], None): inverse of viewport matrix. If None, it will be computed.
            P_inv (np.array[float[4,4]], None): inverse of projection matrix. If None, it will be computed.
            V_inv (np.array[float[4,4]], None): inverse of view matrix. If None, it will be computed.

        Returns:
            np.array[float[4]]: augmented vector coordinates of the corresponding point in the world
        """
        if Vp_inv is None:
            Vp_inv = self.Vp_inv
        if P_inv is None:
            P_inv = self.Pinv
        if V_inv is None:
            V_inv = self.Vinv

        x_ndc = Vp_inv.dot(x_screen)
        x_ndc[1] = -x_ndc[1]  # invert y-axis
        x_ndc[2] = -x_ndc[2]  # invert z-axis
        x_eye = P_inv.dot(x_ndc)
        x_eye = x_eye / x_eye[3]  # normalize
        x_world = V_inv.dot(x_eye)
        return x_world

    def world_to_screen(self, x_world, V=None, P=None, Vp=None):
        """
        Return the corresponding screen coordinates from a 3D point in the world.

        Args:
            x_world (np.array[float[4]]): augmented vector coordinates of a point in the Cartesian world space
            V (np.array[float[4,4]], None): view matrix. If None, it will be computed.
            P (np.array[float[4,4]], None): projection matrix. If None, it will be computed.
            Vp (np.array[float[4,4]], None): viewport matrix. If None, it will be computed.

        Returns:
            np.array[float[4]]: augmented vector coordinates of the corresponding point on the screen
        """
        if V is None:
            V = self.V
        if P is None:
            P = self.P
        if Vp is None:
            Vp = self.Vp

        x_eye = V.dot(x_world)
        x_clip = P.dot(x_eye)
        x_ndc = x_clip / x_clip[3]  # normalize
        x_ndc[1] = -x_ndc[1]  # invert y-axis (as y pointing upward in projection but should point downward in screen)
        x_ndc[2] = -x_ndc[2]  # invert z-axis (to get right-handed coord. system, -1=close and 1=far)
        x_screen = Vp.dot(x_ndc)  # for depth between 0(=close) and 1(=far)
        return x_screen

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return the representation string of the object."""
        return self.__class__.__name__

    def print_info(self):
        """Print information about the camera.

        int: width of the visualizer camera (in pixel)
            int: height of the visualizer camera (in pixel)
            np.array[float[4,4]]: view matrix [4,4]
            np.array[float[4,4]]: perspective projection matrix [4,4]
            np.array[float[3]]: camera up vector expressed in the Cartesian world space
            np.array[float[3]]: forward axis of the camera expressed in the Cartesian world space
            np.array[float[3]]: This is a horizontal vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            np.array[float[3]]: This is a vertical vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            float: yaw angle (in radians) of the camera, in Cartesian local space coordinates
            float: pitch angle (in radians) of the camera, in Cartesian local space coordinates
            float: distance between the camera and the camera target
            np.array[float[3]]: target of the camera, in Cartesian world space coordinates
        """
        info = self.info
        view_inv = np.linalg.inv(info[2])
        position = view_inv[:3, 3]
        orientation = get_quaternion_from_matrix(view_inv[:3, :3])
        print("\nCamera width and height: {}, {}".format(info[0], info[1]))
        print("Camera position: {}".format(position))
        print("Camera orientation (quaternion [x,y,z,w]): {}".format(orientation))
        print("Camera target position: {}".format(info[11]))
        print("Camera yaw and pitch angles (deg): {}, {}".format(np.rad2deg(info[8]), np.rad2deg(info[9])))
        print("Camera distance: {}".format(info[10]))
        print("Camera forward vector: {}".format(info[5]))
        print("Camera up vector: {}".format(info[4]))

    def follow(self, body_id, distance=None, yaw=None, pitch=None):
        """
        Follow the given body in the simulator with the world camera at the specified distance, yaw and pitch angles.

        Args:
            body_id (int, long): body to follow with the world camera.
            distance (float, None): distance (in meter) from the camera and the body position. If None, it will take
                the current distance.
            yaw (float, None): camera yaw angle (in radians) left/right. If None, it will take the current yaw angle.
            pitch (float, None): camera pitch angle (in radians) up/down. If None, it will take the current pitch angle.
        """
        if not isinstance(body_id, (int, long)):
            raise TypeError("Expecting the given 'body_id' to be a unique id (int/long) returned by the simulator, "
                            "instead got: {}".format(type(body_id)))
        target_position = self.sim.get_base_position(body_id)
        self.reset(distance=distance, yaw=yaw, pitch=pitch, target_position=target_position)


# Tests
if __name__ == '__main__':
    from itertools import count
    import time
    from pyrobolearn.simulators import Bullet

    # create simulator
    sim = Bullet()

    # load floor
    floor_id = sim.load_urdf('plane.urdf', use_fixed_base=True)

    # create camera
    camera = WorldCamera(sim)

    # define variables
    radius = 2
    theta = 0
    dtheta = 0.01

    for t in count():
        # get camera information and print them
        position = camera.position
        yaw = camera.yaw
        pitch = camera.pitch
        distance = camera.distance
        target_position = camera.target_position
        print("Position: {}".format(position))
        print("Target position: {}".format(target_position))
        print("Yaw: {}".format(np.rad2deg(yaw)))
        print("Pitch: {}".format(np.rad2deg(pitch)))
        print("Distance: {}".format(distance))
        print("##########\n")

        # move camera
        theta += dtheta
        # camera.position = np.array([0, radius * np.sin(theta), radius * np.cos(theta)])
        camera.position = np.array([radius * np.sin(theta), -radius * np.cos(theta), 2])

        # step in the simulator
        sim.step()
        time.sleep(1./240)
