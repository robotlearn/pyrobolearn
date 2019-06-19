#!/usr/bin/env python
"""Define the OculusTouch class which communicates with Unity (on Windows) using TCP.

Currently, Oculus has only support for Windows systems. However, several libraries such as ROS only runs on
Linux systems. Thus, we can run Unity on a Windows system, associate the Unity scripts with the Oculus GameObjects,
and then run this file on a Unix system (such as Linux or MacOSX) which will communicate by TCP.
The scripts for Unity can be found in the `oculus_windows/unity-scripts/` folder.

Currently, this code is the server while the code running in Unity on Windows is the client.
"""

# TODO: refactor the code, and use the openvr library

import Queue
import socket
import struct
from threading import Thread

import cv2

from pyrobolearn.utils.bullet_utils import RGBAColor
from pyrobolearn.tools.interfaces.vr import VRInterface
# from pyrobolearn.worlds.world import BasicWorld

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class OculusInterface(VRInterface):
    r"""Oculus VR Interface

    Hardware: Oculus Rift headset + Touch controllers
    """

    def __init__(self, world, ip=True, port=5111, use_controllers=True, use_headset=False, use_threading=False,
                 rate=None):
        """Initialize the Oculus interface.

        Args:
            world (World, Env): the world
            use_controllers (bool): True if you want to use the controllers (stream commands from the controllers to
                the simulator, and send feedback
            use_headset (bool): True if you want to use the headsets (stream the pictures from the simulator to the
                headset, and the headset position/orientation to the simulator)
            ip (str): IP of the system to stream and receive the data. This is useful if the VR hardware is not
                supported on the current OS. For instance, if you are running on Linux and the VR hardware only runs
                on Windows, then by specifying the Windows system IP, the data will be streamed between these 2 systems.
                Note that when streaming between 2 systems, they have to agree about the data they are receiving and
                sending. Thus, the (code of the) interface on the Windows system is completely dependent on this
                interface. Use the corresponding code available in the `vr` folder, and follow the set-up instructions.
                Be sure that you use the correct corresponding interface on the other system, which will depend on the
                two previous arguments `use_controllers` and `use_headset`.
                Note that we currently use `Unity` as the interface between the VR hardware and the Windows system, as
                much of the code is already provided. A global overview of the set-up can be depicted as:
                pyrobolearn (Linux/Mac) <--TCP/UDP--> Unity (Windows) <----> VR hardware
            port (int): Port of the system to stream the data
            use_threading (bool): True if you want to use threads to send the pictures
            rate (int): acquisition rate of the camera images
        """
        super(OculusInterface, self).__init__()
        self.MSG_SIZE = 0

        self.use_headset = use_headset  # TODO: if we use the headset, we automatically use threads
        self.use_threading = use_threading

        if use_controllers:
            self.MSG_SIZE = 304

        # Simulator world
        self.world = world
        self.sim = self.world.simulator
        self.task = None

        # create visual spheres in the world for the hands
        self.world_camera = self.world.get_main_camera()
        V, P, Vp, V_inv, P_inv, Vp_inv = self.world_camera.get_matrices(True)
        camera = self.world_camera.get_debug_visualizer_camera(convert=False)
        width, height = camera[:2]
        left_pos = np.array([width / 2 - 20, height / 2, 0.95, 1])
        right_pos = np.array([width / 2 + 20, height / 2, 0.95, 1])
        left_pos = self.world_camera.screen_to_world(left_pos, Vp_inv, P_inv, V_inv)[:3]
        right_pos = self.world_camera.screen_to_world(right_pos, Vp_inv, P_inv, V_inv)[:3]
        self.leftSphere = self.world.loadVisualSphere(left_pos, radius=0.1, color=RGBAColor.red)  # red
        self.rightSphere = self.world.loadVisualSphere(right_pos, radius=0.1, color=RGBAColor.blue)  # blue

        # Check IP: get IP of this computer if not provided
        if ip:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]

        # Threads
        self.running = True
        self.queue = Queue.Queue(10)
        self.threads = []
        if self.use_headset:
            thread = Thread(target=self.run_thread, args=(ip, port + 1))
            self.threads.append(thread)

        # Connection over the network for joysticks
        print('Creating socket')
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_STREAM)  # TCP = SOCK_STREAM
        self.server_address = (ip, port)
        self.sock.bind(self.server_address)
        self.sock.listen(1)  # Only for TCP
        print('Waiting for connection...')
        self.connection, self.client_address = self.sock.accept()

        # VR
        self.head, self.leftHand, self.rightHand = [], [], []
        self.left_joystick, self.right_joystick = [], []
        self.left_vibration, self.right_vibration = 0, 0
        self.vibration_time = 1
        self.prev_oculus_head_pos = None

        # Camera images
        if rate is None:
            rate = np.inf
        self.cnt, self.rate = 0, rate
        self.width, self.height = 400, 400
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        self.left_collided = False
        self.right_collided = False

    def recv(self):
        data = self.connection.recvfrom(self.MSG_SIZE)
        poses = data[0].split(";")
        for pose in poses:
            name, value = pose.split("=")
            values = np.array(value.split(","))
            values = np.asfarray(values, float)
            # TODO: replace the if-else by dict of functions
            if name == "H": # head: position + quaternion
                self.head = [values[:3], values[3:]]
            elif name == "L": # left hand: position + quaternion
                self.leftHand = [values[:3], values[3:]]
            elif name == "R": # right hand: position + quaternion
                self.rightHand = [values[:3], values[3:]]
            elif name == 'JL': # left joystick [touch, button, lateral, forward]
                self.left_joystick = [values[0], values[1], values[-2:]]
            elif name == 'JR': # right joystick: [touch, button, lateral, forward]
                self.right_joystick = [values[0], values[1], values[-2:]]
                # move the camera by rotating
                yaw, pitch = values[-2:]
                # self.worldCamera.add_yaw_pitch(yaw, pitch, radian=False)
                # print(pitch, yaw)
                pos = self.world_camera.target_position
                dist = self.world_camera.distance
            elif name == 'BA': # button A: [touch, button]
                pass
            elif name == 'BB': # button B: [touch, button]
                pass
            elif name == 'BX': # button X: [touch, button]
                pass
            elif name == 'BY': # button Y: [touch, button]
                pass
            elif name == 'BS': # button start: button
                pass
            elif name == 'BLT': # button left index trigger: [button, trigger]
                pass
            elif name == 'BLH': # button left hand trigger: trigger
                pass
            elif name == 'BRT': # button right index trigger: [button, trigger]
                pass
            elif name == 'BRH': # button right hand trigger: trigger
                pass

        # update world #

        head_world_pos = self.world_camera.position
        target_pos = self.world_camera.target_position
        forward_vec, up_vec, lateral_vec = self.world_camera.get_vectors()
        if self.prev_oculus_head_pos is None:
            self.prev_oculus_head_pos = self.head[0]

        # # update world camera position and orientation based on Oculus headset and joysticks
        # R = np.array(self.sim.getMatrixFromQuaternion(self.head[1])).reshape(3,3)
        # target_pos = R.dot(target_pos)
        # target_pos += (self.head[0] - self.prevOculusHeadPos)
        lateral, forward = self.left_joystick[-1]  # move the camera by translating
        target_pos += 0.1 * (forward * forward_vec + lateral * lateral_vec)
        self.world_camera.target_position = target_pos

        # update hand positions in world
        left_hand_world_pos = head_world_pos + (self.leftHand[0] - self.head[0])
        right_hand_world_pos = head_world_pos + (self.rightHand[0] - self.head[0])
        # self.world.move_object(self.leftSphere, self.leftHand[0], (0, 0, 0, 1))
        # self.world.move_object(self.rightSphere, self.rightHand[0], (0, 0, 0, 1))
        self.world.move_object(self.leftSphere, left_hand_world_pos, (0, 0, 0, 1))
        self.world.move_object(self.rightSphere, right_hand_world_pos, (0, 0, 0, 1))

        # change color if hands collide with an object
        self.left_collided = self.update_sphere_color(self.leftSphere, self.left_collided,
                                                      RGBAColor.orange, RGBAColor.red)
        self.right_collided = self.update_sphere_color(self.rightSphere, self.right_collided,
                                                       RGBAColor.green, RGBAColor.blue)

        # get pictures for the eyes
        if self.use_headset and (self.cnt % self.rate) == 0:
            self.cnt = 0
            # get picture for the eyes
            left_pic = self.get_eye_rgb_image(head_world_pos, target_pos, lateral_vec, beta=-0.32)
            right_pic = self.get_eye_rgb_image(head_world_pos, target_pos, lateral_vec, beta=0.32)
            # if self.use_threading:
            # add them to the queue
            self.queue.put((left_pic, right_pic))
            # else:
            #    # compress pictures and send them over the network
            #    self.compress_and_send_picture(left_pic, self.connection)
            #    self.compress_and_send_picture(right_pic, self.connection)

        self.cnt += 1

    def update_sphere_color(self, sphere, has_collided_previously, collision_color, free_color):
        """Update sphere color."""
        aabb = self.world.get_object_aabb(sphere)

        if len(self.world.get_object_ids_in_aabb(aabb[0], aabb[1])) > 1:
            update = not has_collided_previously
        else:
            update = has_collided_previously

        if update:
            has_collided_previously = not has_collided_previously
            if has_collided_previously:
                self.world.change_object_color(sphere, color=collision_color)
            else:
                self.world.change_object_color(sphere, color=free_color)

        return has_collided_previously

    def run_thread(self, ip, port):
        """Run thread."""
        # create socket for image
        sock = socket.socket(socket.AF_INET,  # Internet
                             socket.SOCK_STREAM)  # UDP = SOCK_DGRAM / TCP = SOCK_STREAM
        server_address = (ip, port)  # ip and port
        sock.bind(self.server_address)
        sock.listen(1)  # Only for TCP
        print('Thread: waiting for connection...')
        connection, client_address = sock.accept()
        print('Thread: connected')

        # run thread
        while self.running:
            # get images added in the queue
            images = self.queue.get(block=True)
            # compress pictures and send them over the network
            for image in images:
                self.compress_and_send_picture(image, connection)
            # time.sleep(1./60)

    def get_eye_rgb_image(self, head_world_pos, target_pos, lateral_vec, beta=0.32):
        eye_pos = head_world_pos + beta * lateral_vec
        eye_target_pos = target_pos + beta * lateral_vec
        V = self.sim.computeViewMatrix(cameraEyePosition=eye_pos, cameraTargetPosition=eye_target_pos,
                                       cameraUpVector=(0,0,1))
        pic = np.array(self.sim.get_camera_image(self.width, self.height, viewMatrix=V)[2])
        pic = pic.reshape(self.width, self.height, 4)[:, :, :3]
        return pic

    def compress_and_send_picture(self, image, connection):
        return_value, image = cv2.imencode('.jpg', image, self.encode_params)
        image = image.tostring()
        connection.sendall(struct.pack('<i', len(image)))
        connection.sendall(image)

    def send(self):
        msg = "VL=" + format(self.left_vibration, '03') + "," + format(self.vibration_time, '03') + \
              ";VR=" + format(self.right_vibration, '03') + "," + format(self.vibration_time, '03')
        self.connection.sendall(msg)

    def step(self):
        self.recv()
        self.send()

    # alias
    update = step

    def print_state(self):
        print("Head: {}".format(self.head))
        print("Left hand: {}".format(self.leftHand))
        print("Right hand: {}".format(self.rightHand))

    def set_vibration(self, left=0, right=0, vibration_time=1):
        """
        Set the level of vibration on the corresponding oculus touch for the specified number of iterations/time.
        The level of vibration of each controller is between 0 and 255.
        """
        self.left_vibration, self.right_vibration = min(int(left), 255), min(int(right), 255)
        self.vibration_time = min(int(vibration_time), 200)

    def __del__(self):
        # stop threads
        self.running = False
        for t in self.threads:
            t.join()
        # stop connection
        self.connection.close()
        self.sock.close()


# Test
if __name__ == "__main__":
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld
    import time
    import numpy as np
    from itertools import count

    # create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create interface
    interface = OculusInterface(world, port=5111)

    # run simulation
    for t in count():
        interface.step()
        # interface.print_state()
        # step in the simulation
        world.step()
        time.sleep(1./60)
