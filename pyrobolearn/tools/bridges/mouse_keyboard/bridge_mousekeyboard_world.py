#!/usr/bin/env python
"""Define the Bridge between the mouse-keyboard interface and the world.

Dependencies:
- `pyrobolearn.utils`
- `pyrobolearn.worlds` or `pyrobolearn.envs`
- `pyrobolearn.tools.interfaces.MouseKeyboardInterface`
- `pyrobolearn.tools.bridges.Bridge`
"""

import numpy as np

from pyrobolearn.utils.bullet_utils import RGBColor, Key
from pyrobolearn.utils.math_utils import Plane

from pyrobolearn.worlds import World, WorldCamera
from pyrobolearn.envs import Env
from pyrobolearn.tools.interfaces import MouseKeyboardInterface
from pyrobolearn.tools.bridges import Bridge


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BridgeMouseKeyboardWorld(Bridge):
    r"""Bridge Mouse-Keyboard World

    Bridge between the mouse-keyboard and the world.

    Mouse:
    * predefined in pybullet
        * `scroll wheel`: zoom
        * `ctrl`/`alt` + `scroll button`: move the camera using the mouse
        * `ctrl`/`alt` + `left-click`: rotate the camera using the mouse
        * `left-click` and drag: transport the object
    * `left-click`: select/unselect an object (show bounding box, and print name/id in simulator)
    * `right-click` and drag: perform IK on the selected link until the mouse button is released
    * `ctrl` + `left-click`: select multiple object

    Keyboard:
    * predefined in pybullet:
        * `w`: show the wireframe (collision shapes)
        * `s`: show the reference system
        * `v`: show bounding boxes
        * `g`: show/hide parts of the GUI the side columns (check `sim.configure_debug_visualizer(p.COV_ENABLE_GUI, 0)`)
        * `esc`: quit the simulator
    * `x`: change camera view such that it is perpendicular to the x-axis
    * `y`: change camera view such that it is perpendicular to the y-axis
    * `z`: change camera view such that it is perpendicular to the z-axis
    * `j`: after selecting a robot, display the joint sliders to control them
    * `t`: after selecting a robot link, display the cartesian sliders to control it using IK
    * `b`: after selecting a link of the robot, display the bounding box around the selected link
    * `d`: display what the robot sees (rgbd image)
    * `u`: unselect any objects
    * `c`: clean and reset the given world
    * `a`: apply force/torque on the selected link/joint using sliders (select the link/joint using the mouse)
    * `h`: hide/show the complete GUI; enable/disable the rendering
    * `m`: show/hide the workspace of the corresponding link (not implemented yet)
    * `l`: change mode of locomotion (defined in the robot; select the controller/policy)
    * `<number>`: select and perform corresponding action (10 actions possible, as <number> between 0 and 9)
    * `top arrow`: move the selected robot forward
    * `bottom arrow`: move the selected robot backward
    * `left arrow`: turn the selected robot to the left
    * `right arrow`: turn the selected robot to the right
    * `space`: allow the robot to jump (if implemented)
    """

    def __init__(self, world, interface=None, priority=None, verbose=False):
        """
        Initialize the Bridge between a Mouse-Keyboard interface and the world.

        Args:
            world (World): world instance.
            interface (None, MouseKeyboardInterface): mouse keyboard interface. If None, it will create one.
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # set world
        self.world = world

        # check interface
        if not isinstance(interface, MouseKeyboardInterface):
            interface = MouseKeyboardInterface(self.simulator)

        # call superclass
        super(BridgeMouseKeyboardWorld, self).__init__(interface, priority)

        # camera
        self.world_camera = WorldCamera(self.simulator)
        self._camera = None
        self.default_camera = None
        self.plane, self.depth = None, None
        self.bounding_box = False

        self.robot, self.link_id = None, None
        self.joint_sliders, self.task_sliders = False, {}
        self.hiding_gui = False
        self.pausing = False
        self.debug = verbose

        self.events_fn = {(Key.x,): self.change_camera_view_x,
                          (Key.y,): self.change_camera_view_y,
                          (Key.z,): self.change_camera_view_z,
                          (Key.enter,): self.reset_camera_view,
                          (Key.space,): self.pause,  # pause the simulator
                          (Key.h,): self.gui,  # hide/enable (actually stop/play) GUI
                          (Key.r,): self.reset_world,  # reset world
                          (Key.j,): self.update_joint_sliders,  # add/remove joint (space) sliders
                          (Key.t,): self.update_task_sliders,  # add/remove task (space) sliders
                          (Key.p,): lambda: None,
                          (Key.u,): self.unselect,  # unselect robot/link
                          (Key.d,): lambda: None,  # display robot's camera
                          (Key.m,): lambda: None,  # display link workspace
                          (Key.b,): lambda: None,  # bounding box
                          (Key.a,): lambda: None,  # apply force/torque on selected link/joint
                          (Key.l,): lambda: None,
                          (Key.n0,): lambda: None,
                          (Key.n1,): lambda: None,
                          (Key.n2,): lambda: None,
                          (Key.n3,): lambda: None,
                          (Key.n4,): lambda: None,
                          (Key.n5,): lambda: None,
                          (Key.n6,): lambda: None,
                          (Key.n7,): lambda: None,
                          (Key.n8,): lambda: None,
                          (Key.n9,): lambda: None,
                          (Key.top_arrow,): lambda: None,
                          (Key.bottom_arrow,): lambda: None,
                          (Key.left_arrow,): lambda: None,
                          (Key.right_arrow,): lambda: None}

        # self.vs = self.simulator.createVisualShape(self.simulator.GEOM_SPHERE, radius=0.02, rgbaColor=(0, 0, 1, 1))
        # self.vs1 = self.simulator.createVisualShape(self.simulator.GEOM_SPHERE, radius=0.2, rgbaColor=(1, 0, 0, 1))
        # self.vs2 = self.simulator.createVisualShape(self.simulator.GEOM_SPHERE, radius=0.2, rgbaColor=(0, 1, 0, 1))

    ##############
    # Properties #
    ##############

    @property
    def world(self):
        """Return the world instance."""
        return self._world

    @world.setter
    def world(self, world):
        """Set the world instance."""
        if isinstance(world, Env):
            world = world.world
        if not isinstance(world, World):
            raise TypeError("Expecting the world to be an instance of World or Env, instead got {}".format(type(world)))
        self._world = world

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.world.simulator

    @property
    def camera(self):
        if self._camera is None:
            self._camera = self.world_camera.get_debug_visualizer_camera()
            if self.default_camera is None:
                self.default_camera = self._camera
        return self._camera

    ###########
    # Methods #
    ###########

    def print_debug(self, str1, str2='', condition=True):
        if self.debug:
            if condition:
                print("MouseKeyboardInterface: "+str1)
            else:
                print("MouseKeyboardInterface: "+str2)

    def step(self, update_interface=False):
        """Perform a step: map the mouse-keyboard interface to the world"""
        # update interface
        if update_interface:
            self.interface()

        # check keyboard events
        self.check_key_events()
            
        # check mouse events and map to
        self.check_mouse_events()

        # update joint sliders if present (position control)
        if self.joint_sliders:
            self.robot.update_joint_slider()

        # update task sliders if present (IK)
        if self.task_sliders:
            pass

        # reset camera (the scrolling event is not detected)
        self._camera = None

        # step in the world
        if not self.pausing:
            self.world.step()

    def change_camera_view_x(self):
        """Change camera view X."""
        self.print_debug('change camera view X')
        dist, target = self.camera[-2:]
        self.simulator.reset_debug_visualizer(distance=dist, yaw=np.deg2rad(90.), pitch=0., target_position=target)

    def change_camera_view_y(self):
        """Change camera view Y."""
        self.print_debug('change camera view Y')
        dist, target = self.camera[-2:]
        self.simulator.reset_debug_visualizer(distance=dist, yaw=np.deg2rad(180.), pitch=0., target_position=target)

    def change_camera_view_z(self):
        """Change camera view Z."""
        self.print_debug('change camera view Z')
        dist, target = self.camera[-2:]
        self.simulator.reset_debug_visualizer(distance=dist, yaw=-np.deg2rad(90.), pitch=-np.deg2rad(89.99),
                                              target_position=target)

    def reset_camera_view(self):
        """Reset camera view."""
        self.print_debug('reset camera view')
        if self.default_camera is not None:
            yaw, pitch, dist, target = self.default_camera[-4:]
            self.simulator.reset_debug_visualizer(distance=dist, yaw=yaw, pitch=pitch, target_position=target)
            self.default_camera = None

    def pause(self):
        """Pause the simulation."""
        self.pausing = not self.pausing
        self.print_debug('pause the simulator', 'unpause the simulator', self.pausing)

    def gui(self):
        """show/hide GUI."""
        self.hiding_gui = not self.hiding_gui
        self.simulator.configure_debug_visualizer(self.simulator.COV_ENABLE_GUI, self.hiding_gui)
        self.print_debug('hide the GUI', 'enable the GUI', self.hiding_gui)

    def reset_world(self):
        """Reset the world."""
        self.print_debug('reset the world')
        self.world.reset_robots()
        self.simulator.removeAllUserDebugItems()

    def update_joint_sliders(self):
        """Update joint sliders."""
        if self.robot is not None:
            if self.joint_sliders:  # remove joint sliders
                self.robot.remove_joint_slider()
                self.print_debug('remove joint sliders')
            else:  # add joint sliders
                self.robot.add_joint_slider()
                self.print_debug('add joint sliders')
            self.joint_sliders = not self.joint_sliders

    def update_task_sliders(self):
        """Update task sliders."""
        if self.robot is not None and self.link_id is not None:
            if self.link_id in self.task_sliders:  # remove task sliders
                for idx in self.task_sliders[self.link_id]:
                    self.simulator.remove_user_debug_item(self.task_sliders[self.link_id][idx])
                self.task_sliders.pop(self.link_id)
                self.print_debug('remove task sliders')
            else:  # add task sliders
                self.task_sliders[self.link_id] = {}
                pos = self.robot.get_link_world_positions(self.link_id)
                for i, name in zip(pos, ['x', 'y', 'z']):
                    slider = self.simulator.add_user_debug_parameter(name, i - 2., i + 2., i)
                    self.task_sliders[self.link_id][name] = slider
                self.print_debug('add task sliders')

    def unselect(self):
        """Unselect the robot and link id."""
        self.print_debug('unselect robot/link')
        self.robot, self.link_id = None, None

    def add_world_text(self, string, position, color=(0.,0.,0.), size=1., lifetime=0.):
        """Add world text."""
        self.print_debug('add world text')
        self.simulator.add_user_debug_text(string, position, color, size, lifetime)

    def add_screen_text(self, string, world_position, color=(0.,0.,0.), size=1., lifetime=0.):
        """Add screen text."""
        self.print_debug('add screen text')
        V, P, Vp, V_inv, P_inv, Vp_inv = self.world_camera.get_matrices(True)
        position = self.world_camera.screen_to_world(world_position, Vp_inv, P_inv, V_inv)[:3]
        self.simulator.add_user_debug_text(string, position, color, size, lifetime)

    def check_key_events(self):
        # call function corresponding to key combination
        if self.interface.key_pressed:
            key = tuple(self.interface.key_pressed)
            if key in self.events_fn:
                self.events_fn[key]()

    def check_mouse_events(self):
        # check if the mouse has been released
        if not self.interface.mouse_down:
            self.plane, self.depth = None, None

        # check what object we are trying to grab with the mouse by checking collision
        if self.interface.mouse_pressed:
            V, P, Vp, V_inv, P_inv, Vp_inv = self.world_camera.get_matrices(True)
            camera = self.world_camera.get_debug_visualizer_camera()

            # from the point (x,y) on the screen, get nearest and farthest point on the screen
            x_screen_init = np.array([self.interface.mouse_x, self.interface.mouse_y, 1., 1.])
            x_screen_final = np.array([self.interface.mouse_x, self.interface.mouse_y, 0., 1.])

            # get the corresponding points in the world
            x_world_init = self.world_camera.screen_to_world(x_screen_init, Vp_inv, P_inv, V_inv)
            x_world_final = self.world_camera.screen_to_world(x_screen_final, Vp_inv, P_inv, V_inv)

            # check if there is a collision
            # print(x_world_init[:3], x_world_final[:3])
            collision = self.simulator.ray_test(list(x_world_init[:3]), list(x_world_final[:3]))

            # if collision, proceed the inverse operation to get the depth on the screen
            if len(collision) > 0:
                object_id, link_id, hit_frac, hit_pos, hit_normal = collision[0]
                # self.simulator.addUserDebugLine(list(x_world_init[:3]), list(x_world_final[:3]), (0, 0, 1))
                # bodyId = self.simulator.createMultiBody(baseMass=0, baseVisualShapeIndex=self.vs1,
                #                                   basePosition=list(x_world_init[:3]))
                # bodyId = self.simulator.createMultiBody(baseMass=0, baseVisualShapeIndex=self.vs2,
                #                                   basePosition=list(x_world_final[:3]))
                if object_id != -1 and self.world.is_robot_id(object_id):  # valid object

                    # Set robot and link_id
                    self.robot, self.link_id = self.world.get_body(object_id), link_id

                    width, height = camera[:2]
                    x_screen = np.array([width/2, height/10, 0.95, 1])
                    pos = self.world_camera.screen_to_world(x_screen, Vp_inv, P_inv, V_inv)[:3]
                    # self.simulator.add_user_debug_text(str(self.robot) + ": " + self.robot.get_link_names(self.link_id),
                    #                                 pos, RGBColor.black, textSize=1)

                    # calculate plane
                    # 1. compute the initial point on the plane (collision point)
                    if link_id == -1:  # no link
                        x0 = np.array(self.simulator.get_base_pose(object_id)[0])
                    else:  # link
                        x0 = np.array(self.simulator.get_link_state(object_id, link_id)[0])

                    # 2. calculate normal (=target_position - eyePosition) to the plane
                    # normal = x_world_final[:3] - x_world_init[:3]
                    yaw, pitch, dist, target = camera[-4:]
                    yaw, pitch = np.deg2rad(yaw), np.deg2rad(pitch)
                    normal = dist * np.array([np.cos(pitch) * np.sin(yaw),
                                              -np.cos(pitch) * np.cos(yaw),
                                              -np.sin(pitch)])

                    # 3. create plane
                    self.plane = Plane(x0, normal)

                    # calculate associate depth on the screen (because perspective projection)
                    hit_pos = np.array(list(hit_pos) + [1.])
                    self.depth = self.world_camera.world_to_screen(hit_pos, V, P, Vp)[2]

        elif self.interface.mouse_down and self.interface.mouse_moving and self.robot is not None:
            V, P, Vp, V_inv, P_inv, Vp_inv = self.world_camera.get_matrices(True)
            camera = self.world_camera.get_debug_visualizer_camera()

            if self.plane is not None:
                # project the point on the screen to the world, and check where the line that starts from this point
                # and is perpendicular to the plane (i.e. parallel to the normal) intersects with the aforementioned
                # plane
                x_screen = np.array([self.interface.mouse_x, self.interface.mouse_y, self.depth, 1])
                x_world = self.world_camera.screen_to_world(x_screen, Vp_inv, P_inv, V_inv)[:3]
                point = self.plane.get_intersection_point(x_world)

            # # draw some spheres on the plane
            # if self.display_trajectories:
            #    bodyId = self.simulator.createMultiBody(baseMass=0, baseVisualShapeIndex=self.vs,
            #                                      basePosition=point)
            #    self.visual_points[tuple(point)] = bodyId
            # else:
            #    self.visual_points[tuple(point)] = None

            # # draw trajectories
            # if self.display_trajectories:
            #     self.visual_points.append(point)
            #     if len(self.visual_points) > 1:
            #         self.simulator.addUserDebugLine(self.visual_points[-2], self.visual_points[-1],
            #                                   RGBColor.red, 1., 2.)

            # # perform inverse kinematics
            # q = self.robot.calculate_inverse_kinematics(self.link, point)
            # for i in range(self.robot.getNumberOfJoints()):
            #     self.robot.set_joint_positions(i, q[i])


# Tests
if __name__ == '__main__':
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld
    import time
    from itertools import count

    # create simulator
    sim = BulletSim()

    # sim.configure_debug_visualizer(p.COV_ENABLE_GUI, 0)
    # sim.configure_debug_visualizer(p.COV_ENABLE_RENDERING, 0)
    # sim.configure_debug_visualizer(p.COV_ENABLE_TINY_RENDERER, 1)
    # sim.configure_debug_visualizer(p.COV_ENABLE_WIREFRAME, 1)
    # sim.configure_debug_visualizer(p.COV_ENABLE_Y_AXIS_UP, 0)
    # sim.configure_debug_visualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    # sim.configure_debug_visualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    # sim.configure_debug_visualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # create World
    world = BasicWorld(sim)

    # load robot
    robot = world.load_robot('baxter', fixed_base=True)

    # create bridge/interface
    bridge = BridgeMouseKeyboardWorld(world, verbose=True)

    for _ in count():
        bridge.step(update_interface=True)
        # world.step()
        time.sleep(1. / 100)
