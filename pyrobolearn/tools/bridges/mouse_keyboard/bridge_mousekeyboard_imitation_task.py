#!/usr/bin/env python
"""Define the Bridge between the mouse-keyboard interface and the world.

Dependencies:
- `pyrobolearn.utils`
- `pyrobolearn.worlds` or `pyrobolearn.envs`
- `pyrobolearn.tools.interfaces.MouseKeyboardInterface`
- `pyrobolearn.tools.bridges.Bridge`
"""

from pyrobolearn.utils.bullet_utils import RGBColor, Key

from pyrobolearn.simulators import Simulator
from pyrobolearn.worlds import World
from pyrobolearn.envs import Env

from pyrobolearn.tools.interfaces import MouseKeyboardInterface
# from pyrobolearn.tools.bridges import Bridge
from pyrobolearn.tools.bridges.mouse_keyboard.bridge_mousekeyboard_world import BridgeMouseKeyboardWorld
# from pyrobolearn.tasks.imitation_task import ILTask  # Warning: circular dependency

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BridgeMouseKeyboardImitationTask(BridgeMouseKeyboardWorld):  # Bridge):
    r"""Bridge Mouse-Keyboard Imitation Task

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
        * `g`: show/hide parts of the GUI the side columns (check `sim.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)`)
        * `esc`: quit the simulator
    * `x`: change camera view such that it is perpendicular to the x-axis
    * `y`: change camera view such that it is perpendicular to the y-axis
    * `z`: change camera view such that it is perpendicular to the z-axis
    * `j`: after selecting a robot, display the joint sliders to control them
    * `t`: after selecting a robot link, display the cartesian sliders to control it using IK
    * `b`: after selecting a link of the robot, display the bounding box around the selected link
    * `d`: display what the robot sees (rgbd image)
    * `u`: unselect any objects
    * `r`: start/stop recording using the given recorder (if multiple recorders, select in the parameter column).
           The recorders are the ones who know what/how to record and how to deal with the data. For instance,
           if a policy needs to be trained, you have to interact with the recorder and not the interface.
    * `f`: save recorded trajectory to file (append)
    * `c`: clean and reset the given world
    * `a`: apply force/torque on the selected link/joint using sliders (select the link/joint using the mouse)
    * `h`: hide/show the complete GUI; enable/disable the rendering
    * `m`: show/hide the workspace of the corresponding link (not implemented yet)
    * `p`: show/hide recorded trajectories
        * `shift`+`p`: remove recorded trajectories
    * `o`: optimized the given policy to the demonstrated trajectories (if multiple policy, select one policy)
    * `e`: test the optimized policy associated to the selected robot (if multiple policy, select one policy)
    * `l`: change mode of locomotion (defined in the robot; select the controller/policy)
    * `<number>`: select and perform corresponding action (10 actions possible, as <number> between 0 and 9)
    * `top arrow`: move the selected robot forward
    * `bottom arrow`: move the selected robot backward
    * `left arrow`: turn the selected robot to the left
    * `right arrow`: turn the selected robot to the right
    * `space`: allow the robot to jump (if implemented)
    """

    def __init__(self, world, interface=None, imitation_task=None, priority=None, verbose=False):
        """
        Initialize the Bridge between a Mouse-Keyboard interface and an imitation learning task.

        Args:
            interface (MouseKeyboardInterface, Env, World): mouse keyboard interface.
                If the interface is an instance of Env, World, or Simulator, it will create automatically
                a mouse-keyboard interface.
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # check interface
        if not isinstance(interface, MouseKeyboardInterface):
            if interface is None:
                sim = world.simulator
            elif isinstance(interface, Env):
                sim = interface.simulator
            elif isinstance(interface, World):
                sim = interface.simulator
            elif isinstance(interface, Simulator):
                sim = interface
            else:
                raise TypeError("Expecting interface to be an instance of MouseKeyboardInterface, or World, "
                                "got instead {}".format(type(interface)))
            interface = MouseKeyboardInterface(sim)

        # call superclass
        super(BridgeMouseKeyboardImitationTask, self).__init__(world, interface, priority, verbose=verbose)

        # set task
        self.task = imitation_task

        # define few variables
        self.pausing = False
        self.recording = False
        self.training = False
        self.testing = False
        self.visual_points = []  # {}
        # self.display_trajectories = True

        self.events_fn = {(Key.x,): self.change_camera_view_x,
                          (Key.y,): self.change_camera_view_y,
                          (Key.z,): self.change_camera_view_z,
                          (Key.enter,): self.reset_camera_view,
                          (Key.space,): self.pause,  # pause the simulator
                          (Key.h,): self.gui,  # hide/enable (actually stop/play) GUI
                          (Key.r,): self.reset_world,  # reset world
                          (Key.j,): self.update_joint_sliders,  # add/remove joint (space) sliders
                          (Key.t,): self.update_task_sliders,  # add/remove task (space) sliders
                          (Key.p,): self.display_trajectories,  # display/hide trajectories
                          (Key.ctrl, Key.p): self.reset_visual_points,  # remove trajectories
                          (Key.u,): self.unselect,  # unselect robot/link
                          (Key.ctrl, Key.r): self.record,  # record
                          (Key.ctrl, Key.n): self.add_data_row_in_recorder,  # add a new data row in the recorder's
                                                                             # data "matrix"
                          (Key.shift, Key.r): self.end_recording,  # end recording
                          (Key.ctrl, Key.s): self.save_recording,  # save what has been recorded into a file
                          (Key.ctrl, Key.o): self.train,  # optimize/train the policy
                          (Key.shift, Key.o): self.end_training,  # end training
                          (Key.ctrl, Key.t): self.test,  # test the policy
                          (Key.shift, Key.t): self.end_testing,  # end testing
                          (Key.shift, Key.space): self.end_task,  # end task
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

        # define few variables
        self.enable_training = False
        self.enable_recording = False
        self.enable_testing = False

        # self.vs = sim.createVisualShape(sim.GEOM_SPHERE, radius=0.02, rgbaColor=(0, 0, 1, 1))
        # self.vs1 = sim.createVisualShape(sim.GEOM_SPHERE, radius=0.2, rgbaColor=(1, 0, 0, 1))
        # self.vs2 = sim.createVisualShape(sim.GEOM_SPHERE, radius=0.2, rgbaColor=(0, 1, 0, 1))

    ##############
    # Properties #
    ##############

    @property
    def pausing(self):
        """Return True if we need to pause the simulator."""
        return self._pausing

    @pausing.setter
    def pausing(self, boolean):
        """Specify if we should pause the simulator."""
        self._pausing = boolean

    @property
    def recording(self):
        """Return True if we are in recording mode."""
        return self._recording

    @recording.setter
    def recording(self, boolean):
        """Set recording mode."""
        self._recording = boolean

        # if you are recording, you can not train or test
        if self._recording:
            self._training = False
            self._testing = False

        # notify the task
        if self.task is not None:
            self.task.recording_enabled = boolean

    @property
    def training(self):
        """Return True if we are in training mode."""
        return self._training

    @training.setter
    def training(self, boolean):
        """Set training mode."""
        self._training = boolean

        # if you are training, you can not record or test
        if self._training:
            self._recording = False
            self._testing = False

        # notify the task
        if self.task is not None:
            self.task.training_enabled = boolean

    @property
    def testing(self):
        """Return True if we are in testing mode."""
        return self._testing

    @testing.setter
    def testing(self, boolean):
        """Set testing mode."""
        self._testing = boolean

        # if you are testing, you can not record or train
        if self._testing:
            self._recording = False
            self._training = False

        # notify the task
        if self.task is not None:
            self.task.testing_enabled = boolean

    @property
    def task(self):
        """Return the task instance."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the task."""
        from pyrobolearn.tasks.imitation import ILTask  # to avoid circular dependency
        if task is not None and not isinstance(task, ILTask):
            raise("Expecting the imitation task to be an instance of ILTask, instead got {}".format(type(task)))
        self._task = task

    @property
    def environment(self):
        """Return the environment instance."""
        return self.task.environment

    # @property
    # def world(self):
    #     """Return the world instance."""
    #     return self.task.world

    # @property
    # def simulator(self):
    #     """Return the simulator instance."""
    #     return self.task.simulator

    @property
    def recorders(self):
        """Return the recorder."""
        return self.task.recorders

    ###########
    # Methods #
    ###########

    def print_debug(self, str1, str2='', condition=True):
        if self.debug:
            if condition:
                print("MouseKeyboardInterface: "+str1)
            else:
                print("MouseKeyboardInterface: "+str2)

    # def step(self, update_interface=False):
    #     """Perform a step: map the mouse-keyboard interface to the imitation task"""
    #     # perform a step with the interface
    #     if update_interface:
    #         self.interface.step()
    #
    #     # # record (if specified)
    #     # if self.recording:
    #     #     for recorder in self.recorders:
    #     #         recorder.record()

    def pause(self):
        """pause the simulation"""
        self.pausing = not self.pausing
        self.print_debug('pause the simulator', 'unpause the simulator', self.pausing)

    def record(self):
        """Start/Stop recording"""
        # if self.recorders is not None:
        self.recording = not self.recording
        self.print_debug('start recording', 'stop recording', self.recording)

    def add_data_row_in_recorder(self):
        """Add a new data row in the recorder's data 'matrix'."""
        if self.task is not None:
            self.task.add_data_row_in_recorder()

    def save_recording(self):
        """save recording into file"""
        if self.task is not None:
            self.print_debug('saving what has been recorded')
            self.task.save_recorders()

    def reset_recorder(self):
        """Reset the recorders."""
        if self.task is not None:
            self.print_debug('reset recorders')
            self.task.reset_recorders()

    def end_recording(self):
        """End recording."""
        if self.task is not None:
            self.task.end_recording = True

    def train(self):
        """train the policies from the task"""
        self.print_debug('train the task')
        self.training = not self.training

    def end_training(self):
        """End training."""
        if self.task is not None:
            self.task.end_training = True

    def test(self):
        """test the policies from the task"""
        self.print_debug('test the task')
        self.testing = not self.testing

    def end_testing(self):
        """End testing."""
        if self.task is not None:
            self.task.end_testing = True

    def end_task(self):
        """End task."""
        if self.task is not None:
            self.task.end_task = True

    def display_trajectories(self):
        # if self.display_trajectories: # hide them (pybullet doesn't allow that for now, just remove body)
        #    for key in self.visual_points:
        #        self.sim.removeBody(self.visual_points[key])
        # else:
        #    for key in self.visual_points:
        #        bodyId = self.sim.createMultiBody(baseMass=0, baseVisualShapeIndex=self.vs,
        #                                          basePosition=key)
        #        self.visual_points[key] = bodyId
        # self.display_trajectories = not self.display_trajectories
        for i in range(len(self.visual_points[:-1])):
            self.simulator.addUserDebugLine(self.visual_points[i], self.visual_points[i + 1], RGBColor.red, 1., 2.)

    def reset_visual_points(self):
        # for key in self.visual_points:
        #    self.sim.removeBody(self.visual_points[key])
        # self.visual_points.clear()
        self.visual_points = []
