Interfaces and Bridges
======================

**I/O interfaces** allows you to receive/send data from/to different devices. Interfaces are divided into 3 categories, ``InputInterface`` which can only receive data from a particular device and save it in memory, ``OutputInterface`` which can only send data given by PRL to the interface, and ``InputOutputInterface`` which allows you to receive and send data. Interfaces include for instance webcam, speaker, mouse, keyboard, game controller, and so on.

To avoid a direct coupling with the interface and an element in PRL such as a robot, **bridges** are introduced. Bridges makes the connection between an interface and an element in PRL (like a robot, an object in the world, the world itself, the world camera, etc). For instance, you could have a game controller and when moving one of its joystick forward, you would like for a quadcopter to take off while for a wheeled robot to move forward instead. For both examples, the values returned by the joystick is the same but you would like to have different behaviors depending on the type of robots. It might be even the case that someone would like to map the game controller events in a different way that you did. This is exactly the raison d'être of such bridges; to map an interface with an element in PRL. Different bridges can be implemented for the same interface as the user sees fit.

Available interfaces in PRL include:

- camera: webcam, asus_xtion, kinect, openpose
- controllers: mouse+keyboard, spacemouse, playstation, xbox
- speech: recognizer, translator, and synthesizer
- VR: Oculus (through Windows)

They are available in `pyrobolearn/tools/interfaces/ <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/tools/interfaces>`_ folder while bridges are available in the `pyrobolearn/tools/bridges/ <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/tools/bridges>`_ folder.


How to use an interface/bridge in PRL?
--------------------------------------

The following snippet show how to use the space mouse interface.




You can check for more examples in the `examples/interfaces <https://github.com/robotlearn/pyrobolearn/tree/master/examples/interfaces>`_ folder.



How to create your own interface/bridge?
----------------------------------------

Let's say that you have a new interface, for instance, an EMG sensor that measures the electrical activity of muscles, and you would like based on the sensed values makes a robot behave in a certain way. For instance, you would like the robot to be more stiff (see teleimpedance for more info).

- In order to create your interface, you will have to inherit one of the following interfaces: ``InputInterface``, ``OutputInterface``, ``InputOutputInterface`` based on the type of device you have. In our case, we have an EMG sensor which provides the sensed values as *inputs* to PRL, thus we will inherit from ``InputInterface``.

.. code-block:: python
    :linenos:

    # please add the word `Interface` at the end of your class so we can based on its name alone 
    # knows it is an interface.
    class EMGInterface(InputInterface):
    	"""
    	Description
    	"""

    	def __init__(self, use_thread=False, sleep_dt=0, verbose=False, *args, **kwargs):
    		# initialize your variables/attributes
    		...

    		# call at the end the parent constructor
    		super(EMGInterface, self).__init__(self, use_thread, sleep_dt, verbose)

    	def run(self):
    		"""main method to implement. This method is automatically called when using threads, and you 
    		have to call it when you are not using threads."""
    		# get the last sensed data and save it in one of the attributes of this class
    		...

- Now, let's create a bridge that connects the above interface with a manipulator robot.

.. code-block:: python
    :linenos:

    # please add the word `Interface` at the end of your class so we can based on its name alone 
    # knows it is an interface.
    class EMGBridge(InputInterface):


FAQs and Troubleshootings
-------------------------

- I have an ``ImportError`` with one of the interface, why? Some libraries have to be installed and configured manually. To ease the installation process, there is a docker file as well as bash scripts in the `pyrobolearn/scripts/ <https://github.com/robotlearn/pyrobolearn/tree/master/scripts>`_ folder.


Future works
------------

* add an interface to get the values sensed by an android/Iphone smartphone (which might have an accelerometer, gyroscope, microphone, etc.)
* add HTC Vive interface
* add a Facial Expression Recognition (FER) module
* add Google assistant / Alexa
* implement interfaces for haptic devices
* improve the VR interface; right now, I have something for Oculus but it requires to use a Windows system in parallel (see VIDEO)
* implement the Xsens suit interface; I also have a code for this but pretty old and it also requires a Windows system.
