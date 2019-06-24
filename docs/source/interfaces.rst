Interfaces and Bridges
======================

I/O interfaces.
Bridges.

Available interfaces:
* camera: asus_xtion, webcam, kinect, openpose
* controllers: mouse+keyboard, spacemouse, playstation, xbox
* speech


How to use an interface/bridge in PRL?
--------------------------------------


You can check for more examples in the [`examples/interfaces`](https://github.com/robotlearn/pyrobolearn/tree/master/examples/interfaces) folder.



How to create your own interface/bridge?
----------------------------------------

Let's say that you have a new interface, for instance, a EMG sensor.


FAQs and Troubleshootings
-------------------------

* I have an `ImportError` with one of the interface, why? Some libraries have to be installed and configured manually. To ease the installation process, there is a docker file as well as bash scripts in the `pyrobolearn/scripts/` folder.


Future works
------------

* add a Facial Expression Recognition (FER) module
* add Google assistant / Alexa
* implement interfaces for haptic devices
* improve the VR interface; right now, I have something for Oculus but it requires to use a Windows system in parallel (see VIDEO)
* implement the Xsens suit interface; I also have a code for this but pretty old and it also requires a Windows system.
