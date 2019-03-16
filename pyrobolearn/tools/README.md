## Tools

This folder the *interfaces* and the *bridges*.

* The I/O *interfaces* get (or set) the data from (to) the hardware, process it, and store it inside the class. 
* The *bridges* makes the connection between the interface and a component (such as the world or an element in that world such as a robot) in the framework.

The separation between interfaces and bridges allows for better flexibility. For instance, a game controller interface allows us to get data from the hardware, process it, and store it inside the class. The bridge can then map the specific controller events to a robot. Moving a joystick up could mean to move a UAV robot up in the air, or move a wheeled robot forward. 