## Physics randomizer

This folder provides physics randomizers which randomizes the dynamical attributes / properties of an object.
For instance, it can randomize the mass or inertial matrix of a link, the bounciness of an object, the gravity of 
the world, the contact friction coefficients of the floor and various links, and so on.

Note that physics randomizer instances have access to the simulator in order to modify the physical properties.
Also, note that normally the physics randomizer is called at the beginning of an episode, and not at each time
step. Changing the physical properties at each time step can lead to weird behaviors in the simulator.
