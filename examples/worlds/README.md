## Worlds

This folder contains examples on how to create a world in the simulator, how to load different shapes in it (such 
as spheres, cubes, and others) with collisions or not, load various objects that are specified in URDFs/SDFs (such 
as robots), load a terrain from a heightmap, generate a terrain, and other functionalities.

The world is usually created once the simulator has been selected.

Here are the examples that the user can try:
1. `load_world.py`: load a basic world (i.e. with a floor and gravity enabled) with different objects (only visual, 
and with collisions) that are movable, fixed, or are moving.
2. `load_robot.py`: load a robot in a basic world and distribute randomly few objects on the floor.
3. `load_heightmap.py`: load a terrain from a heightmap (png) and load a robot on it.
4. `generate_terrain.py`: generate a terrain and distribute randomly few objects on the terrain.
