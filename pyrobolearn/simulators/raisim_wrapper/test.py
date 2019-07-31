import numpy as np
import raisimpy as raisim

world = raisim.World()
print("gravity: ", world.get_gravity())
print("set gravity to: np.array([0.,1.,2.])")
world.set_gravity(np.array([0.,1.,2.]))
print("gravity: ", world.get_gravity())
print("set gravity to: np.array([0.,-1.,-2.]).reshape(-1,1)")
world.set_gravity(np.array([0.,-1.,-2.]).reshape(-1,1))
print("gravity: ", world.get_gravity())
print("set gravity to: np.array([0.,1.,2.]).reshape(1,-1)")
world.set_gravity(np.array([0.,1.,2.]).reshape(1,-1))
print("gravity: ", world.get_gravity())
print("set gravity to: range(3,6)")
world.set_gravity(range(3,6))
print("gravity: ", world.get_gravity())

print("set gravity to: range(3,7)")
world.set_gravity(range(3,7))

