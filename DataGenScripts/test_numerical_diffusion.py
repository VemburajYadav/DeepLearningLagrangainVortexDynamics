from phi.flow import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


domain_low = Domain(resolution=[100, 100], box=box[0:100, 0:100], boundaries=OPEN)
domain_high = Domain(resolution=[100, 100], box=box[0:100, 0:100], boundaries=OPEN)

location = np.array([50.0, 50.0], dtype=np.float32)
strength = np.array([2.0], dtype=np.float32)
sigma = np.array([20.0], dtype=np.float32)

def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    return (math.exp(- sq_distance / sigma ** 2)) / math.sqrt(sq_distance)

vorticity = AngularVelocity(location=np.reshape(location, (1,1,2)),
                            strength=np.reshape(strength, (1,1)),
                            falloff=partial(gaussian_falloff, sigma=np.reshape(sigma, (1,1,1))))

FLOW_low = Fluid(domain=domain_low)
FLOW_high = Fluid(domain=domain_high)

velocity_low = vorticity.at(FLOW_low.velocity)
velocity_high = vorticity.at(FLOW_high.velocity)

world_low = World()
world_high = World()

fluid_low = world_low.add(Fluid(domain=domain_low, velocity=velocity_low),
                          physics=[IncompressibleFlow(), lambda fluid, dt: fluid.copied_with(velocity=diffuse(fluid.velocity, 0.1 * dt))])
fluid_high = world_high.add(Fluid(domain=domain_high, velocity=velocity_high),
                            physics=[IncompressibleFlow(), lambda fluid, dt: fluid.copied_with(velocity=diffuse(fluid.velocity, 0.5 * dt))])

for i in range(1):
    world_low.step()
    world_high.step()

error_low_x = ((velocity_low.x.data - fluid_low.velocity.x.data)**2).sum() / (101 * 100)
error_low_y = ((velocity_low.y.data - fluid_low.velocity.y.data)**2).sum() / (101 * 100)

error_high_x = ((velocity_high.x.data - fluid_high.velocity.x.data)**2).sum() / (1001 * 1000)
error_high_y = ((velocity_high.y.data - fluid_high.velocity.y.data)**2).sum() / (1001 * 1000)

