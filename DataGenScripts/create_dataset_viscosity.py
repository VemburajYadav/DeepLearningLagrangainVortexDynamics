from functools import partial
from phi.tf.flow import *
import argparse
import matplotlib.pyplot as plt
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--offset', type=list, default=[40, 40], help='neglect regions near boundaries of the '
                                                                  'domain (as list: [24, 24])')
parser.add_argument('--n_samples', type=int, default=4000, help='number of samples to be generated')
parser.add_argument('--n_particles', type=int, default=10, help='number of vortex particles')
parser.add_argument('--sigma_range', type=list, default=[2.0, 10.0], help='range for core ize sampling')
parser.add_argument('--viscosity_range', type=list, default=[0.0, 3.0], help='range for core ize sampling')
parser.add_argument('--time_step', type=float, default=0.2, help='time step in seconds for running numerical simulations')#
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--num_time_steps', type=int, default=10, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--save_dir', type=str, default='../'
                                                    'data/p10_gaussian_dataset_viscous_120x120_4000',
                    help='diretory to save the generated dataset')



# Parse input arguments
opt = parser.parse_args()

RESOLUTION = opt.domain
OFFSET = opt.offset
SAMPLE_RES = [RESOLUTION[0] - 2 * OFFSET[0], RESOLUTION[1] - 2 * OFFSET[1]]
NSAMPLES = opt.n_samples
NPARTICLES = opt.n_particles
SIGMA_RANGE = opt.sigma_range
TIME_STEP = opt.time_step
VISCOSITY_RANGE = opt.viscosity_range
TRAIN_PERCENT = opt.train_percent
VAL_PERCENT = opt.eval_percent


N_TRAIN_SAMPLES = int(NSAMPLES * TRAIN_PERCENT)
N_VAL_SAMPLES = int(NSAMPLES * VAL_PERCENT)
N_TEST_SAMPLES = NSAMPLES - (N_TRAIN_SAMPLES + N_VAL_SAMPLES)

NUM_TIME_STEPS = opt.num_time_steps
DIRECTORY = opt.save_dir


# Gaussian falloff kernel
def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    falloff = (1.0 - math.exp(- sq_distance / sigma ** 2)) / (2.0 * np.pi * sq_distance)

    return falloff

# Sample core size
sigmas = np.reshape(np.random.random_sample(size=(NPARTICLES * NSAMPLES)) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0], (1, -1, 1))

# Sample multiplying factors to compute the strengths
facs = np.random.random_sample(size=(NPARTICLES * NSAMPLES)) * 15 + 5

np.random.shuffle(sigmas)
np.random.shuffle(facs)

# Randomly make half of the sampled strengths as negative
rands = np.array([-1] * (NSAMPLES * NPARTICLES // 2) + [1] * (NSAMPLES * NPARTICLES // 2))
np.random.shuffle(rands)

strengths = facs * sigmas.reshape((-1)) * rands
strengths = np.reshape(strengths, (-1,))
np.random.shuffle(strengths)

# Randomly sample kinematic viscosities
viscositys = np.sort(np.random.random_sample(size=(NSAMPLES)) * (VISCOSITY_RANGE[1] - VISCOSITY_RANGE[0]) + VISCOSITY_RANGE[0])
np.random.shuffle(viscositys)

# strengths, core sizes and locations of shape: (NSAMPLES, NPARTICLES) and viscosities of shape (NSAMPLES)
strengths = np.reshape(strengths, (NSAMPLES, -1))
sigmas = np.reshape(sigmas, (NSAMPLES, -1))
viscositys = np.reshape(viscositys, (NSAMPLES, -1))

# Randomly sample particle locations
ycoords = np.empty((NSAMPLES, NPARTICLES))
xcoords = np.empty((NSAMPLES, NPARTICLES))

for i in range(NSAMPLES):
    ycoords[i, :] = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[0] + OFFSET[0]
    xcoords[i, :] = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[1] + OFFSET[1]


# Train, Val, Test split
train_ycoords, train_xcoords = ycoords[0: N_TRAIN_SAMPLES, :], xcoords[0: N_TRAIN_SAMPLES, :]
train_strengths, train_sigmas = strengths[0:N_TRAIN_SAMPLES, :], sigmas[0: N_TRAIN_SAMPLES, :]
train_viscositites = viscositys[0:N_TRAIN_SAMPLES, :]

val_ycoords, val_xcoords = ycoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :],\
                           xcoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]
val_strengths, val_sigmas = strengths[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :], \
                            sigmas[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]
val_viscositites = viscositys[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]

test_ycoords, test_xcoords = ycoords[-N_TEST_SAMPLES:, :], xcoords[-N_TEST_SAMPLES:, :]
test_strengths, test_sigmas = strengths[-N_TEST_SAMPLES:, :], sigmas[-N_TEST_SAMPLES:, :]
test_viscosities = viscositys[-N_TEST_SAMPLES:, :]


# filename's for saving velocity fields
velocity_filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]


# Generate and save the training set
train_dir = os.path.join(DIRECTORY, 'train')

if not os.path.isdir(DIRECTORY):
    os.makedirs(DIRECTORY)

with open(os.path.join(DIRECTORY, 'dataset_config'), 'w') as configfile:
    json.dump(vars(opt), configfile, indent=2)

for id in range(N_TRAIN_SAMPLES):
    SCENE = Scene.create(train_dir)
    location = np.reshape(np.stack([train_ycoords[id], train_xcoords[id]], axis=1), (1,NPARTICLES,2)).astype(np.float32)
    strength = np.reshape(train_strengths[id], (NPARTICLES, )).astype(np.float32)
    sigma = np.reshape(train_sigmas[id], (1, NPARTICLES, 1)).astype(np.float32)
    nyu = np.reshape(train_viscositites[id], ()).astype(np.float32)
    domain = Domain(RESOLUTION, boundaries=OPEN)
    FLOW_REF = Fluid(domain)

    vorticity = AngularVelocity(location=location,
                                strength=strength,
                                falloff=partial(gaussian_falloff, sigma=sigma))
    velocity_0 = vorticity.at(FLOW_REF.velocity)

    world_obj = World()

    fluid = world_obj.add(Fluid(domain=domain, velocity=velocity_0),
                          physics=[IncompressibleFlow(), lambda fluid_1, dt: fluid_1.copied_with(velocity=diffuse(fluid_1.velocity,
                                                                                                                  nyu * dt, substeps=5))])

    velocities = [velocity_0]
    for step in range(NUM_TIME_STEPS):
        world_obj.step(dt=TIME_STEP)
        velocities.append(fluid.velocity)

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)
    np.savez_compressed(os.path.join(SCENE.path, 'viscosity.npz'), nyu)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())


# Generate and save the validation set
val_dir = os.path.join(DIRECTORY, 'val')

for id in range(N_VAL_SAMPLES):
    SCENE = Scene.create(val_dir)
    location = np.reshape(np.stack([val_ycoords[id], val_xcoords[id]], axis=1), (1,NPARTICLES,2)).astype(np.float32)
    strength = np.reshape(val_strengths[id], (NPARTICLES, )).astype(np.float32)
    sigma = np.reshape(val_sigmas[id], (1, NPARTICLES, 1)).astype(np.float32)
    nyu = np.reshape(val_viscositites[id], ()).astype(np.float32)

    domain = Domain(RESOLUTION, boundaries=OPEN)
    FLOW_REF = Fluid(domain)

    vorticity = AngularVelocity(location=location,
                                strength=strength,
                                falloff=partial(gaussian_falloff, sigma=sigma))
    velocity_0 = vorticity.at(FLOW_REF.velocity)

    world_obj = World()

    fluid = world_obj.add(Fluid(domain=domain, velocity=velocity_0),
                          physics=[IncompressibleFlow(), lambda fluid_1, dt: fluid_1.copied_with(velocity=diffuse(fluid_1.velocity,
                                                                                                                  nyu * dt, substeps=5))])

    velocities = [velocity_0]
    for step in range(NUM_TIME_STEPS):
        world_obj.step(dt=TIME_STEP)
        velocities.append(fluid.velocity)

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)
    np.savez_compressed(os.path.join(SCENE.path, 'viscosity.npz'), nyu)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())


# Generate and save the TEST set
test_dir = os.path.join(DIRECTORY, 'test')

for id in range(N_TEST_SAMPLES):
    SCENE = Scene.create(test_dir)
    location = np.reshape(np.stack([test_ycoords[id], test_xcoords[id]], axis=1), (1,NPARTICLES,2)).astype(np.float32)
    strength = np.reshape(test_strengths[id], (NPARTICLES,)).astype(np.float32)
    sigma = np.reshape(test_sigmas[id], (1, NPARTICLES, 1)).astype(np.float32)
    nyu = np.reshape(test_viscosities[id], ()).astype(np.float32)

    domain = Domain(RESOLUTION, boundaries=OPEN)
    FLOW_REF = Fluid(domain)

    vorticity = AngularVelocity(location=location,
                                strength=strength,
                                falloff=partial(gaussian_falloff, sigma=sigma))
    velocity_0 = vorticity.at(FLOW_REF.velocity)

    world_obj = World()

    fluid = world_obj.add(Fluid(domain=domain, velocity=velocity_0),
                          physics=[IncompressibleFlow(), lambda fluid_1, dt: fluid_1.copied_with(velocity=diffuse(fluid_1.velocity,
                                                                                                                  nyu * dt, substeps=5))])

    velocities = [velocity_0]
    for step in range(NUM_TIME_STEPS):
        world_obj.step(dt=TIME_STEP)
        velocities.append(fluid.velocity)

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)
    np.savez_compressed(os.path.join(SCENE.path, 'viscosity.npz'), nyu)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())
