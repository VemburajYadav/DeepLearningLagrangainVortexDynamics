"""
Script for generating dataset for vortex particle dynamics in an open domain and no viscosity.

For each data sample
1) Particle locations, core sizes and vortex strengths are randomly sampled at time t0.
2) Corresponding velocity field on grid is computed for t0.
3) Simulation using PhiFlow to obtain grid velocity fields at future time instants.
4) Save the location, strength, core size at t0 and the velocity fields at all time instants.

"""




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
parser.add_argument('--time_step', type=float, default=0.2, help='time step in seconds for running numerical simulations')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--num_time_steps', type=int, default=10, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--save_dir', type=str, default='../'
                                                    'data/p10_gaussian_dataset_120x120_4000',
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

# strengths, core sizes and locations of shape: (NSAMPLES, NPARTICLES)
strengths = np.reshape(strengths, (NSAMPLES, -1))
sigmas = np.reshape(sigmas, (NSAMPLES, -1))

ycoords = np.empty((NSAMPLES, NPARTICLES))
xcoords = np.empty((NSAMPLES, NPARTICLES))

# Randomly sample particle locations
for i in range(NSAMPLES):
    ycoords[i, :] = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[0] + OFFSET[0]
    xcoords[i, :] = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[1] + OFFSET[1]

# Train, Val, Test split
train_ycoords, train_xcoords = ycoords[0: N_TRAIN_SAMPLES, :], xcoords[0: N_TRAIN_SAMPLES, :]
train_strengths, train_sigmas = strengths[0:N_TRAIN_SAMPLES, :], sigmas[0: N_TRAIN_SAMPLES, :]

val_ycoords, val_xcoords = ycoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :],\
                           xcoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]
val_strengths, val_sigmas = strengths[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :], \
                            sigmas[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]

test_ycoords, test_xcoords = ycoords[-N_TEST_SAMPLES:, :], xcoords[-N_TEST_SAMPLES:, :]
test_strengths, test_sigmas = strengths[-N_TEST_SAMPLES:, :], sigmas[-N_TEST_SAMPLES:, :]


## Build a tf graph for executing simulations

# define domain and resolution of the grid
domain = Domain(RESOLUTION, boundaries=OPEN)
FLOW_REF = Fluid(domain)

# Placeholders for location, strength and core size
location_pl = tf.placeholder(shape=(1, NPARTICLES, 2), dtype=tf.float32)
strength_pl = tf.placeholder(shape=(NPARTICLES, ), dtype=tf.float32)
sigma_pl = tf.placeholder(shape=(1, NPARTICLES, 1), dtype=tf.float32)

# vorticity field
vorticity = AngularVelocity(location=location_pl,
                            strength=strength_pl,
                            falloff=partial(gaussian_falloff, sigma=sigma_pl))

# velocity field computed on grid points
velocity_0 = vorticity.at(FLOW_REF.velocity)

velocities_tf = [velocity_0]

# define the physics object for PhiFlow simulations
FLOW = Fluid(domain=domain, velocity=velocity_0)
fluid = world.add(Fluid(domain=domain, velocity=velocity_0), physics=IncompressibleFlow())

# time advancement
for step in range(NUM_TIME_STEPS):
    world.step(dt=TIME_STEP)
    velocities_tf.append(fluid.velocity)

# filename's for saving velocity fields
velocity_filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]


## Execute the constructed tf graph
sess = Session(None)

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
    velocities = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())


# Generate and save the validation set
val_dir = os.path.join(DIRECTORY, 'val')

for id in range(N_VAL_SAMPLES):
    SCENE = Scene.create(val_dir)
    location = np.reshape(np.stack([val_ycoords[id], val_xcoords[id]], axis=1), (1,NPARTICLES,2)).astype(np.float32)
    strength = np.reshape(val_strengths[id], (NPARTICLES, )).astype(np.float32)
    sigma = np.reshape(val_sigmas[id], (1, NPARTICLES, 1)).astype(np.float32)
    velocities = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())


# Generate and save the test set
test_dir = os.path.join(DIRECTORY, 'test')

for id in range(N_TEST_SAMPLES):
    SCENE = Scene.create(test_dir)
    location = np.reshape(np.stack([test_ycoords[id], test_xcoords[id]], axis=1), (1,NPARTICLES,2)).astype(np.float32)
    strength = np.reshape(test_strengths[id], (NPARTICLES,)).astype(np.float32)
    sigma = np.reshape(test_sigmas[id], (1, NPARTICLES, 1)).astype(np.float32)
    velocities = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())
#