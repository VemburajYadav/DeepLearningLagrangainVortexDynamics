from functools import partial
from phi.tf.flow import *
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--offset', type=list, default=[60, 60], help='neglect regions near boundaries of the '
                                                                  'domain (as list: [24, 24])')
parser.add_argument('--n_samples', type=int, default=16000, help='number of samples to be generated')
parser.add_argument('--strength_range', type=list, default=[-2, 2], help='range for strength sampling')
parser.add_argument('--strength_threshold', type=float, default=1.0, help='minimum value of magnitude of strength')
parser.add_argument('--sigma_range', type=list, default=[5.0, 25.0], help='range for core ize sampling')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--num_time_steps', type=int, default=10, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--save_dir', type=str, default='/media/vemburaj/9d072277-d226-41f6-a38d-1db833dca2bd/'
                                                    'data/single_vortex_dataset_256x256_16000',
                    help='diretory to save the generated dataset')

opt = parser.parse_args()

RESOLUTION = opt.domain
OFFSET = opt.offset
SAMPLE_RES = [RESOLUTION[0] - 2 * OFFSET[0], RESOLUTION[1] - 2 * OFFSET[1]]
NSAMPLES = opt.n_samples
STRENGTH_RANGE = opt.strength_range
SIGMA_RANGE = opt.sigma_range
STRENGTH_THRESHOLD_MAG = opt.strength_threshold
TRAIN_PERCENT = opt.train_percent
VAL_PERCENT = opt.eval_percent

N_TRAIN_SAMPLES = int(NSAMPLES * TRAIN_PERCENT)
N_VAL_SAMPLES = int(NSAMPLES * VAL_PERCENT)
N_TEST_SAMPLES = NSAMPLES - (N_TRAIN_SAMPLES + N_VAL_SAMPLES)

NUM_TIME_STEPS = opt.num_time_steps
DIRECTORY = opt.save_dir


def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    return (math.exp(- sq_distance / sigma ** 2)) / math.sqrt(sq_distance)


ycoords = np.sort(np.random.random_sample(size=NSAMPLES) * SAMPLE_RES[0] + OFFSET[0])
xcoords = np.sort(np.random.random_sample(size=NSAMPLES) * SAMPLE_RES[1] + OFFSET[1])

strengths_pos = np.random.random_sample(size=NSAMPLES) * (STRENGTH_RANGE[1] - STRENGTH_THRESHOLD_MAG) + STRENGTH_THRESHOLD_MAG
strengths_neg = np.random.random_sample(size=NSAMPLES) * (-STRENGTH_THRESHOLD_MAG - STRENGTH_RANGE[0]) + STRENGTH_RANGE[0]

strengths = np.sort(np.concatenate([strengths_neg, strengths_pos]))
sigmas = np.sort(np.random.random_sample(size=NSAMPLES) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0])

np.random.shuffle(ycoords)
np.random.shuffle(xcoords)
np.random.shuffle(strengths)
np.random.shuffle(sigmas)

train_ycoords, train_xcoords = ycoords[0: N_TRAIN_SAMPLES], xcoords[0: N_TRAIN_SAMPLES]
train_strengths, train_sigmas = strengths[0:N_TRAIN_SAMPLES], sigmas[0: N_TRAIN_SAMPLES]

val_ycoords, val_xcoords = ycoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES)],\
                           xcoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES)]
val_strengths, val_sigmas = strengths[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES)], \
                            sigmas[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES)]

test_ycoords, test_xcoords = ycoords[-N_TEST_SAMPLES:], xcoords[-N_TEST_SAMPLES:]
test_strengths, test_sigmas = strengths[-N_TEST_SAMPLES:], sigmas[-N_TEST_SAMPLES:]


domain = Domain(RESOLUTION, boundaries=OPEN)
FLOW_REF = Fluid(domain)

location_pl = tf.placeholder(shape=(1, 1, 2), dtype=tf.float32)
strength_pl = tf.placeholder(shape=(1, 1), dtype=tf.float32)
sigma_pl = tf.placeholder(shape=(1, 1, 1), dtype=tf.float32)

vorticity = AngularVelocity(location=location_pl,
                            strength=strength_pl,
                            falloff=partial(gaussian_falloff, sigma=sigma_pl))

velocity_0 = vorticity.at(FLOW_REF.velocity)
velocities_tf = [velocity_0]

FLOW = Fluid(domain=domain, velocity=velocity_0)
fluid = world.add(Fluid(domain=domain, velocity=velocity_0), physics=IncompressibleFlow())

for step in range(NUM_TIME_STEPS):
    world.step()
    velocities_tf.append(fluid.velocity)

velocity_filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
sess = Session(None)

train_dir = os.path.join(DIRECTORY, 'train')

for id in range(N_TRAIN_SAMPLES):
    SCENE = Scene.create(train_dir)
    location = np.reshape(np.array([train_ycoords[id], train_xcoords[id]]), (1, 1, 2)).astype(np.float32)
    strength = np.reshape(train_strengths[id], (1, 1)).astype(np.float32)
    sigma = np.reshape(train_sigmas[id], (1, 1, 1)).astype(np.float32)
    velocities = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())

val_dir = os.path.join(DIRECTORY, 'val')

for id in range(N_VAL_SAMPLES):
    SCENE = Scene.create(val_dir)
    location = np.reshape(np.array([val_ycoords[id], val_xcoords[id]]), (1, 1, 2)).astype(np.float32)
    strength = np.reshape(val_strengths[id], (1, 1)).astype(np.float32)
    sigma = np.reshape(val_sigmas[id], (1, 1, 1)).astype(np.float32)
    velocities = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())

test_dir = os.path.join(DIRECTORY, 'test')

for id in range(N_TEST_SAMPLES):
    SCENE = Scene.create(test_dir)
    location = np.reshape(np.array([test_ycoords[id], test_xcoords[id]]), (1, 1, 2)).astype(np.float32)
    strength = np.reshape(test_strengths[id], (1, 1)).astype(np.float32)
    sigma = np.reshape(test_sigmas[id], (1, 1, 1)).astype(np.float32)
    velocities = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())
