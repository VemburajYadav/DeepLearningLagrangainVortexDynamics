from functools import partial
from phi.tf.flow import *
import argparse
import matplotlib.pyplot as plt
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--offset', type=list, default=[80, 80], help='neglect regions near boundaries of the '
                                                                  'domain (as list: [24, 24])')
parser.add_argument('--n_samples', type=int, default=32000, help='number of samples to be generated')
parser.add_argument('--n_particles', type=int, default=10, help='number of vortex particles')
parser.add_argument('--strength_range', type=list, default=[-2, 2], help='range for strength sampling')
parser.add_argument('--dist_range', type=list, default=[5.0, 60.0], help='distance between particles')
parser.add_argument('--strength_threshold', type=float, default=1.0, help='minimum value of magnitude of strength')
parser.add_argument('--sigma_range', type=list, default=[5.0, 25.0], help='range for core ize sampling')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--num_time_steps', type=int, default=2, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--save_dir', type=str, default='/home/vemburaj/'
                                                    'data/p10_r_dataset_256x256_32000',
                    help='diretory to save the generated dataset')

opt = parser.parse_args()

RESOLUTION = opt.domain
OFFSET = opt.offset
RADII_RANGE = opt.dist_range
SAMPLE_RES = [RESOLUTION[0] - 2 * OFFSET[0], RESOLUTION[1] - 2 * OFFSET[1]]
NSAMPLES = opt.n_samples
NPARTICLES = opt.n_particles
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


strengths_pos = np.random.random_sample(size=(NSAMPLES * NPARTICLES // 2)) * (STRENGTH_RANGE[1] - STRENGTH_THRESHOLD_MAG) + STRENGTH_THRESHOLD_MAG
strengths_neg = np.random.random_sample(size=(NSAMPLES * NPARTICLES // 2)) * (-STRENGTH_THRESHOLD_MAG - STRENGTH_RANGE[0]) + STRENGTH_RANGE[0]

strengths = np.sort(np.concatenate([strengths_neg, strengths_pos]))
sigmas = np.sort(np.random.random_sample(size=(NSAMPLES * NPARTICLES)) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0])
#
np.random.shuffle(strengths)
np.random.shuffle(sigmas)
#
strengths = np.reshape(strengths, (NSAMPLES, -1))
sigmas = np.reshape(sigmas, (NSAMPLES, -1))

NSAMPLES_1 = int(NSAMPLES * 0.75)
NSAMPLES_2 = NSAMPLES - NSAMPLES_1

radii_list = []
mid_dist = (RADII_RANGE[1] + RADII_RANGE[0]) / 2
radii_list.append(np.sort(np.random.random_sample(size=(NSAMPLES_1 * (NPARTICLES - 1))) * (mid_dist - RADII_RANGE[0]) + RADII_RANGE[0]))
radii_list.append(np.sort(np.random.random_sample(size=(NSAMPLES_2 * (NPARTICLES - 1))) * (RADII_RANGE[1] - mid_dist) + mid_dist))

radiis = np.sort(np.hstack(radii_list))
angles = np.sort(np.random.random_sample(size=(NSAMPLES * (NPARTICLES - 1))) * 360.0)

ycoords_ = np.sort(np.random.random_sample(size=(NSAMPLES)) * SAMPLE_RES[0] + OFFSET[0])
xcoords_ = np.sort(np.random.random_sample(size=(NSAMPLES)) * SAMPLE_RES[1] + OFFSET[1])

np.random.shuffle(xcoords_)
np.random.shuffle(ycoords_)
np.random.shuffle(radiis)
np.random.shuffle(angles)

radiis = np.reshape(radiis, (NSAMPLES, NPARTICLES - 1))
angles = np.reshape(angles, (NSAMPLES, NPARTICLES - 1))

delta_y = radiis * np.sin(angles * np.pi / 180.0)
delta_x = radiis * np.cos(angles * np.pi / 180.0)
ycoords_p = np.reshape(ycoords_, (-1, 1)) + delta_y
xcoords_p = np.reshape(xcoords_, (-1, 1)) + delta_x

for i in range(NPARTICLES - 1):
    out_of_box_y0 = np.where(ycoords_p[:, i] < OFFSET[0])[0]
    out_of_box_y1 = np.where(ycoords_p[:, i] > (RESOLUTION[0] - OFFSET[0]))[0]
    out_of_box_x0 = np.where(xcoords_p[:, i] < OFFSET[1])[0]
    out_of_box_x1 = np.where(xcoords_p[:, i] > (RESOLUTION[1] - OFFSET[1]))[0]

    ycoords_p[out_of_box_y0, i] = ycoords_p[out_of_box_y0, i] - 2.0 * delta_y[out_of_box_y0, i]
    ycoords_p[out_of_box_y1, i] = ycoords_p[out_of_box_y1, i] - 2.0 * delta_y[out_of_box_y1, i]
    xcoords_p[out_of_box_x0, i] = xcoords_p[out_of_box_x0, i] - 2.0 * delta_x[out_of_box_x0, i]
    xcoords_p[out_of_box_x1, i] = xcoords_p[out_of_box_x1, i] - 2.0 * delta_x[out_of_box_x1, i]

# delta_y1 = ycoords_ - ycoords_p[:, 0]
# delta_x1 = xcoords_ - xcoords_p[:, 0]
# dist1 = (delta_y1**2 + delta_x1**2)**0.5
#
# plt.figure()
# plt.hist(dist1, bins=50)
# plt.show()
# delta_y2 = ycoords_p[:,1] - ycoords_p[:, 0]
# delta_x2 = xcoords_p[:,1] - xcoords_p[:, 0]
# dist2 = (delta_y2**2 + delta_x2**2)**0.5
#
ycoords = np.hstack([np.reshape(ycoords_, (-1, 1)), ycoords_p])
xcoords = np.hstack([np.reshape(xcoords_, (-1, 1)), xcoords_p])

# plt.figure()
# plt.xlim([0, RESOLUTION[1]])
# plt.ylim([1, RESOLUTION[0]])
# plt.scatter(xcoords[0], ycoords[0])
# plt.show()


train_ycoords, train_xcoords = ycoords[0: N_TRAIN_SAMPLES, :], xcoords[0: N_TRAIN_SAMPLES, :]
train_strengths, train_sigmas = strengths[0:N_TRAIN_SAMPLES, :], sigmas[0: N_TRAIN_SAMPLES, :]

val_ycoords, val_xcoords = ycoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :],\
                           xcoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]
val_strengths, val_sigmas = strengths[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :], \
                            sigmas[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]

test_ycoords, test_xcoords = ycoords[-N_TEST_SAMPLES:, :], xcoords[-N_TEST_SAMPLES:, :]
test_strengths, test_sigmas = strengths[-N_TEST_SAMPLES:, :], sigmas[-N_TEST_SAMPLES:, :]


domain = Domain(RESOLUTION, boundaries=OPEN)
FLOW_REF = Fluid(domain)

location_pl = tf.placeholder(shape=(1, NPARTICLES, 2), dtype=tf.float32)
strength_pl = tf.placeholder(shape=(NPARTICLES, ), dtype=tf.float32)
sigma_pl = tf.placeholder(shape=(1, NPARTICLES, 1), dtype=tf.float32)

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

# max_x = np.abs(velocities[0].x.data[0, :, :, 0]).max()
# min_x = -max_x
#
# max_y = np.abs(velocities[0].y.data[0, :, :, 0]).max()
# min_y = -max_y
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(velocities[0].x.data[0, :, :, 0], cmap='RdYlBu', vmin=min_x, vmax=max_x)
# plt.subplot(1, 2, 2)
# plt.imshow(velocities[1].x.data[0, :, :, 0], cmap='RdYlBu', vmin=min_x, vmax=max_x)
# plt.show()
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(velocities[0].y.data[0, :, :, 0], cmap='RdYlBu', vmin=min_y, vmax=max_y)
# plt.subplot(1, 2, 2)
# plt.imshow(velocities[1].y.data[0, :, :, 0], cmap='RdYlBu', vmin=min_y, vmax=max_y)
# plt.show()
    np.savez_compressed(os.path.join(SCENE.path, 'location_000000.npz'), location)
    np.savez_compressed(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
    np.savez_compressed(os.path.join(SCENE.path, 'sigma_000000.npz'), sigma)

    for frame in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(SCENE.path, velocity_filenames[frame]), velocities[frame].staggered_tensor())

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
