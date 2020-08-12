from functools import partial
from phi.tf.flow import *
import matplotlib.pyplot as plt
import argparse
import random
import copy
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--offset', type=list, default=[24, 24], help='neglect regions near boundaries of the '
                                                                  'domain (as list: [24, 24])')
parser.add_argument('--nzones', type=list, default=[4, 4], help='number of zones in each dimension to sample the data '
                                                                '(as list: for eg. [4, 4])')
parser.add_argument('--samples_per_zone', type=int, default=500, help='number of samples to be generated per zone')
parser.add_argument('--strength_range', type=list, default=[-0.05, 0.05], help='range for strength sampling')
parser.add_argument('--sigma_range', type=list, default=[5.0, 55.0], help='range for core ize sampling')
parser.add_argument('--n_tau_sig_split', type=int, default=10, help='split the strength and sigma ranges into '
                                                                    'n_tau_sig_split splits')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--num_time_steps', type=int, default=2, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--save_dir', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_1',
                    help='diretory to save the generated dataset')

opt = parser.parse_args()

RESOLUTION = opt.domain
OFFSET = opt.offset
SAMPLE_RES = [RESOLUTION[0] - 2 * OFFSET[0], RESOLUTION[1] - 2 * OFFSET[1]]
SAMPLES_PER_ZONE = opt.samples_per_zone
STRENGTH_RANGE = opt.strength_range
SIGMA_RANGE = opt.sigma_range

NZONES = opt.nzones
N_SIG_TAU = opt.n_tau_sig_split

TRAIN_PERCENT = opt.train_percent
VAL_PERCENT = opt.eval_percent

NUM_TIME_STEPS = opt.num_time_steps
DIRECTORY = opt.save_dir

strength_div_size = (STRENGTH_RANGE[1] - STRENGTH_RANGE[0]) / N_SIG_TAU
sigma_div_size = (SIGMA_RANGE[1] - SIGMA_RANGE[0]) / N_SIG_TAU
domain_y_div_size = SAMPLE_RES[0] / NZONES[0]
domain_x_div_size = SAMPLE_RES[1] / NZONES[1]

y_split = np.append(np.arange(0, SAMPLE_RES[1], domain_y_div_size), SAMPLE_RES[1])
x_split = np.append(np.arange(0, SAMPLE_RES[0], domain_x_div_size), SAMPLE_RES[0])
tau_split = np.append(np.arange(STRENGTH_RANGE[0], STRENGTH_RANGE[1], strength_div_size), STRENGTH_RANGE[1])
sigma_split = np.append(np.arange(SIGMA_RANGE[0], SIGMA_RANGE[1], sigma_div_size), SIGMA_RANGE[1])

zone_y_range = [[y_split[i], y_split[i+1]] for i in range(len(y_split)-1)]
zone_x_range = [[x_split[i], x_split[i+1]] for i in range(len(x_split)-1)]

tau_range = [[tau_split[i], tau_split[i+1]] for i in range(len(tau_split)-1)]
sigma_range = [[sigma_split[i], sigma_split[i+1]] for i in range(len(sigma_split)-1)]


def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    return (math.exp(- sq_distance / sigma ** 2)) / math.sqrt(sq_distance)


def create_dataset_vortex(ycoords, xcoords, tau_range_list, sigma_range_list, save_dir):

    NSAMPLES = int(SAMPLES_PER_ZONE / N_SIG_TAU)

    domain = Domain(RESOLUTION, boundaries=OPEN)
    FLOW = Fluid(domain)

    GLOBAL_STEP = 0

    for i in range(len(tau_range_list)):
        tau_min, tau_max = tau_range_list[i][0], tau_range_list[i][1]
        sig_min, sig_max = sigma_range_list[i][0], sigma_range_list[i][1]

        tau = np.random.random_sample(size=NSAMPLES) * (tau_max - tau_min) + tau_min
        sigma = np.random.random_sample(size=NSAMPLES) * (sig_max - sig_min) + sig_min

        yc = ycoords[i*NSAMPLES: (i+1)*NSAMPLES]
        xc = xcoords[i*NSAMPLES: (i+1)*NSAMPLES]

        for sample in range(NSAMPLES):

            SCENE = Scene.create(save_dir)

            location = np.array([yc[sample], xc[sample]], dtype=np.float32).reshape((1,1,2))
            strength = np.array([tau[sample]], dtype=np.float32).reshape((1,1))
            stddev = np.array([sigma[sample]], dtype=np.float32).reshape((1,1,1))

            world_obj = World()
            vorticity = AngularVelocity(location=location,
                                        strength=strength,
                                        falloff=partial(gaussian_falloff, sigma=stddev))

            velocity_0 = vorticity.at(FLOW.velocity)

            fluid = world_obj.add(Fluid(domain=domain, velocity=velocity_0), physics=IncompressibleFlow())

            np.savez(os.path.join(SCENE.path, 'velocity_000000.npz'), velocity_0.sample_at(domain.center_points()))

            for step in range(NUM_TIME_STEPS):
                world_obj.step()
                filename = 'velocity_' + '0' * (6 - len(str(step+1))) + str(step+1) + '.npz'
                np.savez(os.path.join(SCENE.path, filename), fluid.velocity.sample_at(domain.center_points()))

            np.savez(os.path.join(SCENE.path, 'location_000000.npz'), location)
            np.savez(os.path.join(SCENE.path, 'strength_000000.npz'), strength)
            np.savez(os.path.join(SCENE.path, 'sigma_000000.npz'), stddev)

            print('Writing Simulation Case: {}'.format(SCENE.path))
            GLOBAL_STEP += 1


for zone_y_id in range(NZONES[0]):
    for zone_x_id in range(NZONES[1]):#

        print('Creating dataset for zone: ({}, {})'.format(zone_y_id, zone_x_id))
        ycoords = np.random.random_sample(size=SAMPLES_PER_ZONE) * domain_y_div_size + y_split[zone_y_id] + OFFSET[0]
        xcoords = np.random.random_sample(size=SAMPLES_PER_ZONE) * domain_x_div_size + x_split[zone_x_id] + OFFSET[1]

        train_y_coords = ycoords[:int(SAMPLES_PER_ZONE * TRAIN_PERCENT)]
        train_x_coords = xcoords[:int(SAMPLES_PER_ZONE * TRAIN_PERCENT)]

        eval_y_coords = ycoords[int(SAMPLES_PER_ZONE * TRAIN_PERCENT): int(SAMPLES_PER_ZONE * (TRAIN_PERCENT + VAL_PERCENT))]
        eval_x_coords = xcoords[int(SAMPLES_PER_ZONE * TRAIN_PERCENT): int(SAMPLES_PER_ZONE * (TRAIN_PERCENT + VAL_PERCENT))]

        test_y_coords = ycoords[int(SAMPLES_PER_ZONE * (TRAIN_PERCENT + VAL_PERCENT)):]
        test_x_coords = xcoords[int(SAMPLES_PER_ZONE * (TRAIN_PERCENT + VAL_PERCENT)):]

        tau_range_copy = copy.deepcopy(tau_range)
        sigma_range_copy = copy.deepcopy(sigma_range)

        random.shuffle(tau_range_copy)
        random.shuffle(sigma_range_copy)

        train_tau_range_list = tau_range_copy[:int(N_SIG_TAU * TRAIN_PERCENT)]
        train_sigma_range_list = sigma_range_copy[:int(N_SIG_TAU * TRAIN_PERCENT)]

        eval_tau_range_list = tau_range_copy[int(N_SIG_TAU * TRAIN_PERCENT): int(N_SIG_TAU * (TRAIN_PERCENT + VAL_PERCENT))]
        eval_sigma_range_list = sigma_range_copy[int(N_SIG_TAU * TRAIN_PERCENT): int(N_SIG_TAU * (TRAIN_PERCENT + VAL_PERCENT))]

        test_tau_range_list = tau_range_copy[int(N_SIG_TAU * (TRAIN_PERCENT + VAL_PERCENT)):]
        test_sigma_range_list = sigma_range_copy[int(N_SIG_TAU * (TRAIN_PERCENT + VAL_PERCENT)):]

        create_dataset_vortex(train_y_coords, train_x_coords,
                              train_tau_range_list, train_sigma_range_list,
                              os.path.join(DIRECTORY, 'train'))
        create_dataset_vortex(eval_y_coords, eval_x_coords,
                              eval_tau_range_list, eval_sigma_range_list,
                              os.path.join(DIRECTORY, 'eval'))
        create_dataset_vortex(test_y_coords, test_x_coords,
                              test_tau_range_list, test_sigma_range_list,
                              os.path.join(DIRECTORY, 'test'))

#
#





