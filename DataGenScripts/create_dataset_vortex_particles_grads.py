"""
Script for generating dataset with vortex particles and the resulting velocities and
higher order derivatives of velocity at sampled points from the domain and boundary.

For each data sample
1) Particle locations, core sizes and vortex strengths are randomly sampled.
2) Velocity and derivatives of different orders are computed for:
    a) all the grid points in the staggered grid.
    b) randomly sampled non-grid points from the domain.
    c) randomly sampled points from the boundary.
3) Save the location, strength, core size as 'vortex_features.npz'.
4) Save the computed features as:
    a) 'features_points_y.npz' and 'features_points_x.npz' for the staggered grid points corresponding and x and y velocity.
    b) 'features_domain.npz' for non-grid points.
    c) 'features_boundaries.npz' for points in the boundaries.

"""




import torch
from phi.flow import *
import argparse
import os
import matplotlib.pyplot as plt
from core.velocity_derivs import *

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--offset', type=list, default=[20, 20], help='neglect regions near boundaries of the '
                                                                  'domain (as list: [24, 24])')
parser.add_argument('--n_samples', type=int, default=8000, help='number of samples to be generated')
parser.add_argument('--n_particles', type=int, default=10, help='number of vortex particles')
parser.add_argument('--sigma_range', type=list, default=[2.0, 10.0], help='range for core ize sampling')
parser.add_argument('--order', type=int, default=2, help='order of derivatives of velocities to compute')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--save_dir', type=str, default='../'
                                                    'data/bc_net_dataset_p10_gaussian_visc_8000',
                    help='diretory to save the generated dataset')




# Parse input arguments
opt = parser.parse_args()

RESOLUTION = opt.domain
OFFSET = opt.offset
SAMPLE_RES = [RESOLUTION[0] - 2 * OFFSET[0], RESOLUTION[1] - 2 * OFFSET[1]]
NSAMPLES = opt.n_samples
NPARTICLES = opt.n_particles
SIGMA_RANGE = opt.sigma_range
TRAIN_PERCENT = opt.train_percent
VAL_PERCENT = opt.eval_percent

N_TRAIN_SAMPLES = int(NSAMPLES * TRAIN_PERCENT)
N_VAL_SAMPLES = int(NSAMPLES * VAL_PERCENT)
N_TEST_SAMPLES = NSAMPLES - (N_TRAIN_SAMPLES + N_VAL_SAMPLES)

ORDER = opt.order
DIRECTORY = opt.save_dir



# Sample core size#
sigmas = np.reshape(np.random.random_sample(size=(NPARTICLES * NSAMPLES)) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0], (1, -1, 1))

# Sample multiplying factors to compute the strengths
facs = np.random.random_sample(size=(NPARTICLES * NSAMPLES)) * 15 + 5

np.random.shuffle(sigmas)
np.random.shuffle(facs)

# Randomly make half of the sampled strengths as negative
rands = np.array([-1] * (NSAMPLES * NPARTICLES // 2) + [1] * (NSAMPLES * NPARTICLES // 2))
np.random.shuffle(rands)

# strengths, core sizes and locations of shape: (NSAMPLES, NPARTICLES)
strengths = facs * sigmas.reshape((-1)) * rands
strengths = np.reshape(strengths, (-1,))

np.random.shuffle(strengths)

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


# define domain and resolution of the grid
domain = Domain(RESOLUTION, boundaries=CLOSED)
FLOW_REF = Fluid(domain)


# grid points in a staggered grid
points_y = torch.tensor(FLOW_REF.velocity.y.points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.x.points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, RESOLUTION[0] + 1, 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, RESOLUTION[1] + 1, 1), dtype=torch.float32, device='cuda:0')

# Module to compute the velocities and derivatives of velocities due to vorte particles
VelDerivExpRed = VelocityDerivatives(order=ORDER).to('cuda:0')


# Create directories for training, validation and test sets
if not os.path.isdir(DIRECTORY):
    os.makedirs(DIRECTORY)

train_dir = os.path.join(DIRECTORY, 'train')
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)

val_dir = os.path.join(DIRECTORY, 'val')
if not os.path.isdir(val_dir):
    os.makedirs(val_dir)

test_dir = os.path.join(DIRECTORY, 'test')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)

with open(os.path.join(DIRECTORY, 'dataset_config'), 'w') as configfile:
    json.dump(vars(opt), configfile, indent=2)


# Generate and save the training set
for i in range(N_TRAIN_SAMPLES):

    sample_dir = os.path.join(train_dir, 'sim_' + '0' * (6 - len(str(i))) + str(i))
    os.makedirs(sample_dir)

    # get the location, strength and core size for the data sample
    sample_yc = np.reshape(train_ycoords[i, :], (1, -1, 1))
    sample_xc = np.reshape(train_xcoords[i, :], (1, -1, 1))
    sample_tauc = np.reshape(train_strengths[i, :], (1, -1, 1))
    sample_sigc = np.reshape(train_sigmas[i, :], (1, -1, 1))

    feat_expred = np.concatenate([sample_yc, sample_xc, sample_tauc, sample_sigc], axis=-1)

    feat_expred_pt = torch.tensor(feat_expred, dtype=torch.float32, device='cuda:0')
    loc_index = torch.tensor([0, 1], device='cuda:0')

    points_y_res = points_y.view(1, -1, 2)
    points_x_res = points_x.view(1, -1, 2)

    # compute the velocity and its derivatives at the staggered grid points corresponding to y-component of velocity
    vel_y_expred = VelDerivExpRed(feat_expred_pt, points_y_res)
    # compute the velocity and its derivatives at the staggered grid points corresponding to x-component of velocity
    vel_x_expred = VelDerivExpRed(feat_expred_pt, points_x_res)


    p_locs = torch.index_select(feat_expred_pt, index=loc_index, dim=-1)
    p_locs_y, p_locs_x = torch.unbind(p_locs, dim=-1)

    # sample points randomly close to the location of particles (non-grid points)
    dom_points_y_near_p = (p_locs_y.view(1, -1, 1) + torch.randn((1, 9000),
                                                                dtype=torch.float32,
                                                                device='cuda:0').view(1, p_locs_y.shape[-1], -1)).view(1, -1)
    dom_points_x_near_p = (p_locs_x.view(1, -1, 1) + torch.randn((1, 9000),
                                                                dtype=torch.float32,
                                                                device='cuda:0').view(1, p_locs_x.shape[-1], -1)).view(1, -1)

    dom_points_near_p = torch.stack([dom_points_y_near_p, dom_points_x_near_p], dim=-1)

    # sample points randomly in the domain (non-grid points)
    dom_points_y_rand = torch.rand(1000, dtype=torch.float32, device='cuda:0') * RESOLUTION[0]
    dom_points_x_rand = torch.rand(1000, dtype=torch.float32, device='cuda:0') * RESOLUTION[1]
    dom_points_rand = torch.stack([dom_points_y_rand, dom_points_x_rand], dim=-1).view(1, -1, 2)
    dom_points = torch.cat([dom_points_near_p, dom_points_rand], dim=-2)

    # compute velocity and its derivatives for non-grid points
    vel_expred_dom = VelDerivExpRed(feat_expred_pt, dom_points)

    # randomly sample points in the vertical boundaries
    b1_points_y = torch.rand(2000, dtype=torch.float32, device='cuda:0') * RESOLUTION[0]
    b1_points_x = torch.cat([torch.tensor([0] * 1000, dtype=torch.float32, device='cuda:0'),
                               torch.tensor([RESOLUTION[1]] * 1000, dtype=torch.float32, device='cuda:0')], dim=-1)

    # randomly sample points in the horizontal boundaries
    b2_points_x = torch.rand(2000, dtype=torch.float32, device='cuda:0') * RESOLUTION[1]
    b2_points_y = torch.cat([torch.tensor([0] * 1000, dtype=torch.float32, device='cuda:0'),
                               torch.tensor([RESOLUTION[0]] * 1000, dtype=torch.float32, device='cuda:0')], dim=-1)

    b1_points = torch.stack([b1_points_y, b1_points_x], dim=-1).view(1, -1, 2)
    b2_points = torch.stack([b2_points_y, b2_points_x], dim=-1).view(1, -1, 2)

    # compute velocity and its derivatives at the points in the boundary
    vel_expred_b1 = VelDerivExpRed(feat_expred_pt, b1_points)
    vel_expred_b2 = VelDerivExpRed(feat_expred_pt, b2_points)

    # normal vectors for points in the boundary
    n1 = torch.tensor([0.0, 1.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(b1_points.shape[1], 1)
    n2 = torch.tensor([1.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(b2_points.shape[1], 1)

    n1 = n1.view(1, -1, 2)
    n2 = n2.view(1, -1, 2)

    # target velocity labels for the staggered grid points corresponding to y-component of velocity
    label_y = torch.tensor([1.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_y_expred.shape[1], 1)
    # target velocity labels for the staggered grid points corresponding to x-component of velocity
    label_x = torch.tensor([0.0, 1.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_x_expred.shape[1], 1)
    # target velocity labels for non-grid points
    label_dom = torch.tensor([0.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_expred_dom.shape[1], 1)

    label_y = label_y.view(1, -1, 2)
    label_x = label_x.view(1, -1, 2)
    label_dom = label_dom.view(1, -1, 2)

    # build a feature vector with location, velocity & its derivatives and the labels for grid and non-grid points
    dom_features = torch.cat([dom_points, vel_expred_dom, label_dom], dim=-1)
    grid_features_y = torch.cat([points_y_res.view(1, -1, 2), vel_y_expred, label_y], dim=-1)
    grid_features_x = torch.cat([points_x_res.view(1, -1, 2), vel_x_expred, label_x], dim=-1)

    # build a feature vector with location, velocity & its derivatives and the normal vectors for the boundary points
    b1_features = torch.cat([b1_points, vel_expred_b1, n1], dim=-1)
    b2_features = torch.cat([b2_points, vel_expred_b2, n2], dim=-1)
    b_features = torch.cat([b1_features, b2_features], dim=-2)

    # save the feature vectors
    dom_features_np = dom_features.cpu().numpy()
    b_features_np = b_features.cpu().numpy()
    grid_features_y_np = grid_features_y.cpu().numpy()
    grid_features_x_np = grid_features_x.cpu().numpy()

    np.savez_compressed(os.path.join(sample_dir, 'vortex_features.npz'), feat_expred)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_y.npz'), grid_features_y_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_x.npz'), grid_features_x_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_domain.npz'), dom_features_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_boundaries.npz'), b_features_np)



# Generate and save the validation set
for i in range(N_VAL_SAMPLES):

    sample_dir = os.path.join(val_dir, 'sim_' + '0' * (6 - len(str(i))) + str(i))
    os.makedirs(sample_dir)

    sample_yc = np.reshape(val_ycoords[i, :], (1, -1, 1))
    sample_xc = np.reshape(val_xcoords[i, :], (1, -1, 1))
    sample_tauc = np.reshape(val_strengths[i, :], (1, -1, 1))
    sample_sigc = np.reshape(val_sigmas[i, :], (1, -1, 1))

    feat_expred = np.concatenate([sample_yc, sample_xc, sample_tauc, sample_sigc], axis=-1)

    feat_expred_pt = torch.tensor(feat_expred, dtype=torch.float32, device='cuda:0')
    loc_index = torch.tensor([0, 1], device='cuda:0')

    points_y_res = points_y.view(1, -1, 2)
    points_x_res = points_x.view(1, -1, 2)

    vel_y_expred = VelDerivExpRed(feat_expred_pt, points_y_res)
    vel_x_expred = VelDerivExpRed(feat_expred_pt, points_x_res)

    p_locs = torch.index_select(feat_expred_pt, index=loc_index, dim=-1)
    p_locs_y, p_locs_x = torch.unbind(p_locs, dim=-1)

    dom_points_y_near_p = (p_locs_y.view(1, -1, 1) + torch.randn((1, 9000),
                                                                dtype=torch.float32,
                                                                device='cuda:0').view(1, p_locs_y.shape[-1], -1)).view(1, -1)
    dom_points_x_near_p = (p_locs_x.view(1, -1, 1) + torch.randn((1, 9000),
                                                                dtype=torch.float32,
                                                                device='cuda:0').view(1, p_locs_x.shape[-1], -1)).view(1, -1)
    dom_points_near_p = torch.stack([dom_points_y_near_p, dom_points_x_near_p], dim=-1)
    dom_points_y_rand = torch.rand(1000, dtype=torch.float32, device='cuda:0') * RESOLUTION[0]
    dom_points_x_rand = torch.rand(1000, dtype=torch.float32, device='cuda:0') * RESOLUTION[1]
    dom_points_rand = torch.stack([dom_points_y_rand, dom_points_x_rand], dim=-1).view(1, -1, 2)
    dom_points = torch.cat([dom_points_near_p, dom_points_rand], dim=-2)

    vel_expred_dom = VelDerivExpRed(feat_expred_pt, dom_points)

    b1_points_y = torch.rand(2000, dtype=torch.float32, device='cuda:0') * RESOLUTION[0]
    b1_points_x = torch.cat([torch.tensor([0] * 1000, dtype=torch.float32, device='cuda:0'),
                               torch.tensor([RESOLUTION[1]] * 1000, dtype=torch.float32, device='cuda:0')], dim=-1)

    b2_points_x = torch.rand(2000, dtype=torch.float32, device='cuda:0') * RESOLUTION[1]
    b2_points_y = torch.cat([torch.tensor([0] * 1000, dtype=torch.float32, device='cuda:0'),
                               torch.tensor([RESOLUTION[0]] * 1000, dtype=torch.float32, device='cuda:0')], dim=-1)

    b1_points = torch.stack([b1_points_y, b1_points_x], dim=-1).view(1, -1, 2)
    b2_points = torch.stack([b2_points_y, b2_points_x], dim=-1).view(1, -1, 2)

    vel_expred_b1 = VelDerivExpRed(feat_expred_pt, b1_points)
    vel_expred_b2 = VelDerivExpRed(feat_expred_pt, b2_points)

    n1 = torch.tensor([0.0, 1.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(b1_points.shape[1], 1)
    n2 = torch.tensor([1.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(b2_points.shape[1], 1)

    n1 = n1.view(1, -1, 2)
    n2 = n2.view(1, -1, 2)

    label_y = torch.tensor([1.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_y_expred.shape[1], 1)
    label_x = torch.tensor([0.0, 1.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_x_expred.shape[1], 1)
    label_dom = torch.tensor([0.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_expred_dom.shape[1], 1)

    label_y = label_y.view(1, -1, 2)
    label_x = label_x.view(1, -1, 2)
    label_dom = label_dom.view(1, -1, 2)

    dom_features = torch.cat([dom_points, vel_expred_dom, label_dom], dim=-1)
    grid_features_y = torch.cat([points_y_res.view(1, -1, 2), vel_y_expred, label_y], dim=-1)
    grid_features_x = torch.cat([points_x_res.view(1, -1, 2), vel_x_expred, label_x], dim=-1)
    b1_features = torch.cat([b1_points, vel_expred_b1, n1], dim=-1)
    b2_features = torch.cat([b2_points, vel_expred_b2, n2], dim=-1)
    b_features = torch.cat([b1_features, b2_features], dim=-2)

    dom_features_np = dom_features.cpu().numpy()
    b_features_np = b_features.cpu().numpy()
    grid_features_y_np = grid_features_y.cpu().numpy()
    grid_features_x_np = grid_features_x.cpu().numpy()

    np.savez_compressed(os.path.join(sample_dir, 'vortex_features.npz'), feat_expred)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_y.npz'), grid_features_y_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_x.npz'), grid_features_x_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_domain.npz'), dom_features_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_boundaries.npz'), b_features_np)



# Generate and save the test set
for i in range(N_TEST_SAMPLES):

    sample_dir = os.path.join(test_dir, 'sim_' + '0' * (6 - len(str(i))) + str(i))
    os.makedirs(sample_dir)

    sample_yc = np.reshape(test_ycoords[i, :], (1, -1, 1))
    sample_xc = np.reshape(test_xcoords[i, :], (1, -1, 1))
    sample_tauc = np.reshape(test_strengths[i, :], (1, -1, 1))
    sample_sigc = np.reshape(test_sigmas[i, :], (1, -1, 1))

    feat_expred = np.concatenate([sample_yc, sample_xc, sample_tauc, sample_sigc], axis=-1)

    feat_expred_pt = torch.tensor(feat_expred, dtype=torch.float32, device='cuda:0')
    loc_index = torch.tensor([0, 1], device='cuda:0')

    points_y_res = points_y.view(1, -1, 2)
    points_x_res = points_x.view(1, -1, 2)

    vel_y_expred = VelDerivExpRed(feat_expred_pt, points_y_res)
    vel_x_expred = VelDerivExpRed(feat_expred_pt, points_x_res)

    p_locs = torch.index_select(feat_expred_pt, index=loc_index, dim=-1)
    p_locs_y, p_locs_x = torch.unbind(p_locs, dim=-1)

    dom_points_y_near_p = (p_locs_y.view(1, -1, 1) + torch.randn((1, 9000),
                                                                dtype=torch.float32,
                                                                device='cuda:0').view(1, p_locs_y.shape[-1], -1)).view(1, -1)
    dom_points_x_near_p = (p_locs_x.view(1, -1, 1) + torch.randn((1, 9000),
                                                                dtype=torch.float32,
                                                                device='cuda:0').view(1, p_locs_x.shape[-1], -1)).view(1, -1)
    dom_points_near_p = torch.stack([dom_points_y_near_p, dom_points_x_near_p], dim=-1)
    dom_points_y_rand = torch.rand(1000, dtype=torch.float32, device='cuda:0') * RESOLUTION[0]
    dom_points_x_rand = torch.rand(1000, dtype=torch.float32, device='cuda:0') * RESOLUTION[1]
    dom_points_rand = torch.stack([dom_points_y_rand, dom_points_x_rand], dim=-1).view(1, -1, 2)
    dom_points = torch.cat([dom_points_near_p, dom_points_rand], dim=-2)

    vel_expred_dom = VelDerivExpRed(feat_expred_pt, dom_points)

    b1_points_y = torch.rand(2000, dtype=torch.float32, device='cuda:0') * RESOLUTION[0]
    b1_points_x = torch.cat([torch.tensor([0] * 1000, dtype=torch.float32, device='cuda:0'),
                               torch.tensor([RESOLUTION[1]] * 1000, dtype=torch.float32, device='cuda:0')], dim=-1)

    b2_points_x = torch.rand(2000, dtype=torch.float32, device='cuda:0') * RESOLUTION[1]
    b2_points_y = torch.cat([torch.tensor([0] * 1000, dtype=torch.float32, device='cuda:0'),
                               torch.tensor([RESOLUTION[0]] * 1000, dtype=torch.float32, device='cuda:0')], dim=-1)

    b1_points = torch.stack([b1_points_y, b1_points_x], dim=-1).view(1, -1, 2)
    b2_points = torch.stack([b2_points_y, b2_points_x], dim=-1).view(1, -1, 2)

    vel_expred_b1 = VelDerivExpRed(feat_expred_pt, b1_points)
    vel_expred_b2 = VelDerivExpRed(feat_expred_pt, b2_points)

    n1 = torch.tensor([0.0, 1.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(b1_points.shape[1], 1)
    n2 = torch.tensor([1.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(b2_points.shape[1], 1)

    n1 = n1.view(1, -1, 2)
    n2 = n2.view(1, -1, 2)

    label_y = torch.tensor([1.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_y_expred.shape[1], 1)
    label_x = torch.tensor([0.0, 1.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_x_expred.shape[1], 1)
    label_dom = torch.tensor([0.0, 0.0], dtype=torch.float32, device='cuda:0').view(1, -1).repeat(vel_expred_dom.shape[1], 1)

    label_y = label_y.view(1, -1, 2)
    label_x = label_x.view(1, -1, 2)
    label_dom = label_dom.view(1, -1, 2)

    dom_features = torch.cat([dom_points, vel_expred_dom, label_dom], dim=-1)
    grid_features_y = torch.cat([points_y_res.view(1, -1, 2), vel_y_expred, label_y], dim=-1)
    grid_features_x = torch.cat([points_x_res.view(1, -1, 2), vel_x_expred, label_x], dim=-1)
    b1_features = torch.cat([b1_points, vel_expred_b1, n1], dim=-1)
    b2_features = torch.cat([b2_points, vel_expred_b2, n2], dim=-1)
    b_features = torch.cat([b1_features, b2_features], dim=-2)
#
    dom_features_np = dom_features.cpu().numpy()
    b_features_np = b_features.cpu().numpy()
    grid_features_y_np = grid_features_y.cpu().numpy()
    grid_features_x_np = grid_features_x.cpu().numpy()

    np.savez_compressed(os.path.join(sample_dir, 'vortex_features.npz'), feat_expred)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_y.npz'), grid_features_y_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_x.npz'), grid_features_x_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_domain.npz'), dom_features_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_boundaries.npz'), b_features_np)






