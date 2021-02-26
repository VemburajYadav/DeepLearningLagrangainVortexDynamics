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
parser.add_argument('--strength_range', type=list, default=[-2, 2], help='range for strength sampling')
parser.add_argument('--strength_threshold', type=float, default=0.5, help='minimum value of magnitude of strength')
parser.add_argument('--sigma_range', type=list, default=[2.0, 10.0], help='range for core ize sampling')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--save_dir', type=str, default='/home/vemburaj/'
                                                    'data/bc_net_dataset_p10_gaussian_visc_8000',
                    help='diretory to save the generated dataset')

opt = parser.parse_args()

RESOLUTION = opt.domain
OFFSET = opt.offset
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

DIRECTORY = opt.save_dir

# strengths_pos = np.random.random_sample(size=(NSAMPLES * NPARTICLES // 2)) * (STRENGTH_RANGE[1] - STRENGTH_THRESHOLD_MAG) + STRENGTH_THRESHOLD_MAG
# strengths_neg = np.random.random_sample(size=(NSAMPLES * NPARTICLES // 2)) * (-STRENGTH_THRESHOLD_MAG - STRENGTH_RANGE[0]) + STRENGTH_RANGE[0]
#
# strengths = np.sort(np.concatenate([strengths_neg, strengths_pos]))
# sigmas = np.sort(np.random.random_sample(size=(NSAMPLES * NPARTICLES)) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0])
# ds = np.sort(np.random.random_sample(size=(NSAMPLES * NPARTICLES)) * (2.0 - 0.5) + 0.5)
#
# np.random.shuffle(strengths)
# np.random.shuffle(sigmas)
# np.random.shuffle(ds)


# ds = np.reshape(ds, (NSAMPLES, -1))

sigmas = np.reshape(np.random.random_sample(size=(NPARTICLES * NSAMPLES)) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0], (1, -1, 1))
facs = np.random.random_sample(size=(NPARTICLES * NSAMPLES)) * 15 + 5

np.random.shuffle(sigmas)
np.random.shuffle(facs)

rands = np.array([-1] * (NSAMPLES * NPARTICLES // 2) + [1] * (NSAMPLES * NPARTICLES // 2))
np.random.shuffle(rands)

strengths = facs * sigmas.reshape((-1)) * rands
strengths = np.reshape(strengths, (-1,))

np.random.shuffle(strengths)
np.random.shuffle(sigmas)

# strengths = np.reshape(strengths, (NSAMPLES, -1))
# sigmas = np.reshape(sigmas, (NSAMPLES, -1))

strengths = np.reshape(strengths, (NSAMPLES, -1))
sigmas = np.reshape(sigmas, (NSAMPLES, -1))

ycoords = np.empty((NSAMPLES, NPARTICLES))
xcoords = np.empty((NSAMPLES, NPARTICLES))

domain = Domain(RESOLUTION, boundaries=CLOSED)
FLOW_REF = Fluid(domain)

points_y = torch.tensor(FLOW_REF.velocity.y.points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.x.points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, RESOLUTION[0] + 1, 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, RESOLUTION[1] + 1, 1), dtype=torch.float32, device='cuda:0')

VelDerivExpRed = VelocityDerivatives(kernel='GaussianVorticity', order=2).to('cuda:0')

for i in range(NSAMPLES):
    ycoords[i, :] = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[0] + OFFSET[0]
    xcoords[i, :] = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[1] + OFFSET[1]

train_ycoords, train_xcoords = ycoords[0: N_TRAIN_SAMPLES, :], xcoords[0: N_TRAIN_SAMPLES, :]
train_strengths, train_sigmas = strengths[0:N_TRAIN_SAMPLES, :], sigmas[0: N_TRAIN_SAMPLES, :]

val_ycoords, val_xcoords = ycoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :],\
                           xcoords[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]
val_strengths, val_sigmas = strengths[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :], \
                                    sigmas[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES), :]

test_ycoords, test_xcoords = ycoords[-N_TEST_SAMPLES:, :], xcoords[-N_TEST_SAMPLES:, :]
test_strengths, test_sigmas = strengths[-N_TEST_SAMPLES:, :], sigmas[-N_TEST_SAMPLES:, :]

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

for i in range(N_TRAIN_SAMPLES):

    sample_dir = os.path.join(train_dir, 'sim_' + '0' * (6 - len(str(i))) + str(i))
    os.makedirs(sample_dir)

    sample_yc = np.reshape(train_ycoords[i, :], (1, -1, 1))
    sample_xc = np.reshape(train_xcoords[i, :], (1, -1, 1))
    sample_tauc = np.reshape(train_strengths[i, :], (1, -1, 1))
    sample_sigc = np.reshape(train_sigmas[i, :], (1, -1, 1))
    # sample_dc = np.reshape(train_ds[i, :], (1, -1, 1))

    feat_expred = np.concatenate([sample_yc, sample_xc, sample_tauc, sample_sigc], axis=-1)

    feat_expred_pt = torch.tensor(feat_expred, dtype=torch.float32, device='cuda:0')
    loc_index = torch.tensor([0, 1], device='cuda:0')

    points_y_res = points_y.view(1, -1, 2)
    points_x_res = points_x.view(1, -1, 2)

    # points_y_rand_index = torch.randint(0, points_y_res.shape[1], (1000, ), device='cuda:0')
    # points_x_rand_index = torch.randint(0, points_x_res.shape[1], (1000, ), device='cuda:0')
    #
    # points_y_rand = torch.index_select(points_y_res, dim=1, index=points_y_rand_index)
    # points_x_rand = torch.index_select(points_x_res, dim=1, index=points_x_rand_index)
    #
    vel_y_expred = VelDerivExpRed(feat_expred_pt, points_y_res)
    vel_x_expred = VelDerivExpRed(feat_expred_pt, points_x_res)

    # vel_y_y_expred = torch.index_select(vel_y_expred,
    #                                     index=torch.tensor([0], device='cuda:0'),
    #                                     dim=-1).view(1, points_y.shape[1], points_y.shape[2], -1)
    # vel_x_x_expred = torch.index_select(vel_x_expred,
    #                                     index=torch.tensor([1], device='cuda:0'),
    #                                     dim=-1).view(1, points_x.shape[1], points_x.shape[2], -1)
    # vel_expred = torch.cat([torch.cat([vel_y_y_expred, cat_y], dim=-2), torch.cat([vel_x_x_expred, cat_x], dim=-3)], dim=-1)


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

    # vel_expred_sg = StaggeredGrid(vel_expred.cpu().numpy())
    # vel_expred_sg_div_free = divergence_free(vel_expred_sg, domain=domain)


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
    b1_features = torch.cat([b1_points, vel_expred_b1, n1], dim=-1)
    b2_features = torch.cat([b2_points, vel_expred_b2, n2], dim=-1)
    b_features = torch.cat([b1_features, b2_features], dim=-2)
    grid_features_y = torch.cat([points_y_res.view(1, -1, 2), vel_y_expred, label_y], dim=-1)
    grid_features_x = torch.cat([points_x_res.view(1, -1, 2), vel_x_expred, label_x], dim=-1)

    dom_features_np = dom_features.cpu().numpy()
    # b1_features_np = b1_features.cpu().numpy()
    # b2_features_np = b2_features.cpu().numpy()
    b_features_np = b_features.cpu().numpy()
    grid_features_y_np = grid_features_y.cpu().numpy()
    grid_features_x_np = grid_features_x.cpu().numpy()

    np.savez_compressed(os.path.join(sample_dir, 'vortex_features.npz'), feat_expred)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_y.npz'), grid_features_y_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_x.npz'), grid_features_x_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_domain.npz'), dom_features_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_boundaries.npz'), b_features_np)



for i in range(N_VAL_SAMPLES):

    sample_dir = os.path.join(val_dir, 'sim_' + '0' * (6 - len(str(i))) + str(i))
    os.makedirs(sample_dir)

    sample_yc = np.reshape(val_ycoords[i, :], (1, -1, 1))
    sample_xc = np.reshape(val_xcoords[i, :], (1, -1, 1))
    sample_tauc = np.reshape(val_strengths[i, :], (1, -1, 1))
    sample_sigc = np.reshape(val_sigmas[i, :], (1, -1, 1))
    # sample_dc = np.reshape(val_ds[i, :], (1, -1, 1))

    feat_expred = np.concatenate([sample_yc, sample_xc, sample_tauc, sample_sigc], axis=-1)

    feat_expred_pt = torch.tensor(feat_expred, dtype=torch.float32, device='cuda:0')
    loc_index = torch.tensor([0, 1], device='cuda:0')

    points_y_res = points_y.view(1, -1, 2)
    points_x_res = points_x.view(1, -1, 2)

    # points_y_rand_index = torch.randint(0, points_y_res.shape[1], (1000, ), device='cuda:0')
    # points_x_rand_index = torch.randint(0, points_x_res.shape[1], (1000, ), device='cuda:0')
    #
    # points_y_rand = torch.index_select(points_y_res, dim=1, index=points_y_rand_index)
    # points_x_rand = torch.index_select(points_x_res, dim=1, index=points_x_rand_index)
    #
    vel_y_expred = VelDerivExpRed(feat_expred_pt, points_y_res)
    vel_x_expred = VelDerivExpRed(feat_expred_pt, points_x_res)

    # vel_y_y_expred = torch.index_select(vel_y_expred,
    #                                     index=torch.tensor([0], device='cuda:0'),
    #                                     dim=-1).view(1, points_y.shape[1], points_y.shape[2], -1)
    # vel_x_x_expred = torch.index_select(vel_x_expred,
    #                                     index=torch.tensor([1], device='cuda:0'),
    #                                     dim=-1).view(1, points_x.shape[1], points_x.shape[2], -1)
    # vel_expred = torch.cat([torch.cat([vel_y_y_expred, cat_y], dim=-2), torch.cat([vel_x_x_expred, cat_x], dim=-3)], dim=-1)
    #

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

    # vel_expred_sg = StaggeredGrid(vel_expred.cpu().numpy())
    # vel_expred_sg_div_free = divergence_free(vel_expred_sg, domain=domain)


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
    b1_features = torch.cat([b1_points, vel_expred_b1, n1], dim=-1)
    b2_features = torch.cat([b2_points, vel_expred_b2, n2], dim=-1)
    b_features = torch.cat([b1_features, b2_features], dim=-2)
    grid_features_y = torch.cat([points_y_res.view(1, -1, 2), vel_y_expred, label_y], dim=-1)
    grid_features_x = torch.cat([points_x_res.view(1, -1, 2), vel_x_expred, label_x], dim=-1)

    dom_features_np = dom_features.cpu().numpy()
    # b1_features_np = b1_features.cpu().numpy()
    # b2_features_np = b2_features.cpu().numpy()
    b_features_np = b_features.cpu().numpy()
    grid_features_y_np = grid_features_y.cpu().numpy()
    grid_features_x_np = grid_features_x.cpu().numpy()

    np.savez_compressed(os.path.join(sample_dir, 'vortex_features.npz'), feat_expred)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_y.npz'), grid_features_y_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_x.npz'), grid_features_x_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_domain.npz'), dom_features_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_boundaries.npz'), b_features_np)


for i in range(N_TEST_SAMPLES):

    sample_dir = os.path.join(test_dir, 'sim_' + '0' * (6 - len(str(i))) + str(i))
    os.makedirs(sample_dir)

    sample_yc = np.reshape(test_ycoords[i, :], (1, -1, 1))
    sample_xc = np.reshape(test_xcoords[i, :], (1, -1, 1))
    sample_tauc = np.reshape(test_strengths[i, :], (1, -1, 1))
    sample_sigc = np.reshape(test_sigmas[i, :], (1, -1, 1))
    # sample_dc = np.reshape(test_ds[i, :], (1, -1, 1))

    feat_expred = np.concatenate([sample_yc, sample_xc, sample_tauc, sample_sigc], axis=-1)

    feat_expred_pt = torch.tensor(feat_expred, dtype=torch.float32, device='cuda:0')
    loc_index = torch.tensor([0, 1], device='cuda:0')

    points_y_res = points_y.view(1, -1, 2)
    points_x_res = points_x.view(1, -1, 2)

    # points_y_rand_index = torch.randint(0, points_y_res.shape[1], (1000, ), device='cuda:0')
    # points_x_rand_index = torch.randint(0, points_x_res.shape[1], (1000, ), device='cuda:0')
    #
    # points_y_rand = torch.index_select(points_y_res, dim=1, index=points_y_rand_index)
    # points_x_rand = torch.index_select(points_x_res, dim=1, index=points_x_rand_index)

    vel_y_expred = VelDerivExpRed(feat_expred_pt, points_y_res)
    vel_x_expred = VelDerivExpRed(feat_expred_pt, points_x_res)

    # vel_y_y_expred = torch.index_select(vel_y_expred,
    #                                     index=torch.tensor([0], device='cuda:0'),
    #                                     dim=-1).view(1, points_y.shape[1], points_y.shape[2], -1)
    # vel_x_x_expred = torch.index_select(vel_x_expred,
    #                                     index=torch.tensor([1], device='cuda:0'),
    #                                     dim=-1).view(1, points_x.shape[1], points_x.shape[2], -1)
    # vel_expred = torch.cat([torch.cat([vel_y_y_expred, cat_y], dim=-2), torch.cat([vel_x_x_expred, cat_x], dim=-3)], dim=-1)


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

    # vel_expred_sg = StaggeredGrid(vel_expred.cpu().numpy())
    # vel_expred_sg_div_free = divergence_free(vel_expred_sg, domain=domain)


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
    b1_features = torch.cat([b1_points, vel_expred_b1, n1], dim=-1)
    b2_features = torch.cat([b2_points, vel_expred_b2, n2], dim=-1)
    b_features = torch.cat([b1_features, b2_features], dim=-2)
    grid_features_y = torch.cat([points_y_res.view(1, -1, 2), vel_y_expred, label_y], dim=-1)
    grid_features_x = torch.cat([points_x_res.view(1, -1, 2), vel_x_expred, label_x], dim=-1)

    dom_features_np = dom_features.cpu().numpy()
    # b1_features_np = b1_features.cpu().numpy()
    # b2_features_np = b2_features.cpu().numpy()
    b_features_np = b_features.cpu().numpy()
    grid_features_y_np = grid_features_y.cpu().numpy()
    grid_features_x_np = grid_features_x.cpu().numpy()

    np.savez_compressed(os.path.join(sample_dir, 'vortex_features.npz'), feat_expred)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_y.npz'), grid_features_y_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_points_x.npz'), grid_features_x_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_domain.npz'), dom_features_np)
    np.savez_compressed(os.path.join(sample_dir, 'features_boundaries.npz'), b_features_np)






