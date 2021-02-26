import torch
from phi.flow import *
import argparse
import os
import matplotlib.pyplot as plt
from core.custom_functions import *
from core.velocity_derivs import *
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[100, 100], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--n_particles', type=int, default=10, help='number of vortex particles')
parser.add_argument('--strength_range', type=list, default=[-2, 2], help='range for strength sampling')
parser.add_argument('--strength_threshold', type=float, default=0.5, help='minimum value of magnitude of strength')
parser.add_argument('--sigma_range', type=list, default=[20.0, 40.0], help='range for core ize sampling')
parser.add_argument('--viscosity_range', type=list, default=[0.0, 0.5], help='range for core ize sampling')
parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data sampled from each zone for '
                                                                     'training')
parser.add_argument('--eval_percent', type=float, default=0.2, help='percentage of data sampled from each zone for '
                                                                    'validation')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/'
                                                    'data/bc_net_dataset_p50_small',
                    help='diretory to save the generated dataset')

opt = parser.parse_args()


RESOLUTION = opt.domain
NPARTICLES = opt.n_particles
STRENGTH_RANGE = opt.strength_range
SIGMA_RANGE = opt.sigma_range
STRENGTH_THRESHOLD_MAG = opt.strength_threshold
VISCOSITY_RANGE = opt.viscosity_range

TRAIN_PERCENT = opt.train_percent
VAL_PERCENT = opt.eval_percent

DIRECTORY = opt.data_dir

train_dir = os.path.join(DIRECTORY, 'train')
val_dir = os.path.join(DIRECTORY, 'val')
test_dir = os.path.join(DIRECTORY, 'test')

train_samples = sorted(glob.glob(train_dir + '/*'))
val_samples = sorted(glob.glob(val_dir + '/*'))
test_samples = sorted(glob.glob(test_dir + '/*'))

falloff_kernel = GaussExpFalloffKernelReduced()

domain = Domain(RESOLUTION, boundaries=STICKY)
FLOW_REF = Fluid(domain)

points_y = torch.tensor(FLOW_REF.velocity.y.points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.x.points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, RESOLUTION[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, RESOLUTION[1] + 1), dtype=torch.float32, device='cuda:0')


NSAMPLES = len(train_samples) + len(val_samples) + len(test_samples)
N_TRAIN_SAMPLES = len(train_samples)
N_VAL_SAMPLES = len(val_samples)
N_TEST_SAMPLES = len(test_samples)

viscositys = np.sort(np.random.random_sample(size=(NSAMPLES)) * (VISCOSITY_RANGE[1] - VISCOSITY_RANGE[0]) + VISCOSITY_RANGE[0])
np.random.shuffle(viscositys)

train_viscositites = viscositys[0:N_TRAIN_SAMPLES]
val_viscositites = viscositys[N_TRAIN_SAMPLES: (N_TRAIN_SAMPLES + N_VAL_SAMPLES)]
test_viscosities = viscositys[-N_TEST_SAMPLES:]


for i in range(len(train_samples)):
    case_dir = train_samples[i]
    vortex_features = np.load(os.path.join(case_dir, 'vortex_features.npz'))['arr_0']

    vortex_features_pt = torch.tensor(vortex_features, dtype=torch.float32, device='cuda:0')

    vel_y = falloff_kernel(vortex_features_pt, points_y)
    vel_x = falloff_kernel(vortex_features_pt, points_x)

    vel_y_y, vel_y_x = torch.unbind(vel_y, dim=-1)
    vel_x_y, vel_x_x = torch.unbind(vel_x, dim=-1)

    vel_0_pt = torch.stack([torch.cat([vel_y_y, cat_y], dim=-1), torch.cat([vel_x_x, cat_x], dim=-2)], dim=-1)
    vel_0_np = vel_0_pt.cpu().numpy()

    vel_0_sg = StaggeredGrid(vel_0_np)
    vel_0_div_free = divergence_free(vel_0_sg, domain=domain)

    np.savez_compressed(os.path.join(case_dir, 'velocity_000000.npz'), vel_0_sg.staggered_tensor())
    np.savez_compressed(os.path.join(case_dir, 'velocity_div_000000.npz'), vel_0_div_free.staggered_tensor())

    print(i)

for i in range(len(val_samples)):
    case_dir = val_samples[i]
    vortex_features = np.load(os.path.join(case_dir, 'vortex_features.npz'))['arr_0']

    vortex_features_pt = torch.tensor(vortex_features, dtype=torch.float32, device='cuda:0')

    vel_y = falloff_kernel(vortex_features_pt, points_y)
    vel_x = falloff_kernel(vortex_features_pt, points_x)

    vel_y_y, vel_y_x = torch.unbind(vel_y, dim=-1)
    vel_x_y, vel_x_x = torch.unbind(vel_x, dim=-1)

    vel_0_pt = torch.stack([torch.cat([vel_y_y, cat_y], dim=-1), torch.cat([vel_x_x, cat_x], dim=-2)], dim=-1)
    vel_0_np = vel_0_pt.cpu().numpy()

    vel_0_sg = StaggeredGrid(vel_0_np)
    vel_0_div_free = divergence_free(vel_0_sg, domain=domain)

    np.savez_compressed(os.path.join(case_dir, 'velocity_000000.npz'), vel_0_sg.staggered_tensor())
    np.savez_compressed(os.path.join(case_dir, 'velocity_div_000000.npz'), vel_0_div_free.staggered_tensor())

    print(i)

for i in range(len(test_samples)):
    case_dir = test_samples[i]
    vortex_features = np.load(os.path.join(case_dir, 'vortex_features.npz'))['arr_0']

    vortex_features_pt = torch.tensor(vortex_features, dtype=torch.float32, device='cuda:0')

    vel_y = falloff_kernel(vortex_features_pt, points_y)
    vel_x = falloff_kernel(vortex_features_pt, points_x)

    vel_y_y, vel_y_x = torch.unbind(vel_y, dim=-1)
    vel_x_y, vel_x_x = torch.unbind(vel_x, dim=-1)

    vel_0_pt = torch.stack([torch.cat([vel_y_y, cat_y], dim=-1), torch.cat([vel_x_x, cat_x], dim=-2)], dim=-1)
    vel_0_np = vel_0_pt.cpu().numpy()

    vel_0_sg = StaggeredGrid(vel_0_np)
    vel_0_div_free = divergence_free(vel_0_sg, domain=domain)

    np.savez_compressed(os.path.join(case_dir, 'velocity_000000.npz'), vel_0_sg.staggered_tensor())
    np.savez_compressed(os.path.join(case_dir, 'velocity_div_000000.npz'), vel_0_div_free.staggered_tensor())

    print(i)

