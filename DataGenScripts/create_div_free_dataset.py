import torch
from phi.flow import *
import argparse
import os
import matplotlib.pyplot as plt
from core.custom_functions import *
from core.velocity_derivs import *
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--data_dir', type=str, default='../'
                                                    'data/bc_net_dataset_p10_gaussian_visc_8000',
                    help='diretory corresponding to dataset with features of vortex particles')



# Parse input arguments
opt = parser.parse_args()

RESOLUTION = opt.domain
DIRECTORY = opt.data_dir


# training, validation and test directory
train_dir = os.path.join(DIRECTORY, 'train')
val_dir = os.path.join(DIRECTORY, 'val')
test_dir = os.path.join(DIRECTORY, 'test')

# list of data samples
train_samples = sorted(glob.glob(train_dir + '/*'))
val_samples = sorted(glob.glob(val_dir + '/*'))
test_samples = sorted(glob.glob(test_dir + '/*'))


# Gaussian falloff-kernel
falloff_kernel = GaussianFalloffKernelVelocity()


# define domain and resolution of the grid
domain = Domain(RESOLUTION, boundaries=CLOSED)
FLOW_REF = Fluid(domain)

# grid points in a staggered grid
points_y = torch.tensor(FLOW_REF.velocity.y.points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.x.points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, RESOLUTION[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, RESOLUTION[1] + 1), dtype=torch.float32, device='cuda:0')


# Generate and save the training set
for i in range(len(train_samples)):
    case_dir = train_samples[i]

    # read the vortex particle features with size (1, NPARTICLES, 4)
    vortex_features = np.load(os.path.join(case_dir, 'vortex_features.npz'))['arr_0']
    vortex_features_pt = torch.tensor(vortex_features, dtype=torch.float32, device='cuda:0')

    # velocity at the staggered grid points
    vel_y = falloff_kernel(vortex_features_pt, points_y)
    vel_x = falloff_kernel(vortex_features_pt, points_x)

    vel_y_y, vel_y_x = torch.unbind(vel_y, dim=-1)
    vel_x_y, vel_x_x = torch.unbind(vel_x, dim=-1)

    # staggered velocity tensor
    vel_0_pt = torch.stack([torch.cat([vel_y_y, cat_y], dim=-1), torch.cat([vel_x_x, cat_x], dim=-2)], dim=-1)
    vel_0_np = vel_0_pt.cpu().numpy()

    # staggered grid of velocity due to vortex particles
    vel_0_sg = StaggeredGrid(vel_0_np)

    # staggered grid of velocity after the pressure solve
    vel_0_div_free = divergence_free(vel_0_sg, domain=domain)

    # save the velocity fields
    np.savez_compressed(os.path.join(case_dir, 'velocity_000000.npz'), vel_0_sg.staggered_tensor())
    np.savez_compressed(os.path.join(case_dir, 'velocity_div_000000.npz'), vel_0_div_free.staggered_tensor())



# Generate and save the validation set
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



# Generate and save the test set
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

