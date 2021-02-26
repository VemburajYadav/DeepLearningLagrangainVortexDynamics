import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from core.datasets import VortexBoundariesDataset
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import *
import os
from core.velocity_derivs import *
import json
import copy

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[100, 100], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/'
                                                    'data/p10_b_sb_dataset_100x100_4000/train/sim_000520',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='p10_b_BC_T2_exp_red_weight_1.0_depth_5_100_batch_16_lr_1e-2_l2_1e-5_r100_4000_2',
                    help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=2, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# MEAN = [64.0, 0.0, 27.5]
# STDDEV = [23.094, 1.52752, 12.9903]

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride
RESOLUTION = opt.domain

mean_tensor = torch.tensor(MEAN, dtype=torch.float32, device='cuda:0')
stddev_tensor = torch.tensor(MEAN, dtype=torch.float32, device='cuda:0')

opt = parser.parse_args()
case_dir = opt.case_path

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

nparticles = location.shape[1]

velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

velocities[0] = np.load(os.path.join(case_dir, 'velocity_div_000000.npz'))['arr_0']
velocities.append(np.load(os.path.join(case_dir, 'velocity_000000.npz'))['arr_0'])

domain_invis = Domain(resolution=opt.domain, boundaries=SLIPPERY)
flow_invis = Fluid(domain=domain_invis, velocity=velocities[-1])

domain_vis = Domain(resolution=opt.domain, boundaries=STICKY)
flow_vis = Fluid(domain=domain_vis, velocity=copy.deepcopy(velocities[-1]))

FLOW_REF_vis = Fluid(domain=domain_vis)
# fluid_vel_vis = copy.deepcopy(flow_vis.velocity)
# fluid_vel_invis = copy.deepcopy(flow_invis.velocity)

fluid_vel_vis_diff = flow_vis.copied_with(diffuse(flow_vis.velocity, 4.0, substeps=20))
fluid_vel_invis_div = divergence_free(flow_invis.velocity, domain=domain_invis)
fluid_vel_vis_div = divergence_free(fluid_vel_vis_diff.velocity, domain=domain_vis)

max_val = flow_invis.velocity.x.data.max()
min_val = -max_val

# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(flow_invis.velocity.x.data[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('Velocity field from Vorticity')
# plt.subplot(1, 3, 2)
# plt.imshow(fluid_vel_vis_diff.velocity.x.data[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('After diffusion')
# plt.subplot(1, 3, 3)
# plt.imshow(fluid_vel_vis_div.x.data[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('After Pressuree Solve')
# plt.suptitle('Velocity-x')
# plt.show()

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(flow_invis.velocity.y.data[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('Velocity field from Vorticity')
plt.subplot(1, 3, 2)
plt.imshow(fluid_vel_vis_diff.velocity.y.data[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('After diffusion')
plt.subplot(1, 3, 3)
plt.imshow(fluid_vel_vis_div.y.data[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('After Pressure Solve')
plt.suptitle('Velocity-y')
plt.show()
#
# g = flow_vis.velocity.padded(10)
# plt.figure()
# plt.imshow(g.x.data[0, :, :, 0], cmap='RdYlBu')
# plt.show()