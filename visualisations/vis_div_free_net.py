import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from core.datasets import DivFreeNetDataset
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import *
import os
from core.velocity_derivs import *
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/'
                                                    'data/bc_net_dataset_p10_gaussian_visc_8000/train/sim_000331/',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--n_domain_pts', type=int, default=1000, help='Number of points to sample for divegence loss')
parser.add_argument('--n_boundary_pts', type=int, default=100, help='Number of points to sample for boundary condition loss')
parser.add_argument('--load_weights_ex', type=str, default='BC_both_gaussian_weight_1.0_depth_5_100_batch_32_lr_1e-2_l2_1e-5_r120_8000_2', help='name of the experiment')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

opt = parser.parse_args()

case_dir = opt.case_path

vortex_features = np.load(os.path.join(case_dir, 'vortex_features.npz'))['arr_0']
# grid_features_y = np.load(os.path.join(case_dir, 'features_points_y.npz'))['arr_0']
# grid_features_x = np.load(os.path.join(case_dir, 'features_points_x.npz'))['arr_0']
velocity_0 = np.load(os.path.join(case_dir, 'velocity_000000.npz'))['arr_0']
velocity_0_div = np.load(os.path.join(case_dir, 'velocity_div_000000.npz'))['arr_0']

# grid_features_y_pt = torch.tensor(grid_features_y, dtype=torch.float32, device='cuda:0')
# grid_features_x_pt = torch.tensor(grid_features_x, dtype=torch.float32, device='cuda:0')
velocity_0_pt = torch.tensor(velocity_0, dtype=torch.float32, device='cuda:0')
velocity_0_div_pt = torch.tensor(velocity_0_div, dtype=torch.float32, device='cuda:0')
vortex_features_pt = torch.tensor(vortex_features, dtype=torch.float32, device='cuda:0')

domain = Domain(resolution=opt.domain, boundaries=CLOSED)
FLOW_REF = Fluid(domain=domain)

BCNet = BoundaryConditionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True, order=opt.order)

VelDerivExpRed = VelocityDerivatives(kernel='GaussianVorticity', order=2).to('cuda:0')

logs_dir = os.path.join('../logs_p10_gauss_BC', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])
params = torch.load(ckpt_file)['model_state_dict']
BCNet.load_state_dict(params)
#
BCNet.to('cuda:0')
BCNet.eval()

points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

grid_derivs_y_pt = VelDerivExpRed(vortex_features_pt, points_y)
grid_derivs_x_pt = VelDerivExpRed(vortex_features_pt, points_x)

grid_features_y_pt = torch.cat([points_y.view(1, -1, 2), grid_derivs_y_pt], dim=-1)
grid_features_x_pt = torch.cat([points_x.view(1, -1, 2), grid_derivs_x_pt], dim=-1)

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

b_normal_index = torch.tensor([14, 15], device='cuda:0')
b_vel_index = torch.tensor([2, 3], device='cuda:0')
loc_index = torch.tensor([0, 1], device='cuda:0')

if opt.order == 0:
    feat_index = torch.arange(4).to('cuda:0')
    deriv_index = torch.arange(2, 4).to('cuda:0')
elif opt.order == 1:
    feat_index = torch.arange(8).to('cuda:0')
    deriv_index = torch.arange(2, 8).to('cuda:0')
elif opt.order == 2:
    feat_index = torch.arange(14).to('cuda:0')
    deriv_index = torch.arange(2, 14).to('cuda:0')

inp_features_y = torch.index_select(grid_features_y_pt, dim=-1, index=feat_index)
inp_features_x = torch.index_select(grid_features_x_pt, dim=-1, index=feat_index)

vel_vortex_y = torch.index_select(grid_features_y_pt, dim=-1, index=b_vel_index)
vel_vortex_x = torch.index_select(grid_features_x_pt, dim=-1, index=b_vel_index)

vel_vortex_y_y, vel_vortex_y_x = torch.unbind(vel_vortex_y, dim=-1)
vel_vortex_x_y, vel_vortex_x_x = torch.unbind(vel_vortex_x, dim=-1)

out_y = BCNet(inp_features_y)
out_x = BCNet(inp_features_x)

corr_vel_y_y, corr_vel_y_x = torch.unbind(out_y, dim=-1)
corr_vel_x_y, corr_vel_x_x = torch.unbind(out_x, dim=-1)

vel_vortex_y_y = vel_vortex_y_y.view(1, points_y.shape[1], points_y.shape[2])
vel_vortex_x_x = vel_vortex_x_x.view(1, points_x.shape[1], points_x.shape[2])

corr_vel_y_y = corr_vel_y_y.view(1, points_y.shape[1], points_y.shape[2])
corr_vel_x_x = corr_vel_x_x.view(1, points_x.shape[1], points_x.shape[2])

total_vel_y = vel_vortex_y_y + corr_vel_y_y
total_vel_x = vel_vortex_x_x + corr_vel_x_x

total_vel_y_sg = torch.cat([total_vel_y, cat_y], dim=-1)
total_vel_x_sg = torch.cat([total_vel_x, cat_x], dim=-2)

total_vel_sg = torch.stack([total_vel_y_sg, total_vel_x_sg], dim=-1).detach().clone()

loss_before_div = F.mse_loss(velocity_0_pt, velocity_0_div_pt)
loss_after_div = F.mse_loss(total_vel_sg, velocity_0_div_pt)

max_val = velocity_0.max()
min_val = -max_val

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(velocity_0[0, :, :-1, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 2)
plt.imshow(velocity_0_div[0, :, :-1, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 3)
plt.imshow(total_vel_sg.cpu().numpy()[0, :, :-1, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.show()

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(velocity_0[0, :-1, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 2)
plt.imshow(velocity_0_div[0, :-1, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 3)
plt.imshow(total_vel_sg.cpu().numpy()[0, :-1, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.show()
