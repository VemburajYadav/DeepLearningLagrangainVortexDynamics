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
from visualisations.my_plot import set_size

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/'
                                                    'data/p10_b_sb_gaussian_dataset_120x120_4000/train/sim_000530',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='VortexBCNet',
                    help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=1, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='GaussianVorticity', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# MEAN = [64.0, 0.0, 27.5]
# STDDEV = [23.094, 1.52752, 12.9903]

width = 455.24408

# save_dir = '/home/vemburaj/Desktop/Report_Plots/Chapter3/'
save_dir = '/home/vemburaj/Desktop/TwoParticle/Case6'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

plt.style.use('tex')

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

domain = Domain(resolution=opt.domain, boundaries=CLOSED)
FLOW_REF = Fluid(domain=domain)

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

nparticles = location.shape[1]

velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

velocities[0] = np.load(os.path.join(case_dir, 'velocity_div_000000.npz'))['arr_0']
velocities.append(np.load(os.path.join(case_dir, 'velocity_000000.npz'))['arr_0'])

velocities_cg = []

for i in range(len(velocities)):
    domain_ = Domain(resolution=opt.domain, boundaries=CLOSED)
    FLOW_REF_ = Fluid(domain=domain, velocity=velocities[i])
    velocities_cg.append(FLOW_REF_.velocity.at(FLOW_REF_.density).data)


logs_dir = os.path.join('../logs_p10_gauss_BC', opt.load_weights_ex)
logs_dir_VN = os.path.join(logs_dir, 'VortexNet')
logs_dir_BC = os.path.join(logs_dir, 'BCNet')

ckpt_dir_VN = os.path.join(logs_dir_VN, 'ckpt')
ckpt_dir_BC = os.path.join(logs_dir_BC, 'ckpt')

checkpoints_files_VN = os.listdir(os.path.join(ckpt_dir_VN))
epoch_id_VN = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files_VN]))
ckpt_file_VN = os.path.join(ckpt_dir_VN, checkpoints_files_VN[epoch_id_VN])

checkpoints_files_BC = os.listdir(os.path.join(ckpt_dir_BC))
epoch_id_BC = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files_BC]))
ckpt_file_BC = os.path.join(ckpt_dir_BC, checkpoints_files_BC[epoch_id_BC])

params_VN = torch.load(ckpt_file_VN)['model_state_dict']
params_BC = torch.load(ckpt_file_BC)['model_state_dict']

VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV, order=opt.order,
                                        num_steps=opt.num_time_steps, distinct_nets=opt.distinct_nets)
BCNet = BoundaryConditionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True, order=opt.order)

VortexNet.single_step_net.load_state_dict(params_VN)
BCNet.load_state_dict(params_BC)

velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in range(len(velocities))]
velocities_gpu_cg = [torch.tensor(velocities_cg[i], dtype=torch.float32, device='cuda:0') for i in range(len(velocities))]

loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')


points_y_ = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x_ = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

points_cg = torch.tensor(FLOW_REF.density.points.data, dtype=torch.float32, device='cuda:0')
cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')


deriv_module = VelocityDerivatives(order=opt.order, kernel=opt.kernel).to('cuda:0')

VortexNet.to('cuda:0')
VortexNet = VortexNet.eval()

if opt.kernel == 'offset-gaussian':
    off0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    sigl0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), v0, u0, off0, sigl0], dim=-1)
    falloff_kernel = OffsetGaussianFalloffKernel()
if opt.kernel == 'ExpGaussian':
    c0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0')
    d0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), c0, d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernel(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
if opt.kernel == 'ExpGaussianRed':
    d0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernelReduced(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
if opt.kernel == 'GaussianVorticity':
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles)], dim=-1)
    falloff_kernel = GaussianFalloffKernelVelocity()
elif opt.kernel == 'gaussian':
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1)], dim=-1)
    falloff_kernel = GaussianFalloffKernel()

pred_velocites = []
losses= []

vortex_features_out = VortexNet(inp_feature.detach().clone())
vortex_features = [vortex_features_out[i].clone().detach() for i in range(len(vortex_features_out))]

with torch.no_grad():
    for step in range(NUM_TIME_STEPS + 1):
        vel_y = falloff_kernel(vortex_features[step], points_y_)
        vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
        vel_x = falloff_kernel(vortex_features[step], points_x_)
        vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
        vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
        pred_velocites.append(vel.detach().clone())
        losses.append(F.mse_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())

    loss_all = torch.stack(losses, dim=-1)
    features = torch.stack(vortex_features, dim=-1)

BATCH_SIZE = 1

# points_ph = points_cg.view(1, -1, 2)
points_y = points_y_.view(1, -1, 2)
points_x = points_x_.view(1, -1, 2)

y_ph_y, x_ph_y = torch.unbind(points_y, dim=-1)
y_ph_x, x_ph_x = torch.unbind(points_x, dim=-1)

derivs_list_y = []
derivs_list_x = []

velocity_field_by_vortex_y = []
velocity_field_by_vortex_x = []

for i in range(NUM_TIME_STEPS + 1):
    derivs_list_y.append(deriv_module(vortex_features[i], points_y))
    derivs_list_x.append(deriv_module(vortex_features[i], points_x))

    velocity_field_by_vortex_y.append(
        falloff_kernel(vortex_features[i], points_y.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
    velocity_field_by_vortex_x.append(
        falloff_kernel(vortex_features[i], points_x.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))

BCNet.to('cuda:0')
BCNet = BCNet.eval()

# Compute divergence loss
div_dv_dy_list_y = []
div_dv_dy_list_x = []

div_du_dx_list_y = []
div_du_dx_list_x = []

corr_vel_y_list_y = []
corr_vel_y_list_x = []

corr_vel_x_list_y = []
corr_vel_x_list_x = []

for b in range(BATCH_SIZE):
    div_b_dv_dy_list_y = []
    div_b_dv_dy_list_x = []

    div_b_du_dx_list_y = []
    div_b_du_dx_list_x = []

    corr_vel_y_b_list_y = []
    corr_vel_y_b_list_x = []

    corr_vel_x_b_list_y = []
    corr_vel_x_b_list_x = []

    for t in range(NUM_TIME_STEPS + 1):
        paxes_tensor = torch.tensor([b], device='cuda:0')
        y_b_ph_y = torch.index_select(y_ph_y, dim=0, index=paxes_tensor).view(-1)
        x_b_ph_y = torch.index_select(x_ph_y, dim=0, index=paxes_tensor).view(-1)
        y_b_ph_y.requires_grad_(True)
        x_b_ph_y.requires_grad_(True)
        loc_b_ph_y = torch.stack([y_b_ph_y, x_b_ph_y], dim=-1)
        deriv_b_ph_y = torch.index_select(derivs_list_y[t], dim=0, index=paxes_tensor)
        inp_b_ph_y = torch.cat([loc_b_ph_y.view(1, -1, 2), deriv_b_ph_y], dim=-1)
        out_b_ph_y = BCNet(inp_b_ph_y)
        corr_vel_y_bh_y, corr_vel_x_bh_y = torch.unbind(out_b_ph_y.view(-1, 2), dim=-1)
        corr_vel_y_b_list_y.append(corr_vel_y_bh_y)
        corr_vel_x_b_list_y.append(corr_vel_x_bh_y)

        div_dv_dy_y = \
            torch.autograd.grad(torch.unbind(corr_vel_y_bh_y, dim=-1), y_b_ph_y, retain_graph=True, allow_unused=True)[
                0]
        div_du_dx_y = torch.autograd.grad(torch.unbind(corr_vel_x_bh_y, dim=-1), x_b_ph_y, allow_unused=True)[0]
        div_b_dv_dy_list_y.append(div_dv_dy_y)
        div_b_du_dx_list_y.append(div_du_dx_y)

        y_b_ph_x = torch.index_select(y_ph_x, dim=0, index=paxes_tensor).view(-1)
        x_b_ph_x = torch.index_select(x_ph_x, dim=0, index=paxes_tensor).view(-1)
        y_b_ph_x.requires_grad_(True)
        x_b_ph_x.requires_grad_(True)
        loc_b_ph_x = torch.stack([y_b_ph_x, x_b_ph_x], dim=-1)
        deriv_b_ph_x = torch.index_select(derivs_list_x[t], dim=0, index=paxes_tensor)
        inp_b_ph_x = torch.cat([loc_b_ph_x.view(1, -1, 2), deriv_b_ph_x], dim=-1)
        out_b_ph_x = BCNet(inp_b_ph_x)
        corr_vel_y_bh_x, corr_vel_x_bh_x = torch.unbind(out_b_ph_x.view(-1, 2), dim=-1)
        corr_vel_y_b_list_x.append(corr_vel_y_bh_x)
        corr_vel_x_b_list_x.append(corr_vel_x_bh_x)

        div_dv_dy_x = \
            torch.autograd.grad(torch.unbind(corr_vel_y_bh_x, dim=-1), y_b_ph_x, retain_graph=True, allow_unused=True)[
                0]
        div_du_dx_x = torch.autograd.grad(torch.unbind(corr_vel_x_bh_x, dim=-1), x_b_ph_x, allow_unused=True)[0]
        div_b_dv_dy_list_x.append(div_dv_dy_x)
        div_b_du_dx_list_x.append(div_du_dx_x)

    div_dv_dy_list_x.append(torch.stack(div_b_dv_dy_list_x, dim=-1))
    div_du_dx_list_x.append(torch.stack(div_b_du_dx_list_x, dim=-1))
    corr_vel_y_list_x.append(torch.stack(corr_vel_y_b_list_x, dim=-1))
    corr_vel_x_list_x.append(torch.stack(corr_vel_x_b_list_x, dim=-1))

    div_dv_dy_list_y.append(torch.stack(div_b_dv_dy_list_y, dim=-1))
    div_du_dx_list_y.append(torch.stack(div_b_du_dx_list_y, dim=-1))
    corr_vel_y_list_y.append(torch.stack(corr_vel_y_b_list_y, dim=-1))
    corr_vel_x_list_y.append(torch.stack(corr_vel_x_b_list_y, dim=-1))

div_dv_dy_all_y = torch.stack(div_dv_dy_list_y, dim=0)
div_du_dx_all_y = torch.stack(div_du_dx_list_y, dim=0)
corr_vel_x_all_y = torch.stack(corr_vel_x_list_y, dim=0).view(1, points_y_.shape[1], points_y_.shape[2], -1)
corr_vel_y_all_y = torch.stack(corr_vel_y_list_y, dim=0).view(1, points_y_.shape[1], points_y_.shape[2], -1)
corr_vel_all_y = torch.stack([corr_vel_y_all_y, corr_vel_x_all_y], dim=-2)
pred_corr_velocities_y = torch.unbind(corr_vel_all_y, dim=-1)

div_dv_dy_all_x = torch.stack(div_dv_dy_list_x, dim=0)
div_du_dx_all_x = torch.stack(div_du_dx_list_x, dim=0)
corr_vel_x_all_x = torch.stack(corr_vel_x_list_x, dim=0).view(1, points_x_.shape[1], points_x_.shape[2], -1)
corr_vel_y_all_x = torch.stack(corr_vel_y_list_x, dim=0).view(1, points_x_.shape[1], points_x_.shape[2], -1)
corr_vel_all_x = torch.stack([corr_vel_y_all_x, corr_vel_x_all_x], dim=-2)

final_vel_y_sg = torch.unbind(corr_vel_y_all_y, dim=-1)
final_vel_x_sg = torch.unbind(corr_vel_x_all_x, dim=-1)

final_corr_velocities = [torch.stack([torch.cat([final_vel_y_sg[i], cat_y], dim=-1),
                                      torch.cat([final_vel_x_sg[i], cat_x], dim=-2)], dim=-1) for i in range(len(pred_velocites))]


pred_total_velocities = [pred_velocites[i] + final_corr_velocities[i].detach().clone() for i in range(len(pred_velocites))]

losses_total = []

with torch.no_grad():
    for step in range(NUM_TIME_STEPS + 1):
        losses_total.append(F.mse_loss(pred_total_velocities[step], velocities_gpu[step], reduction='sum').detach().clone())
    losses_total_all = torch.stack(losses_total, dim=-1)

div_ph_loss_y = torch.sum((div_du_dx_all_y + div_dv_dy_all_y) ** 2) / (BATCH_SIZE * div_du_dx_all_y.shape[-1] * 100 * 100)
div_ph_loss_x = torch.sum((div_du_dx_all_x + div_dv_dy_all_x) ** 2) / (BATCH_SIZE * div_du_dx_all_x.shape[-1] * 100 * 100)

pred_total_velocities_y_0, pred_total_velocities_x_0 = torch.unbind(pred_total_velocities[0], dim=-1)
pred_total_velocities_y_1, pred_total_velocities_x_1 = torch.unbind(pred_total_velocities[1], dim=-1)

bc_loss_y_0 = torch.sum(pred_total_velocities_y_0[0, 0, :]**2) + torch.sum(pred_total_velocities_y_0[0, -1, :]**2) \
              / pred_total_velocities_y_0.shape[2]
bc_loss_x_0 = torch.sum(pred_total_velocities_x_0[0, :, 0]**2) + torch.sum(pred_total_velocities_x_0[0, :, -1]**2) \
              / pred_total_velocities_x_0.shape[1]

bc_loss_y_1 = torch.sum(pred_total_velocities_y_1[0, 0, :]**2) + torch.sum(pred_total_velocities_y_1[0, -1, :]**2) \
              / pred_total_velocities_y_0.shape[2]
bc_loss_x_1 = torch.sum(pred_total_velocities_x_1[0, :, 0]**2) + torch.sum(pred_total_velocities_x_1[0, :, -1]**2) \
              / pred_total_velocities_x_0.shape[1]

bc_loss_after = (bc_loss_x_1 + bc_loss_y_1) / 2

print('Divergence Loss: {:.4f}, BC Loss: {:.4f}'.format(div_ph_loss_x.item(), bc_loss_after.item()))

print('Loss Vortex Net: {:.4f}'.format(loss_all[1].item()))
print('Loss Vortex Net + BCNet: {:.4f}'.format(losses_total_all[1].item()))

pred_velocities_y_1, pred_velocities_x_1 = torch.unbind(pred_velocites[1], dim=-1)

bc_loss_y_1_before = torch.sum(pred_velocities_y_1[0, 0, :]**2) + torch.sum(pred_velocities_y_1[0, -1, :]**2) \
              / pred_velocities_y_1.shape[2]
bc_loss_x_1_before = torch.sum(pred_velocities_x_1[0, :, 0]**2) + torch.sum(pred_velocities_x_1[0, :, -1]**2) \
              / pred_velocities_x_1.shape[1]

bc_loss_before = (bc_loss_x_1_before + bc_loss_y_1_before) / 2

print('BC Loss decreases from {:.4f} to {:.4f}'.format(bc_loss_before.item(), bc_loss_after.item()))

max_val = np.abs(velocities[0][0, :, :, 0]).max()
min_val = -max_val

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities[0][0, :, :-1, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'velocity_y_0.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities[1][0, :, :-1, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'velocity_y_1.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(pred_velocites[1].cpu().numpy()[0, :, :-1, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'pred_velocity_y_1.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(pred_total_velocities[1].cpu().numpy()[0, :, :-1, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'pred_total_velocity_y_1.pdf'), format='pdf',
            bbox_inches='tight')


max_val = np.abs(velocities[0][0, :, :, 1]).max()
min_val = -max_val

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities[0][0, :-1, :, 1], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'velocity_x_0.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities[1][0, :-1, :, 1], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'velocity_x_1.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(pred_velocites[1].cpu().numpy()[0, :-1, :, 1], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'pred_velocity_x_1.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(pred_total_velocities[1].cpu().numpy()[0, :-1, :, 1], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, 120])
ax.set_ylim([0, 120])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
fig.savefig(os.path.join(save_dir, 'pred_total_velocity_x_1.pdf'), format='pdf',
            bbox_inches='tight')


fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))#
ax.plot(pred_velocities_y_1.cpu().numpy()[0, 0, :])
ax.plot(pred_total_velocities_y_1.cpu().numpy()[0, 0, :])
ax.legend([r'before BC-Net', r'after BC-Net'])
fig.savefig(os.path.join(save_dir, 'top.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))#
ax.plot(pred_velocities_y_1.cpu().numpy()[0, -1, :])
ax.plot(pred_total_velocities_y_1.cpu().numpy()[0, -1, :])
ax.legend([r'before BC-Net', r'after BC-Net'])
fig.savefig(os.path.join(save_dir, 'bottom.pdf'), format='pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))#
ax.plot(pred_velocities_x_1.cpu().numpy()[0, :, 0])
ax.plot(pred_total_velocities_x_1.cpu().numpy()[0, :, 0])
ax.legend([r'before BC-Net', r'after BC-Net'])
fig.savefig(os.path.join(save_dir, 'left.pdf'), format='pdf',
            bbox_inches='tight')


fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))#
ax.plot(pred_velocities_x_1.cpu().numpy()[0, :, -1])
ax.plot(pred_total_velocities_x_1.cpu().numpy()[0, :, -1])
ax.legend([r'before BC-Net', r'after BC-Net'])
fig.savefig(os.path.join(save_dir, 'right.pdf'), format='pdf',
            bbox_inches='tight')

div_x_show_all_1 = (div_du_dx_all_y + div_dv_dy_all_y).view(1, 121, 120, NUM_TIME_STEPS + 1)[0, :, :, 1].cpu().numpy()**2
# div_x_show_all_2 = (div_du_dx_all_y + div_dv_dy_all_y).view(1, 101, 100, NUM_TIME_STEPS + 1)[0, :, :, 2].cpu().numpy()**2

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))#
pos = ax.imshow(div_x_show_all_1)
fig.colorbar(pos, ax=ax)
fig.savefig(os.path.join(save_dir, 'divergence.pdf'), format='pdf',
            bbox_inches='tight')

max_val = np.abs(velocities[0][0, :, :, 1]).max()
min_val = -max_val

plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(velocities[0][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('True Velocity: 0')
plt.subplot(1, 4, 2)
plt.imshow(velocities[1][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('True Velocity: 1')
plt.subplot(1, 4, 3)
plt.imshow(pred_velocites[1].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('VortexNet: 1, Loss: {:.4f}'.format(loss_all[1].item()))
plt.subplot(1, 4, 4)
plt.imshow(pred_total_velocities[1].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('VortexNet + BCNet: 1, Loss: {:.4f}'.format(losses_total_all[1].item()))
plt.suptitle('Velocity Componet - x (T1)')
plt.show()

plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(velocities[0][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('True Velocity: 0')
plt.subplot(1, 4, 2)
plt.imshow(velocities[1][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('True Velocity: 1')
plt.subplot(1, 4, 3)
plt.imshow(pred_velocites[1].cpu().numpy()[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('VortexNet: 1, Loss: {:.4f}'.format(loss_all[1].item()))
plt.subplot(1, 4, 4)
plt.imshow(pred_total_velocities[1].cpu().numpy()[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.title('VortexNet + BCNet: 1, Loss: {:.4f}'.format(losses_total_all[1].item()))
plt.suptitle('Velocity Componet - y (T1)')
plt.show()

# plt.figure()
# plt.subplot(1, 4, 1)
# plt.imshow(velocities[1][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('True Velocity: 0')
# plt.subplot(1, 4, 2)
# plt.imshow(velocities[2][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('True Velocity: 1')
# plt.subplot(1, 4, 3)
# plt.imshow(pred_velocites[2].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('VortexNet: 1, Loss: {:.4f}'.format(loss_all[2].item()))
# plt.subplot(1, 4, 4)
# plt.imshow(pred_total_velocities[2].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('VortexNet + BCNet: 1, Loss: {:.4f}'.format(losses_total_all[2].item()))
# plt.suptitle('Velocity Componet - x (T2)')
# plt.show()
#
# plt.figure()
# plt.subplot(1, 4, 1)
# plt.imshow(velocities[1][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('True Velocity: 1')
# plt.subplot(1, 4, 2)
# plt.imshow(velocities[2][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('True Velocity: 2')
# plt.subplot(1, 4, 3)
# plt.imshow(pred_velocites[2].cpu().numpy()[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('VortexNet: 1, Loss: {:.4f}'.format(loss_all[2].item()))
# plt.subplot(1, 4, 4)
# plt.imshow(pred_total_velocities[2].cpu().numpy()[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.title('VortexNet + BCNet: 1, Loss: {:.4f}'.format(losses_total_all[2].item()))
# plt.suptitle('Velocity Componet - y (T2)')
# plt.show()
#
#
# plt.figure()
# plt.imshow(div_x_show_all_1)
# plt.title('Divergence Map: Time 1, Max: {:.4f}'.format(div_x_show_all_1.max()))
# plt.show()
#
# plt.figure()
# plt.imshow(div_x_show_all_2)
# plt.title('Divergence Map: Time 2, Max: {:.4f}'.format(div_x_show_all_2.max()))
# plt.show()
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(velocities[1][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.subplot(1, 2, 2)
# plt.imshow(velocities_cg[1][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.show()
#

