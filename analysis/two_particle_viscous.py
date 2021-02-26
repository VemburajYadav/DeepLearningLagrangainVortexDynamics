import torch
import numpy as np
from phi.flow import Domain, Fluid, OPEN, math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from core.networks import *
from core.custom_functions import *

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/'
                                                     '/data/p10_gaussian_dataset_viscous_120x120_4000/train/sim_000321',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='p10_b_T1_exp_red_visc_weight_1.0_depth_5_100_batch_16_lr_1e-2_l2_1e-5_r100_4000_2',
                    help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=1, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

# location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
# strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
# sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']
# viscosity = np.load(os.path.join(case_dir, 'viscosity.npz'))['arr_0']

location_1 = np.array([40.0, 50.0])
location_2 = np.array([60.0, 50.0])
location = np.reshape(np.stack([location_1, location_2], axis=0), (1, 2, 2))

strength = np.reshape(np.array([-1.5, 1.5]), (1, 2, 1))
sigma = np.reshape(np.array([40.0, 40.0]), (1, 2, 1))
viscosity = np.arange(0, 0.5, 0.05)
ncases = viscosity.shape[0]

location = np.tile(location, (ncases, 1, 1))
strength = np.tile(strength, (ncases, 1, 1))
sigma = np.tile(sigma, (ncases, 1, 1))

nparticles = location.shape[1]

# velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
#               for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]
#
domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

# loc_x1 = int(location[0, 0, 1])
# loc_y1 = int(location[0, 0, 0])
#
# loc_x2 = int(location[0, 1, 1])
# loc_y2 = int(location[0, 1, 0])
#
# py1 = points_x[0, :, loc_x1, 0]
# px1 = np.array([loc_x1] * len(py1), dtype=np.float32)
#
# py2 = points_x[0, :, loc_x2, 0]
# px2 = np.array([loc_x2] * len(py2), dtype=np.float32)
#
# px1_ = points_y[0, loc_y1, :, 1]
# py1_ = np.array([loc_y1] * len(px1_), dtype=np.float32)
#
# px2_ = points_y[0, loc_y2, :, 1]
# py2_ = np.array([loc_y2] * len(px2_), dtype=np.float32)
#
#
# velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in range(NUM_TIME_STEPS + 1)]
#
loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')
visc_gpu = torch.tensor(viscosity, dtype=torch.float32, device='cuda:0')

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = torch.tensor(points_y, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(points_x, dtype=torch.float32, device='cuda:0')

# cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
# cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')
#
v0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
u0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
#
logs_dir = os.path.join('../logs', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

VortexNet = MultiStepViscousVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                          kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV, order=opt.order,
                                          num_steps=opt.num_time_steps, distinct_nets=opt.distinct_nets)
# VortexNet = InteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
#                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV)

VortexNet.single_step_net.load_state_dict(params)
if opt.num_time_steps > 1 and opt.distinct_nets:
    params2 = torch.load(ckpt_file)['model_state_dict2']
    VortexNet.single_step_net2.load_state_dict(params2)

VortexNet.to('cuda:0')
VortexNet = VortexNet.eval()

if opt.kernel == 'offset-gaussian':
    off0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    sigl0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), v0, u0, off0, sigl0], dim=-1)
    falloff_kernel = OffsetGaussianFalloffKernel()
if opt.kernel == 'ExpGaussian':
    c0 = torch.zeros((ncases, nparticles), dtype=torch.float32, device='cuda:0')
    d0 = torch.zeros((ncases, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), c0, d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernel(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
if opt.kernel == 'ExpGaussianRed':
    d0 = torch.zeros((ncases, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernelReduced(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
elif opt.kernel == 'gaussian':
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1)], dim=-1)
    falloff_kernel = GaussianFalloffKernel()

# pred_velocites = []
# losses= []
#
vortex_features = VortexNet(inp_feature.detach().clone(), visc_gpu.view(-1))

dy_1_p1 = (vortex_features[1][:, 0, 0] - vortex_features[0][:, 0, 0]).detach().clone().cpu().numpy()
dy_1_p2 = (vortex_features[1][:, 1, 0] - vortex_features[0][:, 1, 0]).detach().clone().cpu().numpy()

dx_1_p1 = (vortex_features[1][:, 0, 1] - vortex_features[0][:, 0, 1]).detach().clone().cpu().numpy()
dx_1_p2 = (vortex_features[1][:, 1, 1] - vortex_features[0][:, 1, 1]).detach().clone().cpu().numpy()

tau_1_p1 =  vortex_features[1][:, 0, 2].detach().clone().cpu().numpy()
tau_1_p2 =  vortex_features[1][:, 1, 2].detach().clone().cpu().numpy()

sig_1_p1 =  vortex_features[1][:, 0, 3].detach().clone().cpu().numpy()
sig_1_p2 =  vortex_features[1][:, 1, 3].detach().clone().cpu().numpy()

plt.figure(figsize=(16, 10))
plt.subplot(3, 1, 1)
plt.plot(viscosity, np.abs(dx_1_p1))
plt.legend(['Movement in x'])
plt.xlabel('viscosity')
plt.subplot(3, 1, 2)
plt.plot(viscosity, np.abs(tau_1_p1))
plt.legend(['Strength'])
plt.xlabel('viscosity')
plt.subplot(3, 1, 3)
plt.plot(viscosity, np.abs(sig_1_p1))
plt.legend(['Core Size'])
plt.xlabel('viscosity')
plt.suptitle('Plot of parameter variations for varying viscosities: Particle 1')
plt.show()

plt.figure(figsize=(16, 10))
plt.subplot(3, 1, 1)
plt.plot(viscosity, np.abs(dx_1_p2))
plt.legend(['Movement in x'])
plt.xlabel('viscosity')
plt.subplot(3, 1, 2)
plt.plot(viscosity, np.abs(tau_1_p2))
plt.legend(['Strength'])
plt.xlabel('viscosity')
plt.subplot(3, 1, 3)
plt.plot(viscosity, np.abs(sig_1_p2))
plt.legend(['Core Size'])
plt.xlabel('viscosity')
plt.suptitle('Plot of parameter variations for varying viscosities: Particle 2')
plt.show()
# with torch.no_grad():
#     for step in range(NUM_TIME_STEPS + 1):
#
#         vel_y = falloff_kernel(vortex_features[step], points_y)
#         vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
#         vel_x = falloff_kernel(vortex_features[step], points_x)
#         vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
#         vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
#         pred_velocites.append(vel.detach().clone())
#         losses.append(F.mse_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())
#
#     loss_all = torch.stack(losses, dim=-1)
#     features = torch.stack(vortex_features, dim=-1)
#
# dist = ((loc_x2 - loc_x1)**2 + (loc_y2 - loc_y1)**2)**0.5
#
# fig = plt.figure()
# plt.subplot(2,1,1)
# legend_list = []
# plt.xlim([0.0, 100.0])
# plt.plot(py1, velocities[0][0, :-1, loc_x1, 1])
# legend_list.append('True: {}'.format(0))
# for i in range(NUM_TIME_STEPS):
#     plt.plot(py1, velocities[i+1][0, :-1, loc_x1, 1])
#     legend_list.append('True: {}'.format(i+1))
#     plt.plot(py1, pred_velocites[i+1].cpu().numpy()[0, :-1, loc_x1, 1], '--')
#     legend_list.append('Pred: {}'.format(i+1))
# plt.axvline(x=loc_y2, color='r')
# plt.axvline(x=loc_y1, color='b')
# plt.legend(legend_list)
# plt.title('Particle 1 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}, Loss: {:.2f}'.format(dist, strength[0], sigma[0, 0, 0], loss_all.sum().item()))
#
# plt.subplot(2,1,2)
# legend_list = []
# plt.xlim([0.0, 100.0])
# plt.plot(py2, velocities[0][0, :-1, loc_x2, 1])
# legend_list.append('True: {}'.format(0))
# for i in range(NUM_TIME_STEPS):
#     plt.plot(py2, velocities[i+1][0, :-1, loc_x2, 1])
#     legend_list.append('True: {}'.format(i+1))
#     plt.plot(py2, pred_velocites[i+1].cpu().numpy()[0, :-1, loc_x2, 1], '--')
#     legend_list.append('Pred: {}'.format(i+1))
# plt.axvline(x=loc_y1, color='r')
# plt.axvline(x=loc_y2, color='b')
# plt.legend(legend_list)
# plt.title('Particle 2 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}, Loss: {:.2f}'.format(dist, strength[1], sigma[0, 1, 0], loss_all.sum().item()))
# fig.suptitle('Variation of velocity-x along y-axis')
# plt.show()
#
# plt.figure()
# plt.subplot(2,1,1)
# legend_list = []
# plt.xlim([0.0, 100.0])
# plt.plot(px1_, velocities[0][0, loc_y1, loc_x1-40:loc_x1+40, 0])
# legend_list.append('True: {}'.format(0))
# for i in range(NUM_TIME_STEPS):
#     plt.plot(px1_[loc_x1-40:loc_x1+40], velocities[i+1][0, loc_y1, loc_x1-40:loc_x1+40, 0])
#     legend_list.append('True: {}'.format(i+1))
#     plt.plot(px1_[loc_x1-40:loc_x1+40], pred_velocites[i+1].cpu().numpy()[0, loc_y1, loc_x1-40:loc_x1+40, 0], '--')
#     legend_list.append('Pred: {}'.format(i+1))
# plt.axvline(x=loc_x2, color='r')
# plt.axvline(x=loc_x1, color='b')
# plt.legend(legend_list)
# plt.title('Particle 1 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}, Loss: {:.2f}'.format(dist, strength[0], sigma[0, 0, 0], loss_all.sum().item()))
#
# plt.subplot(2,1,2)
# legend_list = []
# plt.xlim([80.0, 176.0])
# plt.plot(px2_[loc_x2-40:loc_x2+40], velocities[0][0, loc_y2, loc_x2-40:loc_x2+40, 0])
# legend_list.append('True: {}'.format(0))
# for i in range(NUM_TIME_STEPS):
#     plt.plot(px2_[loc_x2-40:loc_x2+40], velocities[i+1][0, loc_y2, loc_x2-40:loc_x2+40, 0])
#     legend_list.append('True: {}'.format(i+1))
#     plt.plot(px2_[loc_x2-40:loc_x2+40], pred_velocites[i+1].cpu().numpy()[0, loc_y2, loc_x2-40:loc_x2+40, 0], '--')
#     legend_list.append('Pred: {}'.format(i+1))
# plt.axvline(x=loc_x1, color='r')
# plt.axvline(x=loc_x2, color='b')
# plt.legend(legend_list)
# plt.title('Particle 2 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}, Loss: {:.2f}'.format(dist, strength[1], sigma[0, 1, 0], loss_all.sum().item()))
# fig.suptitle('Variation of velocity-y along x-axis')
# plt.show()
#
# max_val = np.abs(velocities[0][0, :, :, 1]).max()
# min_val = -max_val
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(velocities[0][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.subplot(1, 3, 2)
# plt.imshow(velocities[1][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.subplot(1, 3, 3)
# plt.imshow(pred_velocites[1].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.show()
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(velocities[0][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.subplot(1, 3, 2)
# plt.imshow(velocities[1][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.subplot(1, 3, 3)
# plt.imshow(pred_velocites[1].cpu().numpy()[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
# plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(24, 10))
#
# ax = axs[0]
# pcm = ax.imshow(velocities[0][0, :, :, 1], cmap='RdYlBu',vmin=min_val, vmax=max_val)
# ax.set_title('Time Step: 0')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax=cax)
#
# ax = axs[1]
# pcm = ax.imshow(velocities[1][0, :, :, 1], cmap='RdYlBu',vmin=min_val, vmax=max_val)
# ax.set_title('Time Step: 1')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax=cax)
#
# ax = axs[2]
# pcm = ax.imshow(velocities[2][0, :, :, 1], cmap='RdYlBu',vmin=min_val, vmax=max_val)
# ax.set_title('Time Step: 2')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax=cax)

# fig.suptitle(' Strength: {:.2f} \n Core Size:'
#              ' {:.2f}'.format(strength[0], sigma[0, 0, 0]))

# plt.show()


