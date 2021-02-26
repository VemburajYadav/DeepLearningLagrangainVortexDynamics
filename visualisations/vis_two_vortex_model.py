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
from visualisations.my_plot import set_size
from phi.flow import *

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/Desktop/TwoParticle/Case4',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='T1_p10_exp_weight_1.0_depth_5_100_batch_32_lr_1e-3_l2_1e-5_r120_4000_2',
                    help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=40, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='GaussianVorticity', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

width = 455.24408
plt.style.use('tex')

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

nparticles = location.shape[1]

velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

loc_x1 = int(location[0, 0, 1])
loc_y1 = int(location[0, 0, 0])

loc_x2 = int(location[0, 1, 1])
loc_y2 = int(location[0, 1, 0])

py1 = points_x[0, :, loc_x1, 0]
px1 = np.array([loc_x1] * len(py1), dtype=np.float32)

py2 = points_x[0, :, loc_x2, 0]
px2 = np.array([loc_x2] * len(py2), dtype=np.float32)

px1_ = points_y[0, loc_y1, :, 1]
py1_ = np.array([loc_y1] * len(px1_), dtype=np.float32)

px2_ = points_y[0, loc_y2, :, 1]
py2_ = np.array([loc_y2] * len(px2_), dtype=np.float32)

device = torch.device('cuda:0')
velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device=device) for i in range(NUM_TIME_STEPS + 1)]

loc_gpu = torch.tensor(location, dtype=torch.float32, device=device)
tau_gpu = torch.tensor(strength, dtype=torch.float32, device=device)
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device=device)

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = torch.tensor(points_y, dtype=torch.float32, device=device)
points_x = torch.tensor(points_x, dtype=torch.float32, device=device)

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device=device)
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device=device)

v0 = torch.zeros((1, 1), dtype=torch.float32, device=device)
u0 = torch.zeros((1, 1), dtype=torch.float32, device=device)

logs_dir = os.path.join('../logs_p10_gauss_inviscid', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV, order=opt.order,
                                        num_steps=opt.num_time_steps, distinct_nets=opt.distinct_nets)
# VortexNet = MultiStepInteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
#                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV)

VortexNet.single_step_net.load_state_dict(params)
if opt.num_time_steps > 1 and opt.distinct_nets:
    params2 = torch.load(ckpt_file)['model_state_dict2']
    VortexNet.single_step_net2.load_state_dict(params2)

VortexNet.to(device)
VortexNet = VortexNet.eval()

if opt.kernel == 'offset-gaussian':
    off0 = torch.zeros((1, 1), dtype=torch.float32, device=device)
    sigl0 = torch.zeros((1, 1), dtype=torch.float32, device=device)
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), v0, u0, off0, sigl0], dim=-1)
    falloff_kernel = OffsetGaussianFalloffKernel()
if opt.kernel == 'ExpGaussian':
    c0 = torch.zeros((1, nparticles), dtype=torch.float32, device=device)
    d0 = torch.zeros((1, nparticles), dtype=torch.float32, device=device) + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), c0, d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernel(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
if opt.kernel == 'ExpGaussianRed':
    d0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernelReduced(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
elif opt.kernel == 'gaussian':
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1)], dim=-1)
    falloff_kernel = GaussianFalloffKernel()
elif opt.kernel == 'GaussianVorticity':
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles)], dim=-1)
    falloff_kernel = GaussianFalloffKernelVelocity()

pred_velocites = []
losses= []

vortex_features = VortexNet(inp_feature.detach().clone())

with torch.no_grad():
    for step in range(NUM_TIME_STEPS + 1):

        vel_y = falloff_kernel(vortex_features[step], points_y)
        vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
        vel_x = falloff_kernel(vortex_features[step], points_x)
        vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
        vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
        pred_velocites.append(vel.detach().clone())
        losses.append(F.mse_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())

    loss_all = torch.stack(losses, dim=-1)
    features = torch.stack(vortex_features, dim=-1)

np.savez_compressed(os.path.join(case_dir, 'nn_features.npz'), features.cpu().numpy())


velocity_filenames = ['pred_velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
for i in range(NUM_TIME_STEPS + 1):
    np.savez_compressed(os.path.join(case_dir, velocity_filenames[i]),
                        StaggeredGrid(pred_velocites[i].cpu().numpy()).staggered_tensor())

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 0, 2, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\Gamma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'strength_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 0, 3, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\sigma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'sigma_p0_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 0, 0, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$y_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'yloc_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 0, 1, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$x_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'xloc_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 1, 2, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\Gamma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'strength_p1_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 1, 3, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\sigma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'sigma_p1_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 1, 0, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$y_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'yloc_p1_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(features.cpu().numpy()[0, 1, 1, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$x_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'xloc_p1_N2_fit.pdf'), format='pdf')
