import torch
import numpy as np
from phi.flow import *
import tensorflow as tf
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.animation as animation
from core.networks import *
from core.custom_functions import *
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2


parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--load_weights_ex', type=str, default='p2_r_T2_exp_red(6)_weight_1.0_depth_5_100_batch_32_lr_1e-3_l2_1e-4_r256_32000_2', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--num_time_steps', type=int, default=20, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# cuda.select_device(0)

# save_dir = os.path.join('../p2_samples/case_2')
#
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

RESOLUTION = opt.domain
NUM_TIME_STEPS = opt.num_time_steps

# location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
# strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
# sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

loc_1 = np.array([120.7, 145.2]).reshape((1, 2))

strength = np.array([1.8, -1.8])
sigma = np.array([20.0, 20.0])

dists = np.linspace(5.0, 30.0, 26).reshape(-1, 1)

angle = 45.0

loc_2 = np.hstack([loc_1[0, 0] + dists * np.sin(angle * np.pi / 180.0), loc_1[0, 1] + dists * np.cos(angle * np.pi / 180.0)])
loc_1 = np.tile(loc_1, (loc_2.shape[0], 1))

location = np.concatenate([loc_1.reshape((-1, 1, 2)), loc_2.reshape((-1, 1, 2))], axis=-2)

strength = np.reshape(np.tile(strength.reshape(-1, 2), (loc_2.shape[0], 1)), (-1, 2, 1))
sigma = np.reshape(np.tile(sigma.reshape(-1, 2), (loc_2.shape[0], 1)), (-1, 2, 1))

nparticles = location.shape[1]

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = torch.tensor(points_y, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(points_x, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

v0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
u0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')

logs_dir = os.path.join('../logs_1', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV, order=opt.order,
                                        num_steps=opt.num_time_steps, distinct_nets=opt.distinct_nets)
# VortexNet = InteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
#                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV)
#
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
    c0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0')
    d0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), c0, d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernel(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
if opt.kernel == 'ExpGaussianRed':
    d0 = torch.zeros((loc_2.shape[0], nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernelReduced(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
elif opt.kernel == 'gaussian':
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1)], dim=-1)
    falloff_kernel = GaussianFalloffKernel()

pred_velocites = []
losses= []


def get_deriv_features(inp):
    y, x, tau, sig, d = torch.unbind(inp, dim=-1)

    inp_clone = inp.detach().clone()

    nparticles = y.shape[1]

    location = torch.stack([y, x], dim=-1)
    location_clone = location.detach().clone()

    feature_list = []
    paxes = np.arange(nparticles)
    for i in range(nparticles):
        paxes_tensor = torch.tensor([i], device='cuda:0')
        p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
        py, px = torch.unbind(p_loc, dim=-1)
        py.requires_grad_(True)
        px.requires_grad_(True)
        p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
        other_p_axes = np.delete(paxes, i)
        other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
        other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
        vel_by_other_ps = falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
        vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

        dv_dy = \
        torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
        dv_dx = \
        torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
        du_dy = \
        torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
        du_dx = \
        torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

        d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
        d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
        d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
        d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
        d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
        d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

        feature_list.append(torch.stack([vel_y.detach().clone(), vel_x.detach().clone(),
                            dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
                            du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0,
                            d2v_dy2.detach().clone() * 100.0, d2v_dx2.detach().clone() * 100.0,
                            d2v_dxdy.detach().clone() * 100.0,
                            d2u_dy2.detach().clone() * 100.0, d2u_dx2.detach().clone() * 100.0,
                            d2u_dydx.detach().clone() * 100.0], dim=-1))

    deriv_features = torch.stack(feature_list, dim=-2)

    return deriv_features

vortex_features = VortexNet(inp_feature.detach().clone())
vortex_features = [f.detach().clone() for f in vortex_features]

deriv_feature_list = []
for step in range(NUM_TIME_STEPS + 1):
    deriv_feature_list.append(get_deriv_features(vortex_features[step]))

dy1 = vortex_features[1][:, 0, 0] - vortex_features[0][:, 0, 0]
dx1 = vortex_features[1][:, 0, 1] - vortex_features[0][:, 0, 1]

dy2 = vortex_features[1][:, 1, 0] - vortex_features[0][:, 1, 0]
dx2 = vortex_features[1][:, 1, 1] - vortex_features[0][:, 1, 1]

fig, ax = plt.subplots(2, 4, sharex=True)
ax[0, 0].plot(dists, dy1.cpu().numpy())
ax[0, 0].set_title('y-displacement of particle 1')
ax[0, 0].set_ylabel('dy')

ax[0, 1].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 0])
ax[0, 1].set_title('y-velocity influence at particle 1')
ax[0, 1].set_ylabel('v')

ax[0, 2].plot(dists, dx1.cpu().numpy())
ax[0, 2].set_title('x-displacement of particle 1')
ax[0, 2].set_ylabel('dx')

ax[0, 3].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 1])
ax[0, 3].set_title('x-velocity influence at particle 1')
ax[0, 3].set_ylabel('u')

ax[1, 0].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 2])
ax[1, 0].set_title('gradient influence at particle 1 (dv / dy)')
ax[1, 0].set_ylabel('dv/dy')

ax[1, 1].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 3])
ax[1, 1].set_title('gradient influence at particle 1 (dv / dx)')
ax[1, 1].set_ylabel('dv/dx')

ax[1, 2].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 4])
ax[1, 2].set_title('gradient influence at particle 1 (du / dy)')
ax[1, 2].set_ylabel('du/dy')

ax[1, 3].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 5])
ax[1, 3].set_title('gradient influence at particle 1 (du / dx)')
ax[1, 3].set_ylabel('du/dx')

plt.show()

fig, ax = plt.subplots(2, 3, sharex=True)

ax[0, 0].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 6])
ax[0, 0].set_title(r"Hessian influence at Particle 1 $(\frac{d^2v}{dy^2})$")
ax[0, 0].set_ylabel(r"$\frac{d^2v}{dy^2}$")

ax[0, 1].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 7])
ax[0, 1].set_title(r"Hessian influence at Particle 1 $(\frac{d^2v}{dx^2})$")
ax[0, 1].set_ylabel(r"$\frac{d^2v}{dx^2}$")

ax[0, 2].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 8])
ax[0, 2].set_title(r"Hessian influence at Particle 1 $(\frac{d^2v}{dxdy})$")
ax[0, 2].set_ylabel(r"$\frac{d^2v}{dxdy}$")

ax[1, 0].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 9])
ax[1, 0].set_title(r"Hessian influence at Particle 1 $(\frac{d^2u}{dy^2})$")
ax[1, 0].set_ylabel(r"$\frac{d^2u}{dy^2}$")

ax[1, 1].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 10])
ax[1, 1].set_title(r"Hessian influence at Particle 1 $(\frac{d^2u}{dx^2})$")
ax[1, 1].set_ylabel(r"$\frac{d^2u}{dx^2}$")

ax[1, 2].plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 11])
ax[1, 2].set_title(r"Hessian influence at Particle 1 $(\frac{d^2u}{dxdy})$")
ax[1, 2].set_ylabel(r"$\frac{d^2u}{dxdy}$")

plt.show()
# plt.figure()
# plt.plot(dists, dx1.cpu().numpy())
# plt.show()
#
# plt.figure()
# plt.plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 0])
# plt.show()
#
# plt.figure()
# plt.plot(dists, deriv_feature_list[0].cpu().numpy()[:, 0, 1])
# plt.show()
#
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

#     loss_all = torch.stack(losses, dim=-1)
#     features = torch.stack(vortex_features, dim=-1)
#
# max_val = np.abs(velocities[0][0, :, :, 1]).max()
# min_val = -max_val
#
# total_velocities_pred = []
# total_velocities = []
# error_map = []
# for step in range(NUM_TIME_STEPS + 1):
#     total_velocities_pred.append(torch.sqrt(torch.sum(pred_velocites[step]**2, dim=-1)))
#     total_velocities.append(torch.sqrt(torch.sum(velocities_gpu[step]**2, dim=-1)))
#     error_map.append(torch.abs((total_velocities[step] - total_velocities_pred[step])))