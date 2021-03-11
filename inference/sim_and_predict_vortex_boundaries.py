import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from core.custom_functions import *
import argparse
import glob
from functools import partial
from phi.flow import *
from core.networks import *
from core.velocity_derivs import *
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--location', type=str, default='../sample/location_000000.npz',
                    help='path to the npz file with initial particle locations')
parser.add_argument('--strength', type=str, default='../sample/strength_000000.npz',
                    help='path to the npz file with initial particle strengths')
parser.add_argument('--core_size', type=str, default='../sample/sigma_000000.npz',
                    help='path to the npz file with initial particle core sizes')
parser.add_argument('--sim', type=bool, default=True,
                    help='whether to run numerical simulations')
parser.add_argument('--sim_time_step', type=float, default=0.2,
                    help='time step in seconds for running numerical simulations (only applicable if argument: "sim" is True)')
parser.add_argument('--network_time_step', type=float, default=1.0,
                    help='time step in seconds over which the neural network is trained to make predictions')
parser.add_argument('--num_time_steps', type=int, default=5, help='number of time steps to make predictions')
parser.add_argument('--network', type=str, default='Vortex',
                    help='type of neural network for VortexNet: Vortex or Interaction')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--logs_dir_vortex', type=str,
                    default=None,
                    help='directory with checkpoints and training summaries for VortexNet')
parser.add_argument('--load_weights_ex_vortex', type=str,
                    default=None,
                    help='name of the experiment to load checkpoint from for VortexNet')
parser.add_argument('--logs_dir_bc', type=str,
                    default=None,
                    help='directory with checkpoints and training summaries for BCNet')
parser.add_argument('--load_weights_ex_bc', type=str,
                    default=None,
                    help='name of the experiment to load checkpoint from for BCNet')
parser.add_argument('--ckpt_path_vortex', type=str, default='../model/ckpt_vortexnet_2_inviscid.pytorch',
                    help='path to the actual checkpoint file for VortexNet '
                         '(overrides the logs_dir_vortex and load_weights_ex_vortex argument)')
parser.add_argument('--ckpt_path_bc', type=str, default='../model/ckpt_bcnet_2.pytorch',
                    help='path to the actual checkpoint file for BCNet '
                         '(overrides the logs_dir_vortex and load_weights_ex_bc argument)')
parser.add_argument('--save_dir', type=str, default='../Visuals/VortexNet_2_BCNet/Case_1',
                    help='directory to write the neural network predictions and plots (leave it to the default value of None if not to save the outputs)')



# Parse input arguments
opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
RESOLUTION = opt.domain
NETWORK = opt.network
ORDER = opt.order
save_dir = opt.save_dir


# filename's and directories for saving outputs
velocity_filenames_vortex = ['velocity_prediction_vortex_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
velocity_filenames_sim = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
velocity_filenames_vortex_bc = ['velocity_prediction_vortex_bc_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
vortex_features_filenames_pred = ['vortex_features_prediction_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
error_vel_vortex_filenames = ['error_vel_mag_vortex_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
error_vel_vortex_bc_filenames = ['error_vel_mag_vortex_bc_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]


out_dir = os.path.join(save_dir, 'outputs')
vis_dir = os.path.join(save_dir, 'plots')


# Read location, strength and core size
location = np.reshape(np.load(os.path.join(opt.location))['arr_0'], (1, -1, 2))
strength = np.reshape(np.load(os.path.join(opt.strength))['arr_0'], (1, -1, 1))
sigma = np.reshape(np.load(os.path.join(opt.core_size))['arr_0'], (1, -1, 1))


# define domain and resolution of the grid
domain = Domain(resolution=opt.domain, boundaries=CLOSED)
FLOW_REF = Fluid(domain=domain)

# points in the staggered grid
points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

points_y_res = points_y.view(1, -1, 2)
points_x_res = points_x.view(1, -1, 2)

# points in th centered grid
points_cg = torch.tensor(FLOW_REF.density.points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

# Gaussian falloff-kernel for velocity
falloff_kernel = GaussianFalloffKernelVelocity()
# Gaussian falloff-kernel for vorticity
falloff_kernel_vorticity = GaussianFalloffKernelVorticity()


# Module to compute the velocities and derivatives of velocities due to vortex particles
VelDerivExpRed = VelocityDerivatives(order=ORDER)


## Execute numerical simulations (if applicable)
if opt.sim:
    SIM_TIME_STEP = opt.sim_time_step
    NN_TIME_STEP = opt.network_time_step
    STRIDE = int(NN_TIME_STEP / SIM_TIME_STEP)

    # Gaussian falloff kernel
    def gaussian_falloff(distance, sigma):
        sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
        falloff = (1.0 - math.exp(- sq_distance / sigma ** 2)) / (2.0 * np.pi * sq_distance)
        return falloff

    # vorticity
    vorticity = AngularVelocity(location=location,
                                strength=np.reshape(strength, (-1,)),
                                falloff=partial(gaussian_falloff, sigma=sigma))

    # velocity at grid points
    velocity_0 = vorticity.at(FLOW_REF.velocity)

    # velocity field after the pressure solve step
    velocity_0_div_free = divergence_free(velocity_0, domain=domain)

    velocities_ = [velocity_0_div_free]

    # PhiFlow physics object
    FLOW = Fluid(domain=domain, velocity=velocity_0_div_free)
    fluid = world.add(Fluid(domain=domain, velocity=velocity_0_div_free),
                      physics=IncompressibleFlow())

    # time advancement
    for step in range(NUM_TIME_STEPS * STRIDE):
        world.step(dt=SIM_TIME_STEP)

        if step % STRIDE == (STRIDE - 1):
            velocities_.append(fluid.velocity)

    velocities = []

    for i in range(NUM_TIME_STEPS + 1):
        vx = np.concatenate([velocities_[i].x.data, np.zeros((1, 1, RESOLUTION[1] + 1, 1))], axis=-3)
        vy = np.concatenate([velocities_[i].y.data, np.zeros((1, RESOLUTION[0] + 1, 1, 1))], axis=-2)
        velocities.append(np.concatenate([vy, vx], axis=-1))




# Neural network for Vortex Particle Dynamics
if NETWORK == 'Vortex':
    VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units,
                                            order=opt.order, num_steps=opt.num_time_steps)
else:
    VortexNet = MultiStepInteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, num_steps=opt.num_time_steps)


# Neural network for predicting correction fields in presence of boundaries
BCNet = BoundaryConditionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, order=opt.order)


# Load weights directly from the path to the checkpoint file or
# the best checkpoint from the experiment in the logs directory
if opt.ckpt_path_vortex is None:
    init_weights_log_dir_vortex = os.path.join(opt.logs_dir_vortex, opt.load_weights_ex_vortex)
    init_weights_ckpt_dir_vortex = os.path.join(init_weights_log_dir_vortex, 'ckpt')

    checkpoints_files_vortex = os.listdir(os.path.join(init_weights_ckpt_dir_vortex))
    epoch_id_vortex = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files_vortex]))
    init_weights_ckpt_file_vortex = os.path.join(init_weights_ckpt_dir_vortex, checkpoints_files_vortex[epoch_id_vortex])
else:
    init_weights_ckpt_file_vortex = os.path.join(opt.ckpt_path_vortex)

params_vortex = torch.load(init_weights_ckpt_file_vortex)['model_state_dict']
VortexNet.single_step_net.load_state_dict(params_vortex)


if opt.ckpt_path_bc is None:
    init_weights_log_dir_bc = os.path.join(opt.logs_dir_bc, opt.load_weights_ex_bc)
    init_weights_ckpt_dir_bc = os.path.join(init_weights_log_dir_bc, 'ckpt')

    checkpoints_files_bc = os.listdir(os.path.join(init_weights_ckpt_dir_bc))
    epoch_id_bc = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files_bc]))
    init_weights_ckpt_file_bc = os.path.join(init_weights_ckpt_dir_bc, checkpoints_files_bc[epoch_id_bc])
else:
    init_weights_ckpt_file_bc = os.path.join(opt.ckpt_path_bc)

params_bc = torch.load(init_weights_ckpt_file_bc)['model_state_dict']
BCNet.load_state_dict(params_bc)


# Neural network to gpu
VortexNet.to('cuda:0')
VortexNet.eval()

BCNet.to('cuda:0')
BCNet.eval()


# tensors to gpu
loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

nparticles = location.shape[1]

py, px = torch.unbind(loc_gpu, dim=-1)
inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles)], dim=-1)

# predictions from neural network
vortex_features = VortexNet(inp_feature.detach().clone())

pred_velocities_vortex = []

# compute mse and mae losses on the velocity fields
with torch.no_grad():
    for step in range(NUM_TIME_STEPS + 1):
        vel_y = falloff_kernel(vortex_features[step], points_y)
        vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
        vel_x = falloff_kernel(vortex_features[step], points_x)
        vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
        vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
        pred_velocities_vortex.append(vel.detach().clone())

    features = torch.stack(vortex_features, dim=-1).detach().clone().cpu().numpy()


pred_velocities_vortex_bc = []
div_loss_mse = []
bc_loss_mse_vortex = []
bc_loss_mse_vortex_bc = []


# Vortex Network  --->   Boundary condition network
for step in range(NUM_TIME_STEPS + 1):

    time_index = torch.tensor([step], device='cuda:0')
    features_time = torch.index_select(torch.tensor(features, dtype=torch.float32, device='cuda:0'),
                                       dim=-1, index=time_index).view(1, -1, 4)

    vel_deriv_points_y = VelDerivExpRed(features_time, points_y_res)
    vel_deriv_points_x = VelDerivExpRed(features_time, points_x_res)

    points_y_y, points_y_x = torch.unbind(points_y_res.view(-1, 2), dim=-1)
    points_x_y, points_x_x = torch.unbind(points_x_res.view(-1, 2), dim=-1)

    points_y_y.requires_grad_(True)
    points_y_x.requires_grad_(True)

    points_y_loc = torch.stack([points_y_y.view(1, -1), points_y_x.view(1, -1)], dim=-1)

    # neural network input for staggered points corresponding to y-component of velocity
    points_y_features = torch.cat([points_y_loc, vel_deriv_points_y], dim=-1)

    # correction velocity predictions for staggered points corresponding to y-component of velocity
    points_y_corr_vel = BCNet(points_y_features)

    points_y_corr_vel_y, points_y_corr_vel_x = torch.unbind(points_y_corr_vel, dim=-1)
    points_y_corr_vel_y_re = points_y_corr_vel_y.view(-1)
    points_y_corr_vel_x_re = points_y_corr_vel_x.view(-1)

    # compute divergence for staggered points corresponding to y-component of velocity
    div_du_dx_points_y = torch.autograd.grad(torch.unbind(points_y_corr_vel_x_re, dim=-1), points_y_x, create_graph=True,
                                             allow_unused=True)[0]
    div_dv_dy_points_y = torch.autograd.grad(torch.unbind(points_y_corr_vel_y_re, dim=-1), points_y_y, create_graph=True,
                                             allow_unused=True)[0]

    div_du_dx_points_y = div_du_dx_points_y.view(1, -1).detach().clone()
    div_dv_dy_points_y = div_dv_dy_points_y.view(1, -1).detach().clone()

    # divergence loss for staggered points corresponding to y-component of velocity
    total_div_points_y = div_du_dx_points_y + div_dv_dy_points_y
    div_loss_points_y = torch.sum(total_div_points_y ** 2) / total_div_points_y.nelement()

    points_x_y.requires_grad_(True)
    points_x_x.requires_grad_(True)

    points_x_loc = torch.stack([points_x_y.view(1, -1), points_x_x.view(1, -1)], dim=-1)

    # neural network input for staggered points corresponding to x-component of velocity
    points_x_features = torch.cat([points_x_loc, vel_deriv_points_x], dim=-1)

    # correction velocity predictions for staggered points corresponding to x-component of velocity
    points_x_corr_vel = BCNet(points_x_features)

    points_x_corr_vel_y, points_x_corr_vel_x = torch.unbind(points_x_corr_vel, dim=-1)
    points_x_corr_vel_y_re = points_x_corr_vel_y.view(-1)
    points_x_corr_vel_x_re = points_x_corr_vel_x.view(-1)

    # compute divergence for staggered points corresponding to x-component of velocity
    div_du_dx_points_x = torch.autograd.grad(torch.unbind(points_x_corr_vel_x_re, dim=-1), points_x_x, create_graph=True,
                                             allow_unused=True)[0]
    div_dv_dy_points_x = torch.autograd.grad(torch.unbind(points_x_corr_vel_y_re, dim=-1), points_x_y, create_graph=True,
                                             allow_unused=True)[0]

    div_du_dx_points_x = div_du_dx_points_x.view(1, -1).detach().clone()
    div_dv_dy_points_x = div_dv_dy_points_x.view(1, -1).detach().clone()

    # divergence loss for staggered points corresponding to x-component of velocity
    total_div_points_x = div_du_dx_points_x + div_dv_dy_points_x
    div_loss_points_x = torch.sum(total_div_points_x ** 2) / total_div_points_x.nelement()

    # divergence loss over all the points in the staggered grid
    div_loss_case = (div_loss_points_y + div_loss_points_x) / 2
    div_loss_mse.append(div_loss_case)

    # correction velocity tensor
    corr_vel_tensor = \
            torch.stack([torch.cat([points_y_corr_vel_y.view(1, RESOLUTION[0] + 1, -1), cat_y], dim=-1),
                         torch.cat([points_x_corr_vel_x.view(1, -1, RESOLUTION[0] + 1), cat_x], dim=-2)], dim=-1).detach().clone()

    # total velocity tensor:  velocity due to vortex particles + correction velocity tensor
    vel_vortex_bc = pred_velocities_vortex[step] + corr_vel_tensor
    pred_velocities_vortex_bc.append(vel_vortex_bc.detach().clone())


    # compute boundary condition loss before and after predictions from BCNet
    bc_loss_before = \
            torch.sum((pred_velocities_vortex[step][0, :, 0, 1]**2 + pred_velocities_vortex[step][0, :, -1, 1]**2 +
                       pred_velocities_vortex[step][0, 0, :, 0]**2 + pred_velocities_vortex[step][0, -1, :, 0]**2))

    bc_loss_after = \
            torch.sum((pred_velocities_vortex_bc[step][0, :, 0, 1]**2 + pred_velocities_vortex_bc[step][0, :, -1, 1]**2 +
                       pred_velocities_vortex_bc[step][0, 0, :, 0]**2 + pred_velocities_vortex_bc[step][0, -1, :, 0]**2))

    bc_loss_mse_vortex.append(bc_loss_before)
    bc_loss_mse_vortex_bc.append(bc_loss_after)


div_loss_all = torch.stack(div_loss_mse, dim=-1)
bc_loss_before_all = torch.stack(bc_loss_mse_vortex, dim=-1)
bc_loss_after_all = torch.stack(bc_loss_mse_vortex_bc, dim=-1)



# compute losses (if applicable)
if opt.sim:
    velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in
                                   range(NUM_TIME_STEPS + 1)]

    mse_losses_vortex = []
    l1_losses_vortex = []
    error_vel_mag_vortex = []
    mse_losses_vortex_bc = []
    l1_losses_vortex_bc = []
    error_vel_mag_vortex_bc = []


    for step in range(NUM_TIME_STEPS + 1):
        # compute mse and mae losses on the velocity fields as a result of predictions from VortexNet
        mse_losses_vortex.append(F.mse_loss(pred_velocities_vortex[step], velocities_gpu[step], reduction='sum').detach().clone())
        l1_losses_vortex.append(F.l1_loss(pred_velocities_vortex[step], velocities_gpu[step], reduction='sum').detach().clone())
        error_vel_mag_vortex.append(torch.sqrt(torch.sum((pred_velocities_vortex[step] - velocities_gpu[step])**2,
                                                         dim=-1, keepdim=True)).detach().clone().cpu().numpy())

        # compute mse and mae losses on the velocity fields as a result of predictions from VortexNet + BCNet
        mse_losses_vortex_bc.append(F.mse_loss(pred_velocities_vortex_bc[step], velocities_gpu[step], reduction='sum'))
        l1_losses_vortex_bc.append(F.l1_loss(pred_velocities_vortex_bc[step], velocities_gpu[step], reduction='sum'))
        error_vel_mag_vortex_bc.append(torch.sqrt(torch.sum((pred_velocities_vortex_bc[step] - velocities_gpu[step])**2,
                                                            dim=-1, keepdim=True)).detach().clone().cpu().numpy())

    loss_all_mse_vortex = torch.stack(mse_losses_vortex, dim=-1).detach().clone().cpu().numpy()
    loss_all_l1_vortex = torch.stack(l1_losses_vortex, dim=-1).detach().clone().cpu().numpy()
    loss_all_mse_vortex_bc = torch.stack(mse_losses_vortex_bc, dim=-1).detach().clone().cpu().numpy()
    loss_all_l1_vortex_bc = torch.stack(l1_losses_vortex_bc, dim=-1).detach().clone().cpu().numpy()



# save outputs from simulation, Vortex-Fit adn Neural networks (if applicable)
if opt.save_dir:

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for i in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(out_dir, velocity_filenames_vortex[i]), pred_velocities_vortex[i].cpu().numpy())
        np.savez_compressed(os.path.join(out_dir, velocity_filenames_vortex_bc[i]), pred_velocities_vortex_bc[i].cpu().numpy())
        np.savez_compressed(os.path.join(out_dir, vortex_features_filenames_pred[i]), features[:, :, :, i])

        if opt.sim:
            np.savez_compressed(os.path.join(out_dir, velocity_filenames_sim[i]), velocities[i])
            np.savez_compressed(os.path.join(out_dir, error_vel_vortex_filenames[i]), error_vel_mag_vortex[i])
            np.savez_compressed(os.path.join(out_dir, error_vel_vortex_bc_filenames[i]), error_vel_mag_vortex_bc[i])



# save plots and images (if applicable)
if opt.save_dir:

    if not os.path.isdir(vis_dir):
        os.makedirs(vis_dir)

    max_vel_x = np.abs(velocities[0][0, :, :, 1]).max()
    max_vel_y = np.abs(velocities[0][0, :, :, 0]).max()
    max_vel_mag = np.sqrt(velocities[0][0, :, :, 1]**2 + velocities[0][0, :, :, 0]**2).max()

    min_vel_x = -max_vel_x
    min_vel_y = -max_vel_y

    for step in range(NUM_TIME_STEPS + 1):

        # Plots for VortexNet
        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(pred_velocities_vortex[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_x_prediction_vortex_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')

        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(pred_velocities_vortex[step][0, :, :, 0].cpu().numpy(), vmin=min_vel_y, vmax=max_vel_y, cmap='coolwarm')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_y_prediction_vortex_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')

        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(np.sqrt(pred_velocities_vortex[step][0, :, :, 1].cpu().numpy()**2 +
                                pred_velocities_vortex[step][0, :, :, 0].cpu().numpy()**2),
                        vmin=0, vmax=max_vel_mag, cmap='viridis')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_mag_prediction_vortex_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')


        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(error_vel_mag_vortex[step][0, :, :, 0], cmap='Greys')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'error_vel_mag_vortex_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')


        # Plots for VortexNet + BCNet
        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(pred_velocities_vortex_bc[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_x_prediction_vortex_bc_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')

        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(pred_velocities_vortex_bc[step][0, :, :, 0].cpu().numpy(), vmin=min_vel_y, vmax=max_vel_y, cmap='coolwarm')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_y_prediction_vortex_bc_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')

        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(np.sqrt(pred_velocities_vortex_bc[step][0, :, :, 1].cpu().numpy()**2 +
                                pred_velocities_vortex_bc[step][0, :, :, 0].cpu().numpy()**2),
                        vmin=0, vmax=max_vel_mag, cmap='viridis')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_mag_prediction_vortex_bc_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')


        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(error_vel_mag_vortex_bc[step][0, :, :, 0], cmap='Greys')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'error_vel_mag_vortex_bc_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')


        if opt.sim:
            fig, ax = plt.subplots(1, 1)
            pos = ax.imshow(velocities[step][0, :, :, 1], vmin=min_vel_x, vmax=max_vel_x,
                            cmap='coolwarm')
            ax.set_xlim([0, RESOLUTION[1]])
            ax.set_ylim([0, RESOLUTION[0]])
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$', rotation=0)
            fig.colorbar(pos, ax=ax)
            fig.savefig(
                os.path.join(vis_dir, 'velocity_x_' + '0' * (6 - len(str(step))) + str(step) + '.png'),
                format='png',
                bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            pos = ax.imshow(velocities[step][0, :, :, 0], vmin=min_vel_y, vmax=max_vel_y,
                            cmap='coolwarm')
            ax.set_xlim([0, RESOLUTION[1]])
            ax.set_ylim([0, RESOLUTION[0]])
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$', rotation=0)
            fig.colorbar(pos, ax=ax)
            fig.savefig(
                os.path.join(vis_dir, 'velocity_y_' + '0' * (6 - len(str(step))) + str(step) + '.png'),
                format='png',
                bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            pos = ax.imshow(np.sqrt(velocities[step][0, :, :, 1]**2 +
                                    velocities[step][0, :, :, 0]**2),
                            vmin=0, vmax=max_vel_mag, cmap='viridis')
            ax.set_xlim([0, RESOLUTION[1]])
            ax.set_ylim([0, RESOLUTION[0]])
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$', rotation=0)
            fig.colorbar(pos, ax=ax)
            fig.savefig(
                os.path.join(vis_dir, 'velocity_mag_' + '0' * (6 - len(str(step))) + str(step) + '.png'),
                format='png',
                bbox_inches='tight')



    ## Make videos
    video_dir = os.path.join(vis_dir, 'video_temp')

    if not os.path.isdir(video_dir):
        os.makedirs(video_dir)

    # make video of neural network predictions
    for step in range(NUM_TIME_STEPS + 1):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        pcm = ax.imshow(pred_velocities_vortex_bc[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
        ax.set_title(r'VortexNet + BCNet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pcm, cax=cax)

        fig.suptitle(r'Time: {}s'.format(step))

        filename = os.path.join(video_dir, 'vis_' + '0' * (6 - len(str(step))) + str(step) + '.png')
        plt.savefig(filename)


    video_name = os.path.join(vis_dir, 'video_nn.avi')

    images = [img for img in sorted(os.listdir(video_dir)) if img.endswith(".png") and img.startswith("vis")]
    frame = cv2.imread(os.path.join(video_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(video_dir, image)))

    video.release()

    for step in range(NUM_TIME_STEPS + 1):
        os.remove(os.path.join(video_dir, images[step]))


    if opt.sim:

        # make video of neural network predictions and simulations in comparison
        for step in range(NUM_TIME_STEPS + 1):
            fig, axs = plt.subplots(1, 3, figsize=(20, 5))

            ax = axs[0]
            pcm = ax.imshow(velocities[step][0, :, :, 1], vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'Simulation')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[1]
            pcm = ax.imshow(pred_velocities_vortex[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'VortexNet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[2]
            pcm = ax.imshow(pred_velocities_vortex_bc[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'VortexNet + BCNet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            fig.suptitle(r'Time: {}s'.format(step))

            filename = os.path.join(video_dir, 'vis_' + '0' * (6 - len(str(step))) + str(step) + '.png')
            plt.savefig(filename)

        video_name = os.path.join(vis_dir, 'video_sim_nn.avi')

        images = [img for img in sorted(os.listdir(video_dir)) if img.endswith(".png") and img.startswith("vis")]
        frame = cv2.imread(os.path.join(video_dir, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(video_dir, image)))

        video.release()

        # make video of neural network predictions and simulations in comparison along with error maps
        for step in range(NUM_TIME_STEPS + 1):
            os.remove(os.path.join(video_dir, images[step]))


        for step in range(NUM_TIME_STEPS + 1):
            fig, axs = plt.subplots(1, 5, figsize=(35, 5))

            ax = axs[0]
            pcm = ax.imshow(velocities[step][0, :, :, 1], vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'Simulation')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[1]
            pcm = ax.imshow(pred_velocities_vortex[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x,
                            cmap='coolwarm')
            ax.set_title(r'VortexNet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[2]
            pcm = ax.imshow(error_vel_mag_vortex[step][0, :, :, 0], cmap='Greys')
            ax.set_title(r'Error map: VortexNet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[3]
            pcm = ax.imshow(pred_velocities_vortex_bc[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x,
                            cmap='coolwarm')
            ax.set_title(r'VortexNet + BCNet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[4]
            pcm = ax.imshow(error_vel_mag_vortex_bc[step][0, :, :, 0], cmap='Greys')
            ax.set_title(r'Error map: VortexNet + BCNet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            fig.suptitle(r'Time: {}s'.format(step))

            filename = os.path.join(video_dir, 'vis_' + '0' * (6 - len(str(step))) + str(step) + '.png')
            plt.savefig(filename)

        video_name = os.path.join(vis_dir, 'video_sim_nn_error.avi')

        images = [img for img in sorted(os.listdir(video_dir)) if img.endswith(".png") and img.startswith("vis")]
        frame = cv2.imread(os.path.join(video_dir, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(video_dir, image)))

        video.release()

        for step in range(NUM_TIME_STEPS + 1):
            os.remove(os.path.join(video_dir, images[step]))

        os.rmdir(video_dir)





