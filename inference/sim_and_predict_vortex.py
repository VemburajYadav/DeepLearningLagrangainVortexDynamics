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
parser.add_argument('--vortex_fit', type=bool, default=True,
                    help='whether to do vortex-fit on velocity fields from simulation (only applicable if argument: "sim" is True)')
parser.add_argument('--network_time_step', type=float, default=1.0,
                    help='time step in seconds over which the neural network is trained to make predictions')
parser.add_argument('--num_time_steps', type=int, default=20, help='number of time steps to make predictions')
parser.add_argument('--network', type=str, default='Vortex',
                    help='type of neural network for VortexNet: Vortex or Interaction')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--logs_dir', type=str, default=None, help='directory with checkpoints and training summaries')
parser.add_argument('--load_weights_ex', type=str, default=None,
                    help='name of the experiment to load checkpoint from')
parser.add_argument('--ckpt_path', type=str, default='../model/ckpt_vortexnet_2_inviscid.pytorch',
                    help='path to the actual checkpoint file (overrides the logs_dir and load_weights_ex argument)')
parser.add_argument('--save_dir', type=str, default='../Visuals/VortexNet_2_Inviscid/Case_1',
                    help='directory to write the neural network predictions and plots (leave it to the default value of None if not to save the outputs)')



# Parse input arguments
opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
RESOLUTION = opt.domain
NETWORK = opt.network
save_dir = opt.save_dir


# filename's and directories for saving outputs
velocity_filenames_pred = ['velocity_prediction_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
velocity_filenames_sim = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
velocity_filenames_fit = ['velocity_fit_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
vorticity_filenames_pred = ['vorticity_prediction_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
vorticity_filenames_fit = ['vorticity_fit_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
vortex_features_filenames_fit = ['vortex_features_fit_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
vortex_features_filenames_pred = ['vortex_features_prediction_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
error_vel_filenames = ['error_vel_mag_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
error_particle_features_filenames = ['error_abs_particle_features_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]


out_dir = os.path.join(save_dir, 'outputs')
vis_dir = os.path.join(save_dir, 'plots')


# Read location, strength and core size
location = np.reshape(np.load(os.path.join(opt.location))['arr_0'], (1, -1, 2))
strength = np.reshape(np.load(os.path.join(opt.strength))['arr_0'], (1, -1, 1))
sigma = np.reshape(np.load(os.path.join(opt.core_size))['arr_0'], (1, -1, 1))


# define domain and resolution of the grid
domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW_REF = Fluid(domain=domain)

# points in the staggered grid
points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

# points in th centered grid
points_cg = torch.tensor(FLOW_REF.density.points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

# Gaussian falloff-kernel for velocity
falloff_kernel = GaussianFalloffKernelVelocity()
# Gaussian falloff-kernel for vorticity
falloff_kernel_vorticity = GaussianFalloffKernelVorticity()


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
    velocities_ = [velocity_0]

    # PhiFlow physics object
    FLOW = Fluid(domain=domain, velocity=velocity_0)
    fluid = world.add(Fluid(domain=domain, velocity=velocity_0),
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


    ## Perform Vortex-Fit (if applicable)
    if opt.vortex_fit:
        particle_features_np = np.concatenate([location.reshape((1, -1, 2)),
                                               strength.reshape((1, -1, 1)),
                                               sigma.reshape((1, -1, 1))], axis=-1)

        opt_velocities = [velocities[0]]
        opt_features = [particle_features_np]

        opt_vorticity  = [falloff_kernel_vorticity(torch.tensor(opt_features[0],
                                                                device='cuda:0',
                                                                dtype=torch.float32,
                                                                requires_grad=False), points_cg).detach().clone().cpu().numpy()]


        for step in range(NUM_TIME_STEPS):

            particle_features_pt = torch.nn.Parameter(torch.tensor(opt_features[step],
                                                                   device='cuda:0',
                                                                   dtype=torch.float32, requires_grad=True))

            target_vel = torch.tensor(velocities[step + 1], device='cuda:0', dtype=torch.float32)

            optimizer = Adam(params=[particle_features_pt], lr=1e-1, weight_decay=1e-5)
            lambda_lr = lambda epoch: 0.95 ** epoch
            scheduler = LambdaLR(optimizer, lambda_lr)

            for epoch in range(1000):
                optimizer.zero_grad()
                vel_yy, vel_yx = torch.unbind(falloff_kernel(particle_features_pt, points_y), dim=-1)
                vel_xy, vel_xx = torch.unbind(falloff_kernel(particle_features_pt, points_x), dim=-1)

                vel_y = torch.cat([vel_yy, cat_y], dim=-1)
                vel_x = torch.cat([vel_xx, cat_x], dim=-2)

                pred_vel = torch.stack([vel_y, vel_x], dim=-1)

                loss = torch.sum((pred_vel - target_vel) ** 2)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch=epoch)

                print('Time Step: {}:-  Epoch: {}, Loss: {:.4f}'.format(step+1, epoch, loss.item()))

            opt_velocities.append(pred_vel.detach().clone().cpu().numpy())
            opt_features.append(particle_features_pt.detach().clone().cpu().numpy())
            opt_vorticity.append(falloff_kernel_vorticity(torch.tensor(opt_features[step + 1],
                                                                device='cuda:0',
                                                                dtype=torch.float32,
                                                                requires_grad=False), points_cg).detach().clone().cpu().numpy())

        opt_features_all = np.stack(opt_features, axis=-1)

## Neural network for Vortex Particle Dynamics
if NETWORK == 'Vortex':
    VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units,
                                            order=opt.order, num_steps=opt.num_time_steps)
else:
    VortexNet = MultiStepInteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, num_steps=opt.num_time_steps)



# Load weights directly from the path to the checkpoint file or
# the best checkpoint from the experiment in the logs directory
if opt.ckpt_path is None:
    init_weights_log_dir = os.path.join(opt.logs_dir, opt.load_weights_ex)
    init_weights_ckpt_dir = os.path.join(init_weights_log_dir, 'ckpt')

    checkpoints_files = os.listdir(os.path.join(init_weights_ckpt_dir))
    epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
    init_weights_ckpt_file = os.path.join(init_weights_ckpt_dir, checkpoints_files[epoch_id])
else:
    init_weights_ckpt_file = os.path.join(opt.ckpt_path)

params = torch.load(init_weights_ckpt_file)['model_state_dict']
VortexNet.single_step_net.load_state_dict(params)


# Neural network to gpu
VortexNet.to('cuda:0')
VortexNet.eval()


# tensors to gpu
loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

nparticles = location.shape[1]

py, px = torch.unbind(loc_gpu, dim=-1)
inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles)], dim=-1)

# predictions from neural network
vortex_features = VortexNet(inp_feature.detach().clone())

pred_velocities = []
pred_vorticity = []

# compute mse and mae losses on the velocity fields
with torch.no_grad():
    for step in range(NUM_TIME_STEPS + 1):
        vel_y = falloff_kernel(vortex_features[step], points_y)
        vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
        vel_x = falloff_kernel(vortex_features[step], points_x)
        vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
        vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
        pred_velocities.append(vel.detach().clone())
        vort = falloff_kernel_vorticity(vortex_features[step], points_cg)
        pred_vorticity.append(vort.detach().clone().cpu().numpy())

    features = torch.stack(vortex_features, dim=-1).detach().clone().cpu().numpy()

    # compute losses (if applicable)
    if opt.sim:
        velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in
                          range(NUM_TIME_STEPS + 1)]

        mse_losses = []
        l1_losses = []
        error_vel_mag = []

        for step in range(NUM_TIME_STEPS + 1):
            mse_losses.append(F.mse_loss(pred_velocities[step], velocities_gpu[step], reduction='sum').detach().clone())
            l1_losses.append(F.l1_loss(pred_velocities[step], velocities_gpu[step], reduction='sum').detach().clone())
            error_vel_mag.append(torch.sqrt(torch.sum((pred_velocities[step] - velocities_gpu[step])**2,
                                                      dim=-1, keepdim=True)).detach().clone().cpu().numpy())

        loss_all_mse = torch.stack(mse_losses, dim=-1).detach().clone().cpu().numpy()
        loss_all_l1 = torch.stack(l1_losses, dim=-1).detach().clone().cpu().numpy()

        # compute absolute error on particle locations, strengths and core sizes in comparison with Vortex-Fit (if applicable)
        if opt.vortex_fit:
            error_particle_features = []

            for step in range(NUM_TIME_STEPS + 1):
                p_error = np.sum(np.abs(features[0, :, :, step] - opt_features[step][0, :, :]), axis=0) / nparticles
                error_particle_features.append(p_error)



# save outputs from simulation, Vortex-Fit adn Neural networks (if applicable)
if opt.save_dir:

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for i in range(NUM_TIME_STEPS + 1):
        np.savez_compressed(os.path.join(out_dir, velocity_filenames_pred[i]), pred_velocities[i].cpu().numpy())
        np.savez_compressed(os.path.join(out_dir, vorticity_filenames_pred[i]), pred_vorticity[i])
        np.savez_compressed(os.path.join(out_dir, vortex_features_filenames_pred[i]), features[:, :, :, i])

        if opt.sim:
            np.savez_compressed(os.path.join(out_dir, velocity_filenames_sim[i]), velocities[i])
            np.savez_compressed(os.path.join(out_dir, error_vel_filenames[i]), error_vel_mag[i])

            if opt.vortex_fit:
                np.savez_compressed(os.path.join(out_dir, velocity_filenames_fit[i]), opt_velocities[i])
                np.savez_compressed(os.path.join(out_dir, vorticity_filenames_fit[i]), opt_vorticity[i])
                np.savez_compressed(os.path.join(out_dir, vortex_features_filenames_fit[i]), opt_features[i])
                np.savez_compressed(os.path.join(out_dir, error_particle_features_filenames[i]), error_particle_features[i])


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
        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(pred_velocities[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_x_prediction_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')

        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(pred_velocities[step][0, :, :, 0].cpu().numpy(), vmin=min_vel_y, vmax=max_vel_y, cmap='coolwarm')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_y_prediction_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')

        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(np.sqrt(pred_velocities[step][0, :, :, 1].cpu().numpy()**2 +
                                pred_velocities[step][0, :, :, 0].cpu().numpy()**2),
                        vmin=0, vmax=max_vel_mag, cmap='viridis')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'velocity_mag_prediction_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
                    bbox_inches='tight')


        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(error_vel_mag[step][0, :, :, 0], cmap='Greys')
        ax.set_xlim([0, RESOLUTION[1]])
        ax.set_ylim([0, RESOLUTION[0]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0)
        fig.colorbar(pos, ax=ax)
        fig.savefig(os.path.join(vis_dir, 'error_vel_mag_' + '0' * (6 - len(str(step))) + str(step) + '.png'), format='png',
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


    if opt.vortex_fit:

        particle_features_dir = os.path.join(vis_dir, 'particle_dynamics')

        if not os.path.isdir(particle_features_dir):
            os.makedirs(particle_features_dir)

        t = np.arange(0, (NUM_TIME_STEPS + 1) * NN_TIME_STEP, NN_TIME_STEP)

        for p in range(nparticles):
            fig, ax = plt.subplots(1, 1)
            ax.plot(t, opt_features_all[0, p, 1, :])
            ax.plot(t, features[0, p, 1, :])
            ax.set_xlabel(r'time $t(s)$')
            ax.set_ylabel(r'$x_{}(t)$'.format(p), rotation=0)
            plt.legend([r'Vortex-Fit', r'Neural Network'])
            plt.show()
            fig.savefig(os.path.join(particle_features_dir, 'x_p{:02d}.png'.format(p)), format='png')

            fig, ax = plt.subplots(1, 1)
            ax.plot(t, opt_features_all[0, p, 0, :])
            ax.plot(t, features[0, p, 0, :])
            ax.set_xlabel(r'time $t(s)$')
            ax.set_ylabel(r'$y_{}(t)$'.format(p), rotation=0)
            plt.legend([r'Vortex-Fit', r'Neural Network'])
            plt.show()
            fig.savefig(os.path.join(particle_features_dir, 'y_p{:02d}.png'.format(p)), format='png')

            fig, ax = plt.subplots(1, 1)
            ax.plot(t, opt_features_all[0, p, 2, :])
            ax.plot(t, features[0, p, 2, :])
            ax.set_xlabel(r'time $t(s)$')
            ax.set_ylabel(r'$\Gamma_{}(t)$'.format(p), rotation=0)
            plt.legend([r'Vortex-Fit', r'Neural Network'])
            plt.show()
            fig.savefig(os.path.join(particle_features_dir, 'strength_p{:02d}.png'.format(p)), format='png')

            fig, ax = plt.subplots(1, 1)
            ax.plot(t, opt_features_all[0, p, 3, :])
            ax.plot(t, features[0, p, 3, :])
            ax.set_xlabel(r'time $t(s)$')
            ax.set_ylabel(r'$\sigma_{}(t)$'.format(p), rotation=0)
            plt.legend([r'Vortex-Fit', r'Neural Network'])
            plt.show()
            fig.savefig(os.path.join(particle_features_dir, 'core_size_p{:02d}.png'.format(p)), format='png')


    ## Make videos
    video_dir = os.path.join(vis_dir, 'video_temp')

    if not os.path.isdir(video_dir):
        os.makedirs(video_dir)

    # make video of neural network predictions
    for step in range(NUM_TIME_STEPS + 1):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        pcm = ax.imshow(pred_velocities[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
        ax.set_title(r'Neural Network')
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
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))

            ax = axs[0]
            pcm = ax.imshow(velocities[step][0, :, :, 1], vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'Simulation')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[1]
            pcm = ax.imshow(pred_velocities[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'Neural Network')
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
            fig, axs = plt.subplots(1, 3, figsize=(20, 5))

            ax = axs[0]
            pcm = ax.imshow(velocities[step][0, :, :, 1], vmin=min_vel_x, vmax=max_vel_x, cmap='coolwarm')
            ax.set_title(r'Simulation')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[1]
            pcm = ax.imshow(pred_velocities[step][0, :, :, 1].cpu().numpy(), vmin=min_vel_x, vmax=max_vel_x,
                            cmap='coolwarm')
            ax.set_title(r'Neural Network')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

            ax = axs[2]
            pcm = ax.imshow(error_vel_mag[step][0, :, :, 0], cmap='Greys')
            ax.set_title(r'Error map')
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





