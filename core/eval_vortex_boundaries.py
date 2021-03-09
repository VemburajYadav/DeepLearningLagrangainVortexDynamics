import numpy as np
import torch
import torch.nn.functional as F
from core.custom_functions import *
import argparse
import glob
from phi.flow import *
from core.networks import *
from core.velocity_derivs import *
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--data_dir', type=str, default='../'
                                                    'data/p10_b_sb_gaussian_dataset_120x120_4000/val/',
                    help='path to the directory with data samples to compute the performance metrics')
parser.add_argument('--network', type=str, default='Vortex',
                    help='type of neural network for VortexNet: Vortex or Interaction')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=1, help='number of time steps to evaluate the metrics for')
parser.add_argument('--sim_time_step', type=float, default=0.2,
                    help='time step in seconds for running numerical simulations')
parser.add_argument('--network_time_step', type=float, default=1.0,
                    help='time step in seconds over which the neural network is trained to make predictions')
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
parser.add_argument('--save_dir', type=str, default=None,
                    help='directory to write the neural network products (leave it to the default value of None if not to save the outputs)')



# Parse Input arguments
opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
RESOLUTION = opt.domain
NETWORK = opt.network
ORDER = opt.order
data_dir = opt.data_dir
save_dir = opt.save_dir

SIM_TIME_STEP = opt.sim_time_step
NN_TIME_STEP = opt.network_time_step
STRIDE = int(NN_TIME_STEP / SIM_TIME_STEP)


# get the directory paths for individual data samples
check_single_case = True in [i.endswith('.npz') for i in sorted(glob.glob(os.path.join(data_dir, '*')))]

if check_single_case:
    data_cases = [os.path.join(data_dir)]
else:
    data_cases = sorted(glob.glob(os.path.join(data_dir, '*')))


# define domain and resolution of the grid
domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW_REF = Fluid(domain=domain)

# points in the staggered grid
points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

points_y_res = points_y.view(1, -1, 2)
points_x_res = points_x.view(1, -1, 2)

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

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


# Neural networks to gpu
VortexNet.to('cuda:0')
VortexNet.eval()

BCNet.to('cuda:0')
BCNet.eval()


# Gaussian falloff-kernel
falloff_kernel = GaussianFalloffKernelVelocity()


# Module to compute the velocities and derivatives of velocities due to vortex particles
VelDerivExpRed = VelocityDerivatives(order=ORDER)


# filename's for saving velocity fields
velocity_filenames_vortex = ['velocity_prediction_VortexNet_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]
velocity_filenames_vortex_bc = ['velocity_prediction_VortexBCNet_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]


# track losses for all the data samples
loss_track_mse_vortex = 0.0
loss_track_l1_vortex = 0.0
loss_track_mse_vortex_bc = 0.0
loss_track_l1_vortex_bc = 0.0
div_loss_track = 0.0
bc_loss_track_vortex = 0.0
bc_loss_track_vortex_bc = 0.0


# make predictions using neural network for each data sample
for case in range(len(data_cases)):

    # Read locations, strengths and core sizes
    location = np.load(os.path.join(data_cases[case], 'location_000000.npz'))['arr_0']
    strength = np.load(os.path.join(data_cases[case], 'strength_000000.npz'))['arr_0']
    sigma = np.load(os.path.join(data_cases[case], 'sigma_000000.npz'))['arr_0']

    nparticles = location.shape[1]

    # read velocity fields from simulation
    velocities = [np.load(os.path.join(data_cases[case], 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
                  for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

    velocities[0] = np.load(os.path.join(data_cases[case], 'velocity_div_000000.npz'))['arr_0']

    velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in
                      range(NUM_TIME_STEPS + 1)]

    loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
    tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
    sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles)], dim=-1)

    # predictions from neural network
    vortex_features = VortexNet(inp_feature.detach().clone())

    pred_velocities_vortex = []
    mse_losses_vortex = []
    l1_losses_vortex = []

    # compute mse and mae losses on the velocity fields as a result of predictions from VortexNet
    with torch.no_grad():
        for step in range(NUM_TIME_STEPS + 1):
            vel_y = falloff_kernel(vortex_features[step], points_y)
            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_x = falloff_kernel(vortex_features[step], points_x)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
            vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
            pred_velocities_vortex.append(vel.detach().clone())
            mse_losses_vortex.append(F.mse_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())
            l1_losses_vortex.append(F.l1_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())

        loss_all_mse_vortex = torch.stack(mse_losses_vortex, dim=-1)
        loss_all_l1_vortex = torch.stack(l1_losses_vortex, dim=-1)

        features = torch.stack(vortex_features, dim=-1)

        loss_track_mse_vortex = loss_track_mse_vortex + loss_all_mse_vortex
        loss_track_l1_vortex = loss_track_l1_vortex + loss_all_l1_vortex



    pred_velocities_vortex_bc = []
    mse_losses_vortex_bc = []
    l1_losses_vortex_bc = []
    div_loss_mse = []
    bc_loss_mse_vortex = []
    bc_loss_mse_vortex_bc = []


    # Vortex Network  --->   Boundary condition network
    for step in range(NUM_TIME_STEPS + 1):

        time_index = torch.tensor([step], device='cuda:0')
        features_time = torch.index_select(features, dim=-1, index=time_index).view(1, -1, 4)

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
        pred_velocities_vortex_bc.append(vel_vortex_bc)

        # compute mse and mae losses on the velocity fields as a result of predictions from VortexNet + BCNet
        mse_losses_vortex_bc.append(F.mse_loss(vel_vortex_bc, velocities_gpu[step], reduction='sum'))
        l1_losses_vortex_bc.append(F.l1_loss(vel_vortex_bc, velocities_gpu[step], reduction='sum'))

        # compute boundary condition loss before and after predictions from BCNet
        bc_loss_before = \
            torch.sum((pred_velocities_vortex[step][0, :, 0, 1]**2 + pred_velocities_vortex[step][0, :, -1, 1]**2 +
                       pred_velocities_vortex[step][0, 0, :, 0]**2 + pred_velocities_vortex[step][0, -1, :, 0]**2))

        bc_loss_after = \
            torch.sum((pred_velocities_vortex_bc[step][0, :, 0, 1]**2 + pred_velocities_vortex_bc[step][0, :, -1, 1]**2 +
                       pred_velocities_vortex_bc[step][0, 0, :, 0]**2 + pred_velocities_vortex_bc[step][0, -1, :, 0]**2))

        bc_loss_mse_vortex.append(bc_loss_before)
        bc_loss_mse_vortex_bc.append(bc_loss_after)

    loss_all_mse_vortex_bc = torch.stack(mse_losses_vortex_bc, dim=-1)
    loss_all_l1_vortex_bc = torch.stack(l1_losses_vortex_bc, dim=-1)
    div_loss_all = torch.stack(div_loss_mse, dim=-1)
    bc_loss_before_all = torch.stack(bc_loss_mse_vortex, dim=-1)
    bc_loss_after_all = torch.stack(bc_loss_mse_vortex_bc, dim=-1)

    loss_track_mse_vortex_bc = loss_track_mse_vortex_bc + loss_all_mse_vortex_bc
    loss_track_l1_vortex_bc = loss_track_l1_vortex_bc + loss_all_l1_vortex_bc
    div_loss_track = div_loss_track + div_loss_all
    bc_loss_track_vortex = bc_loss_track_vortex + bc_loss_before_all
    bc_loss_track_vortex_bc = bc_loss_track_vortex_bc + bc_loss_after_all

    # save the predictions (if applicable)
    if opt.save_dir is not None:
        if os.path.join(data_cases[case]).endswith('/'):
            case_dir = data_cases[case].split('/')[-2]
        else:
            case_dir = data_cases[case].split('/')[-1]

        case_dir_path = os.path.join(save_dir, case_dir)

        if not os.path.isdir(case_dir_path):
            os.makedirs(case_dir_path)

        for frame in range(NUM_TIME_STEPS + 1):
            np.savez_compressed(os.path.join(case_dir_path, velocity_filenames_vortex[frame]),
                                pred_velocities_vortex[frame].cpu().numpy())
            np.savez_compressed(os.path.join(case_dir_path, velocity_filenames_vortex_bc[frame]),
                                pred_velocities_vortex_bc[frame].cpu().numpy())

        np.savez_compressed(os.path.join(case_dir_path, 'vortex_features_predictions.npz'), features.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'mse_loss_vortex.npz'), loss_all_mse_vortex.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'mse_loss_vortex_bc.npz'), loss_all_mse_vortex_bc.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'mae_loss_vortex.npz'), loss_all_l1_vortex.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'mae_loss_vortex_bc.npz'), loss_all_l1_vortex_bc.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'bc_loss_vortex.npz'), bc_loss_before_all.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'bc_loss_vortex_bc.npz'), bc_loss_after_all.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'div_loss.npz'), div_loss_all.cpu().numpy())


# final metrics over all the data sample sin the directory
loss_mse_vortex = loss_track_mse_vortex / len(data_cases)
loss_l1_vortex = loss_track_l1_vortex / len(data_cases)
loss_mse_vortex_bc = loss_track_mse_vortex_bc / len(data_cases)
loss_l1_vortex_bc = loss_track_l1_vortex_bc / len(data_cases)
bc_loss_vortex = bc_loss_track_vortex / len(data_cases)
bc_loss_vortex_bc = bc_loss_track_vortex_bc / len(data_cases)
loss_div = div_loss_track / len(data_cases)


# print the metrics
for i in range(NUM_TIME_STEPS):
    print('MSE loss for time step: {} from VortexNet = {:.4f}'.format(i+1, loss_mse_vortex[i+1].item()))
    print('MAE loss for time step: {} from VortexNet = {:.4f}'.format(i+1, loss_l1_vortex[i+1].item()))
    print('MSE loss for time step: {} from VortexNet + BCNet = {:.4f}'.format(i+1, loss_mse_vortex_bc[i+1].item()))
    print('MAE loss for time step: {} from VortexNet + BCNet = {:.4f}'.format(i+1, loss_l1_vortex_bc[i+1].item()))
    print('Boundary Condition loss for time step: {} from VortexNet = {:.4f}'.format(i+1, bc_loss_vortex[i+1].item()))
    print('Boundary Condition loss for time step: {} from VortexNet + BCNet = {:.4f}'.format(i+1, bc_loss_vortex_bc[i+1].item()))
    print('Divergence loss for time step: {} from VortexNet + BCNet = {:.4f}'.format(i+1, loss_div[i+1].item()))
    print('')





