import numpy as np
import torch
import torch.nn.functional as F
from core.custom_functions import *
import argparse
import glob
from phi.flow import *
from core.networks import *
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--data_dir', type=str, default='../'
                                                    'data/p10_gaussian_dataset_120x120_4000/val/',
                    help='path to the directory with data samples to compute the performance metrics')
parser.add_argument('--network', type=str, default='Vortex',
                    help='type of neural network: Vortex or Interaction')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=2, help='number of time steps to evaluate the metrics for')
parser.add_argument('--stride', type=int, default=5, help='skip intermediate time frames corresponding to stride in the dataset '
                                                          'for evaluation')
parser.add_argument('--logs_dir', type=str, default=None, help='directory with checkpoints and training summaries')
parser.add_argument('--load_weights_ex', type=str, default=None,
                    help='name of the experiment to load checkpoint from')
parser.add_argument('--ckpt_path', type=str, default='../model/ckpt_vortexnet_2_inviscid.pytorch',
                    help='path to the actual checkpoint file (overrides the logs_dir and load_weights_ex argument)')
parser.add_argument('--save_dir', type=str, default=None,
                    help='directory to write the neural network products (leave it to the default value of None if not to save the outputs)')



# Parse Input arguments
opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride
RESOLUTION = opt.domain
NETWORK = opt.network
data_dir = opt.data_dir
save_dir = opt.save_dir

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

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

# Neural network for Vortex Particle Dynamics
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


# Gaussian falloff-kernel
falloff_kernel = GaussianFalloffKernelVelocity()


# filename's for saving velocity fields
velocity_filenames = ['velocity_prediction_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]


# track losses for all the data samples
loss_track_mse = 0.0
loss_track_l1 = 0.0


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

    velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in
                      range(NUM_TIME_STEPS + 1)]

    loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
    tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
    sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles)], dim=-1)

    # predictions from neural network
    vortex_features = VortexNet(inp_feature.detach().clone())

    pred_velocities = []
    mse_losses = []
    l1_losses = []

    # compute mse and mae losses on the velocity fields
    with torch.no_grad():
        for step in range(NUM_TIME_STEPS + 1):
            vel_y = falloff_kernel(vortex_features[step], points_y)
            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_x = falloff_kernel(vortex_features[step], points_x)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
            vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
            pred_velocities.append(vel.detach().clone())
            mse_losses.append(F.mse_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())
            l1_losses.append(F.l1_loss(vel, velocities_gpu[step], reduction='sum').detach().clone())

        loss_all_mse = torch.stack(mse_losses, dim=-1)
        loss_all_l1 = torch.stack(l1_losses, dim=-1)

        features = torch.stack(vortex_features, dim=-1)

        loss_track_mse = loss_track_mse + loss_all_mse
        loss_track_l1 = loss_track_l1 + loss_all_l1


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
            np.savez_compressed(os.path.join(case_dir_path, velocity_filenames[frame]),
                                pred_velocities[frame].cpu().numpy())

        np.savez_compressed(os.path.join(case_dir_path, 'vortex_features_predictions.npz'), features.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'mse_loss.npz'), loss_all_mse.cpu().numpy())
        np.savez_compressed(os.path.join(case_dir_path, 'mae_loss.npz'), loss_all_l1.cpu().numpy())


# final metrics over all the data sample sin the directory
loss_mse = loss_track_mse / len(data_cases)
loss_l1 = loss_track_l1 / len(data_cases)


# print the metrics
for i in range(NUM_TIME_STEPS):
    print('MSE loss for time step: {} = {:.4f}'.format(i+1, loss_mse[i+1].item()))
    print('MAE loss for time step: {} = {:.4f}'.format(i+1, loss_l1[i+1].item()))
    print('')











