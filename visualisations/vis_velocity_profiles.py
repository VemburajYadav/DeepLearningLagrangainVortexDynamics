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

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/data/single_vortex_dataset_256x256_16000/train/sim_006632',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='T2_exp(3)_weight_1.0_depth_2_100_batch_32_lr_5e-3_l2_1e-4_r256_16000', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=2, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--num_time_steps', type=int, default=5, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

loc_x = int(location[0, 0, 1])
loc_y = int(location[0, 0, 0])

py = points_x[0, :, loc_x, 0]
px = np.array([loc_x] * len(py), dtype=np.float32)

velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in range(NUM_TIME_STEPS + 1)]

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

logs_dir = os.path.join('../logs', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

VortexNet = MultiStepVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                   kernel=opt.kernel, num_steps=opt.num_time_steps, norm_mean=MEAN, norm_stddev=STDDEV,
                                   distinct_nets=opt.distinct_nets)

VortexNet.single_step_net.load_state_dict(params)
if opt.num_time_steps > 1 and opt.distinct_nets:
    params2 = torch.load(ckpt_file)['model_state_dict2']
    VortexNet.single_step_net2.load_state_dict(params2)

VortexNet.to('cuda:0')
VortexNet.eval()

if opt.kernel == 'offset-gaussian':
    off0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    sigl0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), off0, sigl0], dim=-1).view(-1, 1, 6)
    falloff_kernel = OffsetGaussianFalloffKernel()
if opt.kernel == 'ExpGaussian':
    c0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    d0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), c0, d0], dim=-1).view(-1, 1, 5)
    falloff_kernel = GaussExpFalloffKernel(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
if opt.kernel == 'ExpGaussianRed':
    d0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), d0], dim=-1).view(-1, 1, 5)
    falloff_kernel = GaussExpFalloffKernelReduced(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
elif opt.kernel == 'gaussian':
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1)], dim=-1).view(-1, 1, 4)
    falloff_kernel = GaussianFalloffKernel()

pred_velocites = []
losses= []

with torch.no_grad():

    vortex_features = VortexNet(inp_feature)

    for step in range(NUM_TIME_STEPS + 1):

        vel_y = falloff_kernel(vortex_features[step], points_y)
        vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
        vel_x = falloff_kernel(vortex_features[step], points_x)
        vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
        vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
        pred_velocites.append(vel)
        losses.append(F.mse_loss(vel, velocities_gpu[step], reduction='sum'))

loss_all = torch.stack(losses, dim=-1)
features = torch.stack(vortex_features, dim=-1)

plt.figure()
legend_list = []
for i in range(opt.num_time_steps + 1):
    plt.plot(velocities[i][0, loc_y-20:loc_y+20, loc_x, 1])
    legend_list.append('True: {}'.format(i*opt.stride))
    if i > 0:
        plt.plot(pred_velocites[i].cpu().numpy()[0, loc_y-20:loc_y+20, loc_x, 1], '--')
        legend_list.append('Pred: {}'.format(i*opt.stride))
plt.legend(legend_list)
plt.title('Strength: {:.2f}, Stddev: {:.2f}, Loss: {:.2f}'.format(strength[0, 0], sigma[0, 0, 0], loss_all.sum().item()))
plt.show()

max_val = np.abs(velocities[0][0, :, :, 1]).max()
min_val = -max_val

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(velocities[0][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 2)
plt.imshow(velocities[1][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 3)
plt.imshow(pred_velocites[1].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.show()

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(velocities[0][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 2)
plt.imshow(velocities[1][0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.subplot(1, 3, 3)
plt.imshow(pred_velocites[1].cpu().numpy()[0, :, :, 0], cmap='RdYlBu', vmin=min_val, vmax=max_val)
plt.show()

fig, axs = plt.subplots(1, 3)

ax = axs[0]
pcm = ax.imshow(velocities[0][0, :, :, 1],  cmap='RdYlBu',vmin=min_val, vmax=max_val)
ax.set_title('Time Step: 0')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(pcm, cax=cax)

ax = axs[1]
pcm = ax.imshow(velocities[1][0, :, :, 1],  cmap='RdYlBu',vmin=min_val, vmax=max_val)
ax.set_title('Time Step: 1')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(pcm, cax=cax)

ax = axs[2]
pcm = ax.imshow(velocities[2][0, :, :, 1],  cmap='RdYlBu',vmin=min_val, vmax=max_val)
ax.set_title('Time Step: 2')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(pcm, cax=cax)

fig.suptitle(' Velocity-x \n Strength: {:.2f} \n Core Size:'
             ' {:.2f}'.format(strength[0, 0], sigma[0, 0, 0]))

plt.show()