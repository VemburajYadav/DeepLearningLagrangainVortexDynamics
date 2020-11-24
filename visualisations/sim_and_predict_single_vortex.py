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
parser.add_argument('--case_path', type=str, default='/media/vemburaj/9d072277-d226-41f6-a38d-1db833dca2bd/'
                                                     'data/p2_r_dataset_256x256_32000/train/sim_007246',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='T2_exp_red(5)_weight_1.0_depth_2_100_batch_32_lr_1e-3_l2_1e-4_r256_16000', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=2, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--num_time_steps', type=int, default=50, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# cuda.select_device(0)

save_dir = os.path.join('../p1_samples/case_1')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

RESOLUTION = opt.domain

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

# location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
# strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
# sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

loc = [120.7, 145.2]
location = np.array(loc).reshape((1, 1, 2))

strength = np.array([1.9])
sigma = np.array([6.0]).reshape((1, 1, 1))

NPARTICLES = location.shape[1]

def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    return (math.exp(- sq_distance / sigma ** 2)) / math.sqrt(sq_distance)

domain = Domain(RESOLUTION, boundaries=OPEN)
FLOW_REF = Fluid(domain)

# location_pl = tf.placeholder(shape=(1, NPARTICLES, 2), dtype=tf.float32)
# strength_pl = tf.placeholder(shape=(NPARTICLES, ), dtype=tf.float32)
# sigma_pl = tf.placeholder(shape=(1, NPARTICLES, 1), dtype=tf.float32)

vorticity = AngularVelocity(location=location,
                            strength=strength,
                            falloff=partial(gaussian_falloff, sigma=sigma))

velocity_0 = vorticity.at(FLOW_REF.velocity)
velocities_ = [velocity_0]

FLOW = Fluid(domain=domain, velocity=velocity_0)
fluid = world.add(Fluid(domain=domain, velocity=velocity_0), physics=IncompressibleFlow())

for step in range(NUM_TIME_STEPS):
    world.step()
    velocities_.append(fluid.velocity)

# sess = Session(None)
# velocities_ = sess.run(velocities_tf, feed_dict={location_pl: location, strength_pl: strength, sigma_pl: sigma})

# cuda.close()
#
# cuda.select_device(0)

velocities = []
for i in range(NUM_TIME_STEPS + 1):
    vx = np.concatenate([velocities_[i].x.data, np.zeros((1, 1, RESOLUTION[1] + 1, 1))], axis=-3)
    vy = np.concatenate([velocities_[i].y.data, np.zeros((1, RESOLUTION[0] + 1, 1, 1))], axis=-2)
    velocities.append(np.concatenate([vy, vx], axis=-1))

nparticles = location.shape[1]

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
    d0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0') + 0.001
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), c0, d0], dim=-1).view(-1, 1, 6)
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

max_val = np.abs(velocities[0][0, :, :, 1]).max()
min_val = -max_val

total_velocities_pred = []
total_velocities = []
error_map = []
for step in range(NUM_TIME_STEPS + 1):
    total_velocities_pred.append(torch.sqrt(torch.sum(pred_velocites[step]**2, dim=-1)))
    total_velocities.append(torch.sqrt(torch.sum(velocities_gpu[step]**2, dim=-1)))
    error_map.append(torch.abs((total_velocities[step] - total_velocities_pred[step])))

# cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']
# pylab.plot(velocities[0][0, :, :, 1], color=cycle[0])

min_val = total_velocities[0].min()
max_val = total_velocities[0].max()

for step in range(NUM_TIME_STEPS + 1):
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))

    ax = axs[0]
    pcm = ax.imshow(total_velocities[step].cpu().numpy()[0, 80:180, 80:180], vmin=min_val, vmax=max_val)
    ax.set_title('Simulation')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax)

    ax = axs[1]
    pcm = ax.imshow(total_velocities_pred[step].cpu().numpy()[0, 80:180, 80:180], vmin=min_val, vmax=max_val)
    ax.set_title('Neural Network')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax)

    ax = axs[2]
    pcm = ax.imshow(error_map[step].cpu().numpy()[0, 80:180, 80:180]**2, cmap='Greys')
    ax.set_title('Error Map: {:.2f}'.format(np.sum(error_map[step].cpu().numpy()[0, 80:180, 80:180]**2)))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax)

    fig.suptitle(' Strength: {:.2f} \n Core Size:'
                 ' {:.2f} \n Velocity - Time Step: {}'.format(strength[0], sigma[0, 0, 0], step))

    filename = os.path.join(save_dir, 'vis_' + '0' * (6 - len(str(step))) + str(step) + '.png')
    plt.savefig(filename)

plt.figure(figsize=(16, 10))
legend_list = []
for i in range(6):
    plt.plot(math.abs(velocities[i][0, loc_y - 20:loc_y + 20, loc_x, 1]))
    legend_list.append('True: {}'.format(i * opt.stride))
    if i > 0:
        plt.plot(math.abs(pred_velocites[i].cpu().numpy()[0, loc_y - 20:loc_y + 20, loc_x, 1]), '--')
        legend_list.append('Pred: {}'.format(i * opt.stride))
plt.legend(legend_list)
plt.title("Variation of velocity-x along y-axis \n 'Strength: {:.2f}, "
          "Stddev: {:.2f}".format(strength[0], sigma[0, 0, 0]))
plt.savefig(os.path.join(save_dir, 'velocity-profile-y.png'))
plt.show()
#