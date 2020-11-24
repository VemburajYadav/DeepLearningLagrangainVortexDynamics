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

parser.add_argument('--domain', type=list, default=[200, 200], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/'
                                                     'data/p10_b_dataset_200x200_16000/train/sim_007217',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='p10_b_T2_exp_red(6)_weight_1.0_depth_10_100_batch_32_lr_5e-3_l2_1e-4_r256_16000_2', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=10, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--num_time_steps', type=int, default=50, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# cuda.select_device(0)

save_dir = os.path.join('../p10_samples/case_7')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

RESOLUTION = opt.domain

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']
#
# loc_1 = [120.7, 145.2]
# dist = 10.0
# angle = 45.0
# loc_2 = [loc_1[0] + dist * np.sin(angle * np.pi / 180.0), loc_1[1] + dist * np.cos(angle * np.pi / 180.0)]
# location = np.array([loc_1, loc_2]).reshape((1, 2, 2))
#
# strength = np.array([1.9, 1.9])
# sigma = np.array([20.0, 20.0]).reshape((1, 2, 1))
#
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

if NPARTICLES > 2:
    loc_x3 = int(location[0, 2, 1])
    loc_y3 = int(location[0, 2, 0])
    loc_x4 = int(location[0, 3, 1])
    loc_y4 = int(location[0, 3, 0])

    py3 = points_x[0, :, loc_x3, 0]
    px3 = np.array([loc_x3] * len(py3), dtype=np.float32)

    py4 = points_x[0, :, loc_x4, 0]
    px4 = np.array([loc_x4] * len(py4), dtype=np.float32)

    px3_ = points_y[0, loc_y3, :, 1]
    py3_ = np.array([loc_y3] * len(px3_), dtype=np.float32)

    px4_ = points_y[0, loc_y4, :, 1]
    py4_ = np.array([loc_y4] * len(px4_), dtype=np.float32)

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
    d0 = torch.zeros((1, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
    py, px = torch.unbind(loc_gpu, dim=-1)
    inp_feature = torch.stack([py, px, tau_gpu.view(-1, nparticles), sig_gpu.view(-1, nparticles), d0], dim=-1)
    falloff_kernel = GaussExpFalloffKernelReduced(dt=torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0'))
elif opt.kernel == 'gaussian':
    inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1)], dim=-1)
    falloff_kernel = GaussianFalloffKernel()

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
    pcm = ax.imshow(total_velocities[step].cpu().numpy()[0, 50:150, 50:150], vmin=min_val, vmax=max_val)
    ax.set_title('Simulation')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax)

    ax = axs[1]
    pcm = ax.imshow(total_velocities_pred[step].cpu().numpy()[0, 50:150, 50:150], vmin=min_val, vmax=max_val)
    ax.set_title('Neural Network')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax)

    ax = axs[2]
    pcm = ax.imshow(error_map[step].cpu().numpy()[0, 50:150, 50:150]**2, cmap='Greys')
    ax.set_title('Error Map: {:.2f}'.format(np.sum(error_map[step].cpu().numpy()[0, 50:150, 50:150]**2)))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax)

    fig.suptitle(' Strengths: {:.2f}, {:.2f} \n Core Sizes:'
                 ' {:.2f}, {:.2f} \n Velocity - Time Step: {}'.format(strength[0], strength[1], sigma[0, 0, 0], sigma[0, 1, 0], step))

    filename = os.path.join(save_dir, 'vis_' + '0' * (6 - len(str(step))) + str(step) + '.png')
    plt.savefig(filename)
#

if NPARTICLES == 2:

    dist = ((loc_x2 - loc_x1) ** 2 + (loc_y2 - loc_y1) ** 2) ** 0.5

    plt.figure(figsize=(16, 16))
    plt.subplot(2,1,1)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(py1[loc_y1-80:loc_y1+80], math.abs(velocities[0][0, loc_y1-80:loc_y1+80, loc_x1, 1]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(py1[loc_y1-80:loc_y1+80], math.abs(velocities[i+1][0, loc_y1-80:loc_y1+80, loc_x1, 1]))
        legend_list.append('True: {}'.format(i+1))
        plt.plot(py1[loc_y1-80:loc_y1+80], math.abs(pred_velocites[i+1].cpu().numpy()[0, loc_y1-80:loc_y1+80, loc_x1, 1]), '--')
        legend_list.append('Pred: {}'.format(i+1))
    plt.axvline(x=loc_y1, color='b')
    plt.axvline(x=loc_y2, color='r')
    plt.legend(legend_list)
    plt.title('Particle 1 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}'.format(dist, strength[0], sigma[0, 0, 0]))

    plt.subplot(2,1,2)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(py2[loc_y2-80:loc_y2+80], math.abs(velocities[0][0, loc_y2-80:loc_y2+80, loc_x2, 1]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(py2[loc_y2-80:loc_y2+80], math.abs(velocities[i+1][0, loc_y2-80:loc_y2+80, loc_x2, 1]))
        legend_list.append('True: {}'.format(i+1))
        plt.plot(py2[loc_y2-80:loc_y2+80], math.abs(pred_velocites[i+1].cpu().numpy()[0, loc_y2-80:loc_y2+80, loc_x2, 1]), '--')
        legend_list.append('Pred: {}'.format(i+1))
    plt.axvline(x=loc_y2, color='b')
    plt.axvline(x=loc_y1, color='r')
    plt.legend(legend_list)
    plt.title('Particle 2 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}'.format(dist, strength[1], sigma[0, 1, 0]))
    plt.suptitle('Variation of velocity-x along y-axis')
    plt.savefig(os.path.join(save_dir, 'velocity-profile-x.png'))
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.subplot(2,1,1)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(px1_[loc_x1-80:loc_x1+80], math.abs(velocities[0][0, loc_y1, loc_x1-80:loc_x1+80, 0]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(px1_[loc_x1-80:loc_x1+80], math.abs(velocities[i+1][0, loc_y1, loc_x1-80:loc_x1+80, 0]))
        legend_list.append('True: {}'.format(i+1))
        plt.plot(px1_[loc_x1-80:loc_x1+80], math.abs(pred_velocites[i+1].cpu().numpy()[0, loc_y1, loc_x1-80:loc_x1+80, 0]), '--')
        legend_list.append('Pred: {}'.format(i+1))
    plt.axvline(x=loc_x2, color='r')
    plt.axvline(x=loc_x1, color='b')
    plt.legend(legend_list)
    plt.title('Particle 1 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}'.format(dist, strength[0], sigma[0, 0, 0]))
    plt.subplot(2,1,2)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(px2_[loc_x2-80:loc_x2+80], math.abs(velocities[0][0, loc_y2, loc_x2-80:loc_x2+80, 0]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(px2_[loc_x2-80:loc_x2+80], math.abs(velocities[i+1][0, loc_y2, loc_x2-80:loc_x2+80, 0]))
        legend_list.append('True: {}'.format(i+1))
        plt.plot(px2_[loc_x2-80:loc_x2+80], math.abs(pred_velocites[i+1].cpu().numpy()[0, loc_y2, loc_x2-80:loc_x2+80, 0]), '--')
        legend_list.append('Pred: {}'.format(i+1))
    plt.axvline(x=loc_x1, color='r')
    plt.axvline(x=loc_x2, color='b')
    plt.legend(legend_list)
    plt.title('Particle 2 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}'.format(dist, strength[1], sigma[0, 1, 0]))
    plt.suptitle('Variation of velocity-y along x-axis')
    plt.savefig(os.path.join(save_dir, 'velocity-profile-y.png'))
    plt.show()

else:

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(py1[loc_y1 - 80:loc_y1 + 80], math.abs(velocities[0][0, loc_y1 - 80:loc_y1 + 80, loc_x1, 1]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(py1[loc_y1 - 80:loc_y1 + 80], math.abs(velocities[i + 1][0, loc_y1 - 80:loc_y1 + 80, loc_x1, 1]))
        legend_list.append('True: {}'.format(i + 1))
        plt.plot(py1[loc_y1 - 80:loc_y1 + 80],
                 math.abs(pred_velocites[i + 1].cpu().numpy()[0, loc_y1 - 80:loc_y1 + 80, loc_x1, 1]), '--')
        legend_list.append('Pred: {}'.format(i + 1))
    plt.axvline(x=loc_y1, color='b')
    plt.legend(legend_list)
    plt.title('Particle 1 (Blue) :- , Strength: {:.2f}, Stddev: {:.2f}'.format(strength[0],
                                                                                                    sigma[0, 0, 0]))

    plt.subplot(2, 2, 2)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(py2[loc_y2 - 80:loc_y2 + 80], math.abs(velocities[0][0, loc_y2 - 80:loc_y2 + 80, loc_x2, 1]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(py2[loc_y2 - 80:loc_y2 + 80], math.abs(velocities[i + 1][0, loc_y2 - 80:loc_y2 + 80, loc_x2, 1]))
        legend_list.append('True: {}'.format(i + 1))
        plt.plot(py2[loc_y2 - 80:loc_y2 + 80],
                 math.abs(pred_velocites[i + 1].cpu().numpy()[0, loc_y2 - 80:loc_y2 + 80, loc_x2, 1]), '--')
        legend_list.append('Pred: {}'.format(i + 1))
    plt.axvline(x=loc_y2, color='b')
    plt.legend(legend_list)
    plt.title('Particle 2 (Blue) :- Strength: {:.2f}, Stddev: {:.2f}'.format(strength[1],
                                                                                                    sigma[0, 1, 0]))

    plt.subplot(2, 2, 3)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(py3[loc_y3 - 80:loc_y3 + 80], math.abs(velocities[0][0, loc_y3 - 80:loc_y3 + 80, loc_x3, 1]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(py3[loc_y3 - 80:loc_y3 + 80], math.abs(velocities[i + 1][0, loc_y3 - 80:loc_y3 + 80, loc_x3, 1]))
        legend_list.append('True: {}'.format(i + 1))
        plt.plot(py3[loc_y3 - 80:loc_y3 + 80],
                 math.abs(pred_velocites[i + 1].cpu().numpy()[0, loc_y3 - 80:loc_y3 + 80, loc_x3, 1]), '--')
        legend_list.append('Pred: {}'.format(i + 1))
    plt.axvline(x=loc_y3, color='b')
    plt.legend(legend_list)
    plt.title('Particle 2 (Blue) :- Strength: {:.2f}, Stddev: {:.2f}'.format(strength[2],
                                                                                                    sigma[0, 2, 0]))

    plt.subplot(2, 2, 4)
    legend_list = []
    plt.xlim([80.0, 176.0])
    plt.plot(py4[loc_y4 - 80:loc_y4 + 80], math.abs(velocities[0][0, loc_y4 - 80:loc_y4 + 80, loc_x4, 1]))
    legend_list.append('True: {}'.format(0))
    for i in range(2):
        plt.plot(py4[loc_y4 - 80:loc_y4 + 80], math.abs(velocities[i + 1][0, loc_y4 - 80:loc_y4 + 80, loc_x4, 1]))
        legend_list.append('True: {}'.format(i + 1))
        plt.plot(py4[loc_y4 - 80:loc_y4 + 80],
                 math.abs(pred_velocites[i + 1].cpu().numpy()[0, loc_y4 - 80:loc_y4 + 80, loc_x4, 1]), '--')
        legend_list.append('Pred: {}'.format(i + 1))
    plt.axvline(x=loc_y4, color='b')
    plt.legend(legend_list)
    plt.title('Particle 2 (Blue) :-  Strength: {:.2f}, Stddev: {:.2f}'.format(strength[3],
                                                                                                    sigma[0, 3, 0]))

    plt.suptitle('Variation of velocity-x along y-axis')
    plt.savefig(os.path.join(save_dir, 'velocity-profile-x.png'))
    plt.show()

    # plt.figure(figsize=(16, 16))
    # plt.subplot(2, 1, 1)
    # legend_list = []
    # plt.xlim([80.0, 176.0])
    # plt.plot(px1_[loc_x1 - 80:loc_x1 + 80], math.abs(velocities[0][0, loc_y1, loc_x1 - 80:loc_x1 + 80, 0]))
    # legend_list.append('True: {}'.format(0))
    # for i in range(2):
    #     plt.plot(px1_[loc_x1 - 80:loc_x1 + 80], math.abs(velocities[i + 1][0, loc_y1, loc_x1 - 80:loc_x1 + 80, 0]))
    #     legend_list.append('True: {}'.format(i + 1))
    #     plt.plot(px1_[loc_x1 - 80:loc_x1 + 80],
    #              math.abs(pred_velocites[i + 1].cpu().numpy()[0, loc_y1, loc_x1 - 80:loc_x1 + 80, 0]), '--')
    #     legend_list.append('Pred: {}'.format(i + 1))
    # plt.axvline(x=loc_x2, color='r')
    # plt.axvline(x=loc_x1, color='b')
    # plt.legend(legend_list)
    # plt.title('Particle 1 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}'.format(dist, strength[0],
    #                                                                                                 sigma[0, 0, 0]))
    # plt.subplot(2, 1, 2)
    # legend_list = []
    # plt.xlim([80.0, 176.0])
    # plt.plot(px2_[loc_x2 - 80:loc_x2 + 80], math.abs(velocities[0][0, loc_y2, loc_x2 - 80:loc_x2 + 80, 0]))
    # legend_list.append('True: {}'.format(0))
    # for i in range(2):
    #     plt.plot(px2_[loc_x2 - 80:loc_x2 + 80], math.abs(velocities[i + 1][0, loc_y2, loc_x2 - 80:loc_x2 + 80, 0]))
    #     legend_list.append('True: {}'.format(i + 1))
    #     plt.plot(px2_[loc_x2 - 80:loc_x2 + 80],
    #              math.abs(pred_velocites[i + 1].cpu().numpy()[0, loc_y2, loc_x2 - 80:loc_x2 + 80, 0]), '--')
    #     legend_list.append('Pred: {}'.format(i + 1))
    # plt.axvline(x=loc_x1, color='r')
    # plt.axvline(x=loc_x2, color='b')
    # plt.legend(legend_list)
    # plt.title('Particle 2 (Blue) :- ' + 'Distance: {:.2f}, Strength: {:.2f}, Stddev: {:.2f}'.format(dist, strength[1],
    #                                                                                                 sigma[0, 1, 0]))
    # plt.suptitle('Variation of velocity-y along x-axis')
    # plt.savefig(os.path.join(save_dir, 'velocity-profile-y.png'))
    # plt.show()
#
# /ys = [vortex_features[i][0, 0, 0].item() for i in range(NUM_TIME_STEPS + 1)]
# xs = [vortex_features[i][0, 0, 0].item() for i in range(NUM_TIME_STEPS + 1)]
#
# plt.figure()
# plt.scatter(xs, ys)
# plt.plot(xs[0], ys[0])
# plt.plot(xs[-1], ys[-1])
#
# plt.xlim([0, RESOLUTION[1]])
# plt.ylim([0, RESOLUTION[0]])
# plt.show()
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
