import torch
import numpy as np
from phi.flow import *
import tensorflow as tf
from functools import partial
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from visualisations.my_plot import set_size
import argparse
import os
import matplotlib.animation as animation
from core.networks import *
from core.custom_functions import *
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2


parser = argparse.ArgumentParser()

parser.add_argument('--resolution', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--num_time_steps', type=int, default=10, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# mpl.rcParams['text.usetex'] = True
# Using seaborn's style
# plt.style.use('dark_background')
width = 455.24408

save_dir = '/home/vemburaj/Desktop/Report_Plots/Chapter3/'

plt.style.use('tex')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

RESOLUTION = opt.resolution
BOX = opt.domain

NUM_TIME_STEPS = opt.num_time_steps

loc = [60.0, 60.0]
location = np.array(loc).reshape((1, 1, 2))

strength = np.array([20.0])
sigma = np.array([2.0]).reshape((1, 1, 1))

NPARTICLES = location.shape[1]

def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    falloff_1 = (math.exp(- sq_distance / sigma ** 2)) / math.sqrt(sq_distance)
    falloff_2 = (1.0 - math.exp(- sq_distance / sigma ** 2)) / (2.0 * np.pi * sq_distance)

    return falloff_2

domain = Domain(RESOLUTION, box=box[0: BOX[0], 0: BOX[1]], boundaries=OPEN)
FLOW_REF = Fluid(domain)

vorticity = AngularVelocity(location=location,
                            strength=strength,
                            falloff=partial(gaussian_falloff, sigma=sigma))

velocity_0 = vorticity.at(FLOW_REF.velocity)
velocities_ = [velocity_0]

FLOW = Fluid(domain=domain, velocity=velocity_0)
fluid = world.add(Fluid(domain=domain, velocity=velocity_0), physics=[IncompressibleFlow(),
                           lambda fluid_1, dt: fluid_1.copied_with(velocity=diffuse(fluid_1.velocity, 0.0 * dt, substeps=5))])

for step in range(NUM_TIME_STEPS):
    world.step(dt=0.2)
    velocities_.append(fluid.velocity)

velocities = []
for i in range(NUM_TIME_STEPS + 1):
    vx = np.concatenate([velocities_[i].x.data, np.zeros((1, 1, RESOLUTION[1] + 1, 1))], axis=-3)
    vy = np.concatenate([velocities_[i].y.data, np.zeros((1, RESOLUTION[0] + 1, 1, 1))], axis=-2)
    velocities.append(np.concatenate([vy, vx], axis=-1))

domain_ = Domain(RESOLUTION, box=box[0: BOX[0], 0: BOX[1]], boundaries=OPEN)
FLOW_ = Fluid(domain=domain_)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

loc_x = int(location[0, 0, 1] * RESOLUTION[1] / BOX[1])
loc_y = int(location[0, 0, 0] * RESOLUTION[0] / BOX[0])

py = points_x[0, :, loc_x, 0]
px = np.array([loc_x] * len(py), dtype=np.float32)

# plt.figure()
# plt.imshow(velocities_[0].x.data[0, :, :, 0], cmap='RdYlBu')
# plt.show()
#
# plt.figure()
# plt.imshow(np.sqrt(velocities[0][0, :, :, 0]**2 + velocities[0][0, :, :, 1]**2))
# plt.show()

# plt.style.use('seaborn')
fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(velocities_[0].x.data[0, :, loc_x, 0])
ax.plot(velocities_[5].x.data[0, :, loc_x, 0])
ax.plot(velocities_[10].x.data[0, :, loc_x, 0])
ax.scatter(59.5, 0, color='r')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$v$', rotation=0)
plt.legend([r'$t=0s$', r'$t=1s$'])
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_profile_dt0.1s.pdf'), format='pdf')
# plt.show()
#
# vel_cg_0 = velocities_[0].at(FLOW.density)
# gx, gy = vel_cg_0.points.data[0, 80:120, 80:120, 1], vel_cg_0.points.data[0, 80:120, 80:120, 0]
# vx, vy = vel_cg_0.data[0, 80:120, 80:120, 1], vel_cg_0.data[0, 80:120, 80:120, 0]
#
# vy = vy[:, ::-1]

# fig.savefig(os.path.join(save_dir, 'velocity_vector.pdf'), format='pdf', bbox_inches='tight')

# particle_features_np = np.concatenate([location.reshape((1, -1, 2)),
#                                        strength.reshape((1, -1, 1)),
#                                        sigma.reshape((1, -1, 1))], axis=-1)
#
# particle_features_pt = torch.tensor(particle_features_np, dtype=torch.float32, device='cuda:0')
#
# points_cg = torch.tensor(FLOW.density.points.data, dtype=torch.float32, device='cuda:0')
# falloff_kernel = GaussianFalloffKernelVorticity().to('cuda:0')
#
# vorticity_cg = falloff_kernel(particle_features_pt, points_cg)

# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
# ax.set_xlim([80, 120])
# ax.set_ylim([80, 120])
# pos = ax.imshow(vorticity_cg[0, :, :, 0].cpu().numpy(), cmap='viridis')
# plt.scatter(100.0 -0.5, 100.0-0.5, color='r')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$', rotation=0)
# ax.set_ylim(ax.get_ylim()[::-1])
# fig.colorbar(pos, ax=ax)
# plt.show()
# fig.savefig(os.path.join(save_dir, 'vorticity.pdf'), format='pdf', bbox_inches='tight')

# plt.style.use('seaborn')
# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
# plt.quiver(gx,gy,vx,vy)
# plt.scatter(100.0, 100.0, color='r')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$', rotation=0)
# ax.set_ylim(ax.get_ylim()[::-1])
# plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_vector.pdf'), format='pdf', bbox_inches='tight')

# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1.0, subplots=(1, 1)))
# ax.plot(vel_cg_0.points.data[0, 80:120, 100, 0], vorticity_cg[0, 80:120, 100, 0].cpu().numpy())
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$\omega $', rotation=0)
# ax.plot(vel_cg_0.points.data[0, 80:120, 100, 0], vel_cg_0.data[0, 80:120, 100, 1])
# plt.show()
# fig.savefig(os.path.join(save_dir, 'vorticity_profile.pdf'), format='pdf')

# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1.0, subplots=(1, 1)))
# ax.plot(vel_cg_0.points.data[0, 80:120, 100, 0], vel_cg_0.data[0, 80:120, 100, 1])
# ax.scatter(100, 0, color='r')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$v$', rotation=0)
# plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_profile.pdf'), format='pdf')

# min_val = total_velocities[0].min()
# max_val = total_velocities[0].max()
#
# for step in range(NUM_TIME_STEPS + 1):
#     fig, axs = plt.subplots(1, 3, figsize=(24, 10))
#
#     ax = axs[0]
#     pcm = ax.imshow(total_velocities[step].cpu().numpy()[0, 80:180, 80:180], vmin=min_val, vmax=max_val)
#     ax.set_title('Simulation')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(pcm, cax=cax)
#
#     ax = axs[1]
#     pcm = ax.imshow(total_velocities_pred[step].cpu().numpy()[0, 80:180, 80:180], vmin=min_val, vmax=max_val)
#     ax.set_title('Neural Network')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(pcm, cax=cax)
#
#     ax = axs[2]
#     pcm = ax.imshow(error_map[step].cpu().numpy()[0, 80:180, 80:180]**2, cmap='Greys')
#     ax.set_title('Error Map: {:.2f}'.format(np.sum(error_map[step].cpu().numpy()[0, 80:180, 80:180]**2)))
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(pcm, cax=cax)
#
#     fig.suptitle(' Strength: {:.2f} \n Core Size:'
#                  ' {:.2f} \n Velocity - Time Step: {}'.format(strength[0], sigma[0, 0, 0], step))
#
#     filename = os.path.join(save_dir, 'vis_' + '0' * (6 - len(str(step))) + str(step) + '.png')
#     plt.savefig(filename)
#
# plt.figure(figsize=(16, 10))
# legend_list = []
# for i in range(6):
#     plt.plot(math.abs(velocities[i][0, loc_y - 20:loc_y + 20, loc_x, 1]))
#     legend_list.append('True: {}'.format(i * opt.stride))
#     if i > 0:
#         plt.plot(math.abs(pred_velocites[i].cpu().numpy()[0, loc_y - 20:loc_y + 20, loc_x, 1]), '--')
#         legend_list.append('Pred: {}'.format(i * opt.stride))
# plt.legend(legend_list)
# plt.title("Variation of velocity-x along y-axis \n 'Strength: {:.2f}, "
#           "Stddev: {:.2f}".format(strength[0], sigma[0, 0, 0]))
# plt.savefig(os.path.join(save_dir, 'velocity-profile-y.png'))
# plt.show()
