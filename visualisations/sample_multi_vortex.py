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
parser.add_argument('--offset', type=list, default=[40, 40], help='neglect regions near boundaries of the '
                                                                  'domain (as list: [24, 24])')
parser.add_argument('--strength_range', type=list, default=[20.0, 100.0], help='range for strength sampling')
parser.add_argument('--sigma_range', type=list, default=[2.0, 10.0], help='range for core ize sampling')
parser.add_argument('--num_time_steps', type=int, default=25, help='number of time steps to adfvance the simulation '
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

OFFSET = opt.offset
RESOLUTION = opt.resolution
BOX = opt.domain
SAMPLE_RES = [BOX[0] - 2 * OFFSET[0], BOX[1] - 2 * OFFSET[1]]
STRENGTH_RANGE = opt.strength_range
SIGMA_RANGE = opt.sigma_range
NUM_TIME_STEPS = opt.num_time_steps

NPARTICLES = 10

ycoords = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[0] + OFFSET[0]
xcoords = np.random.random_sample(size=(NPARTICLES)) * SAMPLE_RES[1] + OFFSET[1]

np.random.shuffle(ycoords)
np.random.shuffle(xcoords)

location = np.reshape(np.stack([ycoords, xcoords], axis=0), (1, -1, 2))

sigma = np.reshape(np.random.random_sample(size=(NPARTICLES)) * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) + SIGMA_RANGE[0], (1, -1, 1))

fac = np.random.random_sample(size=NPARTICLES) * 10 + 5

rands = np.array([-1] * (NPARTICLES // 2) + [1] * (NPARTICLES // 2))
np.random.shuffle(rands)

strengths = fac * sigma.reshape((-1)) * rands
# strengths_pos = np.random.random_sample(size=(NPARTICLES // 2)) * (STRENGTH_RANGE[1] - STRENGTH_RANGE[0]) + STRENGTH_RANGE[0]
# strengths_neg = -(np.random.random_sample(size=(NPARTICLES // 2)) * (STRENGTH_RANGE[1] - STRENGTH_RANGE[0]) + STRENGTH_RANGE[0])
# strengths = np.concatenate([strengths_neg, strengths_pos])
# np.random.shuffle(strengths)
strength = np.reshape(strengths, (-1,))

# location = np.array([[[61.15070502, 44.71726535],
#         [65.29162084, 68.7584254 ],
#         [56.45237209, 32.50151245],
#         [32.86068525, 37.0007516 ],
#         [55.97778915, 36.194138  ],
#         [55.55060348, 61.57875753],
#         [35.86446531, 43.55183681],
#         [41.77849965, 63.39204938],
#         [53.55559502, 40.19090948],
#         [46.50963625, 30.76135377]]])
#
# strength = np.array([-39.84272255,  88.50597011,  23.41458642, -89.15081008,
#         90.63114055, -87.852772  ,  45.70981426,  76.95656346,
#        -32.70897278, -69.87962647])
#
# sigma = np.array([[[5.07109363],
#         [9.85502543],
#         [9.47074758],
#         [7.4990762 ],
#         [9.31928655],
#         [8.33432468],
#         [5.98056061],
#         [9.66075559],
#         [3.8633071 ],
#         [7.30133859]]])

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
fluid = world.add(Fluid(domain=domain, velocity=velocity_0), physics=IncompressibleFlow())

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

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
max_val = np.abs(velocities_[0].x.data[0, :, :, 0]).max()
min_val = -max_val
pos = ax.imshow(velocities_[0].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, BOX[1]])
ax.set_ylim([0, BOX[0]])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_x_N20_T0.pdf'), format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities_[5].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, BOX[1]])
ax.set_ylim([0, BOX[0]])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_x_N20_T1.pdf'), format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities_[15].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, BOX[1]])
ax.set_ylim([0, BOX[0]])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_x_N20_T2.pdf'), format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
pos = ax.imshow(velocities_[20].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax.set_xlim([0, BOX[1]])
ax.set_ylim([0, BOX[0]])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_x_N20_T3.pdf'), format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 4, figsize=set_size(width, fraction=1, subplots=(1, 4)))
ax[0].imshow(velocities_[0].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax[1].imshow(velocities_[5].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
ax[2].imshow(velocities_[10].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
pos = ax[3].imshow(velocities_[15].x.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
plt.show()

loc_x1 = int(location[0, 0, 1] * RESOLUTION[1] / BOX[1])
loc_y1 = int(location[0, 0, 0] * RESOLUTION[0] / BOX[0])

loc_x2 = int(location[0, 1, 1] * RESOLUTION[1] / BOX[1])
loc_y2 = int(location[0, 1, 0] * RESOLUTION[0] / BOX[0])

plt.figure()
plt.plot(velocities_[0].x.data[0, :, loc_x1, 0])
plt.plot(velocities_[5].x.data[0, :, loc_x1, 0])
plt.plot(velocities_[10].x.data[0, :, loc_x1, 0])
plt.plot(velocities_[15].x.data[0, :, loc_x1, 0])
plt.plot(velocities_[20].x.data[0, :, loc_x1, 0])
plt.legend(['T0', 'T1', 'T2', 'T3', 'T4'])
plt.show()

# fig.savefig(os.path.join(save_dir, 'velocity_x_N20_T3.pdf'), format='pdf', bbox_inches='tight')
# fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
# max_val = np.abs(velocities_[0].y.data[0, :, :, 0]).max()
# min_val = -max_val
# pos = ax.imshow(velocities_[0].y.data[0, :, :, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
# ax.set_xlim([0, BOX[1]])
# ax.set_ylim([0, BOX[0]])
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$', rotation=0)
# fig.colorbar(pos, ax=ax)
# plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_y_N20_1.pdf'), format='pdf', bbox_inches='tight')
#

# fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
# pos = ax.imshow(np.sqrt(velocities[0][0, :, :, 0]**2 + velocities[0][0, :, :, 1]**2)[:-1, :-1])
# ax.set_xlim([0, BOX[1]])
# ax.set_ylim([0, BOX[0]])
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$', rotation=0)
# fig.colorbar(pos, ax=ax)
# plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_total_N20_1.pdf'), format='pdf', bbox_inches='tight')


# plt.figure(figsize=set_size(width, fraction=1, subplots=(1, 1)))
# plt.plot(velocities_[0].x.data[0, :, loc_x, 0])
# plt.plot(velocities_[10].x.data[0, :, loc_x, 0])
# plt.plot(velocities_[50].x.data[0, :, loc_x, 0])
# plt.plot(velocities_[60].x.data[0, :, loc_x, 0])
#
# plt.legend(['T0', 'T1', 'T5', 'T6'])
# plt.show()

vel_cg_0 = velocities_[0].at(FLOW.density)
gx, gy = vel_cg_0.points.data[0, :, :, 1], vel_cg_0.points.data[0, :, :, 0]
vx, vy = vel_cg_0.data[0, :, :, 1], vel_cg_0.data[0, :, :, 0]

vy = vy[:, ::-1]
#
# fig.savefig(os.path.join(save_dir, 'velocity_vector.pdf'), format='pdf', bbox_inches='tight')

particle_features_np = np.concatenate([location.reshape((1, -1, 2)),
                                       strength.reshape((1, -1, 1)),
                                       sigma.reshape((1, -1, 1))], axis=-1)

particle_features_pt = torch.tensor(particle_features_np, dtype=torch.float32, device='cuda:0')

points_cg = torch.tensor(FLOW.density.points.data, dtype=torch.float32, device='cuda:0')
falloff_kernel = GaussianFalloffKernelVorticity().to('cuda:0')

vorticity_cg = falloff_kernel(particle_features_pt, points_cg)

# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
# max_val = np.abs(vorticity_cg[0, :, :, 0].cpu().numpy()).max()
# min_val = -max_val
# pos = ax.imshow(vorticity_cg[0, :, :, 0].cpu().numpy(), cmap='RdBu', vmin=min_val, vmax=max_val)
# ax.set_xlim([0, BOX[1]])
# ax.set_ylim([0, BOX[0]])
# plt.scatter(100.0 -0.5, 100.0-0.5, color='r')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$', rotation=0)
# ax.set_ylim(ax.get_ylim()[::-1])
# fig.colorbar(pos, ax=ax)
# plt.show()
# fig.savefig(os.path.join(save_dir, 'vorticity_N20_1.pdf'), format='pdf', bbox_inches='tight')

plt.style.use('seaborn')
# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
# plt.quiver(gx,gy,vx,vy, np.arctan2(vy, vx), angles='xy', scale_units='xy', scale=1)
# plt.scatter(100.0, 100.0, color='r')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$', rotation=0)
# ax.set_ylim(ax.get_ylim()[::-1])
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
for i in range(NPARTICLES):
    ax.scatter(location[0, i, 1], location[0, i, 0], color=numpy.random.rand(3,))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
ax.set_xlim(0, BOX[1])
ax.set_ylim(0, BOX[0])
ax.set_ylim(ax.get_ylim()[::-1])
ax.grid(True)
plt.show()
# fig.savefig(os.path.join(save_dir, 'particle_locations_N20_T0.pdf'), format='pdf', bbox_inches='tight')

# fig, ax = plt.subplots(5, 2, sharex=True, sharey=True, figsize=set_size(width, fraction=1, subplots=(5, 2)))
# for i in range(5):
#     for j in range(2):
#         p_id = i * 2 + j
#         loc_y_p = int(location[0, p_id, 0] * RESOLUTION[0] / BOX[0])
#         loc_x_p = int(location[0, p_id, 1] * RESOLUTION[1] / BOX[1])
#         ax[i, j].plot(velocities_[0].y.data[0, loc_y_p, :, 0])
#         ax[i, j].axvline(x=loc_x_p, color='red')
# plt.show()
# fig.text(0.5, 0.04, r'$x$', ha='center')
# fig.text(0.04, 0.5, r'$v$', va='center', rotation=0)
# fig.savefig(os.path.join(save_dir, 'velocity_profiles_N20_1.pdf'), format='pdf', bbox_inches='tight')

# fig, ax = plt.subplots(1, 2, figsize=set_size(width, fraction=1.0, subplots=(1, 2)))
# ax[0].plot(vel_cg_0.points.data[0, 80:120, 100, 0], vorticity_cg[0, 80:120, 100, 0].cpu().numpy())
# ax[0].set_xlabel(r'$x$')
# ax[0].set_ylabel(r'$\omega $', rotation=0)
# ax[0].set_title('(a) Variation of vorticity along x-axis', y=-0.1)
# ax[1].plot(vel_cg_0.points.data[0, 80:120, 100, 0], vel_cg_0.data[0, 80:120, 100, 1])
# ax[1].scatter(100, 0, color='r')
# ax[1].set_xlabel(r'$x$')
# ax[1].set_ylabel(r'$v$', rotation=0)
# ax[1].set_title('(b) Variation of y-component of velocity along x-axis', y=-0.1)
# plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_vorticity_profile.pdf'), format='pdf')

# fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1.0, subplots=(1, 1)))
# ax.plot(vel_cg_0.points.data[0, 80:120, 100, 0], vorticity_cg[0, 80:120, 100, 0].cpu().numpy())
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$\omega $', rotation=0)
# ax.set_title('(a) Variation of vorticity along x-axis', y=-0.1)
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
