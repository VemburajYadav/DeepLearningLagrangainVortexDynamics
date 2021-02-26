import torch
import numpy as np
from phi.flow import *
import tensorflow as tf
from functools import partial
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
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
from visualisations.my_plot import set_size



parser = argparse.ArgumentParser()

parser.add_argument('--resolution', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--dt', type=float, default=0.2, help='time step size for simulation')
parser.add_argument('--stride', type=int, default=5, help='stride on time steps to optimize for')
parser.add_argument('--optimize_N', type=int, default=1, help='optimize for field at time step N')
parser.add_argument('--num_time_steps', type=int, default=200, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

width = 455.24408

# save_dir = '/home/vemburaj/Desktop/Report_Plots/Chapter3/'
save_dir_data = '/home/vemburaj/Desktop/TwoParticle/Case4'

plt.style.use('tex')

# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
#
if not os.path.isdir(save_dir_data):
    os.makedirs(save_dir_data)

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

RESOLUTION = opt.resolution
BOX = opt.domain

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride

location_1 = np.array([50.0, 60.0])
location_2 = np.array([70.0, 60.0])
location = np.reshape(np.stack([location_1, location_2], axis=0), (1, 2, 2))
strength = np.reshape(np.array([50.0, 50.0]), (2,))
sigma = np.reshape(np.array([5.0, 5.0]), (1, 2, 1))

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
fluid = world.add(Fluid(domain=domain, velocity=velocity_0),
                  physics=[IncompressibleFlow(),
                           lambda fluid_1, dt: fluid_1.copied_with(velocity=diffuse(fluid_1.velocity, 0.0 * dt))])

for step in range(NUM_TIME_STEPS):
    world.step(dt=opt.dt)
    velocities_.append(fluid.velocity)

velocities = []
for i in range(NUM_TIME_STEPS + 1):
    vx = np.concatenate([velocities_[i].x.data, np.zeros((1, 1, RESOLUTION[1] + 1, 1))], axis=-3)
    vy = np.concatenate([velocities_[i].y.data, np.zeros((1, RESOLUTION[0] + 1, 1, 1))], axis=-2)
    velocities.append(np.concatenate([vy, vx], axis=-1))

velocity_filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz' for i in range(NUM_TIME_STEPS + 1)]

np.savez_compressed(os.path.join(save_dir_data, 'location_000000.npz'), location.reshape((1, -1, 2)))
np.savez_compressed(os.path.join(save_dir_data, 'strength_000000.npz'), strength.reshape((1, -1, 1)))
np.savez_compressed(os.path.join(save_dir_data, 'sigma_000000.npz'), sigma.reshape((1, -1, 1)))

for i in range((NUM_TIME_STEPS + 1) // STRIDE + 1):
    np.savez_compressed(os.path.join(save_dir_data, velocity_filenames[i]),
                        StaggeredGrid(velocities[i*STRIDE]).staggered_tensor())

domain_ = Domain(RESOLUTION, box=box[0: BOX[0], 0: BOX[1]], boundaries=OPEN)
FLOW = Fluid(domain=domain_)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

loc_x1 = int(location[0, 0, 1] * RESOLUTION[1] / BOX[1])
loc_y1 = int(location[0, 0, 0] * RESOLUTION[0] / BOX[0])

loc_x2 = int(location[0, 1, 1] * RESOLUTION[1] / BOX[1])
loc_y2 = int(location[0, 1, 0] * RESOLUTION[0] / BOX[0])

py1 = points_x[0, :, loc_x1, 0]
px1 = np.array([loc_x1] * len(py1), dtype=np.float32)

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
max_val = velocities_[0].x.data[0, :, :, 0].max()
min_val = -max_val
pos = ax.imshow(velocities_[0].x.data[0, :, :, 0], vmin=min_val, vmax=max_val, cmap='RdYlBu')
# ax.scatter(location_1[1], location_1[0], color='fuchsia')
# ax.scatter(location_2[1], location_2[0], color='fuchsia')
ax.axvline(x=location_1[1], color='black')
ax.axhline(y=location_1[0], color='black')
ax.axhline(y=location_2[0], color='black')
ax.set_xlim([0, BOX[1]])
ax.set_ylim([0, BOX[0]])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation=0)
fig.colorbar(pos, ax=ax)
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_x_N2_fit.pdf'), format='pdf', bbox_inches='tight')

plt.figure()
plt.imshow(np.sqrt(velocities[0][0, :, :, 0]**2 + velocities[0][0, :, :, 1]**2))
plt.show()

plt.figure()
plt.plot(velocities_[0].x.data[0, :, loc_x1, 0])
plt.plot(velocities_[1].x.data[0, :, loc_x1, 0])
plt.plot(velocities_[2].x.data[0, :, loc_x1, 0])
plt.legend(['T0', 'T1', 'T2'])
plt.show()

particle_features_np = np.concatenate([location.reshape((1, -1, 2)),
                                       strength.reshape((1, -1, 1)),
                                       sigma.reshape((1, -1, 1))], axis=-1)

points_y_gpu = torch.tensor(points_y, device='cuda:0', dtype=torch.float32)
points_x_gpu = torch.tensor(points_x, device='cuda:0', dtype=torch.float32)

cat_y = torch.zeros((1, RESOLUTION[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, RESOLUTION[0] + 1), dtype=torch.float32, device='cuda:0')

falloff_kernel = GaussianFalloffKernelVelocity().to('cuda:0')

opt_velocities = [velocities[0]]

opt_features = [particle_features_np]

for step in range(NUM_TIME_STEPS // STRIDE):

    particle_features_pt = torch.nn.Parameter(torch.tensor(opt_features[step],
                                                           device='cuda:0',
                                                           dtype=torch.float32, requires_grad=True))

    target_vel = torch.tensor(velocities[(step + 1) * STRIDE], device='cuda:0', dtype=torch.float32)

    optimizer = Adam(params=[particle_features_pt], lr=1e-1, weight_decay=1e-5)
    lambda1 = lambda epoch: 0.95 ** epoch
    scheduler = LambdaLR(optimizer, lambda1)

    for epoch in range(1000):

        optimizer.zero_grad()
        vel_yy, vel_yx = torch.unbind(falloff_kernel(particle_features_pt, points_y_gpu), dim=-1)
        vel_xy, vel_xx = torch.unbind(falloff_kernel(particle_features_pt, points_x_gpu), dim=-1)

        vel_y = torch.cat([vel_yy, cat_y], dim=-1)
        vel_x = torch.cat([vel_xx, cat_x], dim=-2)

        pred_vel = torch.stack([vel_y, vel_x], dim=-1)

        loss = torch.sum((pred_vel - target_vel)**2) #/ torch.sum(target_vel**2)
        loss.backward()
        optimizer.step()
        # scheduler.step(epoch=epoch)

        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

    opt_velocities.append(pred_vel.detach().clone())
    opt_features.append(particle_features_pt.detach().clone().cpu().numpy())

plt.style.use('seaborn')
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.plot(velocities_[0].x.data[0, :, loc_x1, 0])
legend_list = [r'Actual: $t = {}s$'.format(0)]
for step in range(NUM_TIME_STEPS // STRIDE):
    ax.plot(velocities_[(step + 1) * STRIDE].x.data[0, :, loc_x1, 0])
    pred_vel_y, pred_vel_x = torch.unbind(opt_velocities[step + 1], dim=-1)
    pred_vel_y_np = pred_vel_y.cpu().numpy()[:, :, :-1]
    pred_vel_x_np = pred_vel_x.cpu().numpy()[:, :-1, :]
    ax.plot(pred_vel_x_np[0, :, loc_x1], linestyle='dashed')
    legend_list.append(r'Actual: $t = {}s$'.format(step+1))
    legend_list.append(r'Vortex Fit: $t = {}s$'.format(step+1))
# ax.scatter(location_1[0]-0.5, 0.0, color='r')
# ax.scatter(location_2[0]-0.5, 0.0, color='r')
ax.axvline(x=location_1[0], color='red')
ax.axvline(x=location_2[0], color='red')

ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$u$', rotation=0)
plt.legend(legend_list)
plt.show()
# fig.savefig(os.path.join(save_dir, 'velocity_N2_fit.pdf'), format='pdf')

opt_features_all = np.stack(opt_features, axis=-1)

np.savez_compressed(os.path.join(save_dir_data, 'opt_features.npz'), opt_features_all)

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 0, 2, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\Gamma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'strength_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 0, 3, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\sigma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'sigma_p0_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 0, 0, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$y_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'yloc_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 0, 1, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$x_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'xloc_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 1, 2, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\Gamma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'strength_p1_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 1, 3, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\sigma_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'sigma_p1_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 1, 0, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$y_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'yloc_p1_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features_all[0, 1, 1, :])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$x_p(t)$', rotation=0)
plt.show()
# fig.savefig(os.path.join(save_dir, 'xloc_p1_N2_fit.pdf'), format='pdf')
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
