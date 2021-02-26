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
parser.add_argument('--num_time_steps', type=int, default=20, help='number of time steps to adfvance the simulation '
                                                                   'for each sample')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')

width = 455.24408

save_dir_data = '/home/vemburaj/Desktop/TwoParticle/Case1'
save_dir = '/home/vemburaj/Desktop/TwoParticle/Case1_Plots'

plt.style.use('tex')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

RESOLUTION = opt.resolution
STRIDE = opt.stride
NUM_TIME_STEPS = opt.num_time_steps

velocities = [np.load(os.path.join(save_dir_data, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

pred_velocities = [np.load(os.path.join(save_dir_data, 'pred_velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

location = np.load(os.path.join(save_dir_data, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(save_dir_data, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(save_dir_data, 'sigma_000000.npz'))['arr_0']

opt_features = np.load(os.path.join(save_dir_data, 'opt_features.npz'))['arr_0']
nn_features = np.load(os.path.join(save_dir_data, 'nn_features.npz'))['arr_0']

# location_1 = np.array([40.0, 60.0])
# location_2 = np.array([80.0, 60.0])
# location = np.reshape(np.stack([location_1, location_2], axis=0), (1, 2, 2))
# strength = np.reshape(np.array([100.0, -100.0]), (2,))
# sigma = np.reshape(np.array([5.0, 5.0]), (1, 2, 1))

NPARTICLES = location.shape[1]

def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    falloff_1 = (math.exp(- sq_distance / sigma ** 2)) / math.sqrt(sq_distance)
    falloff_2 = (1.0 - math.exp(- sq_distance / sigma ** 2)) / (2.0 * np.pi * sq_distance)

    return falloff_2



max_val = np.abs(velocities[0][0, :-1, :-1, 1]).max()
min_val = -max_val

for i in range(NUM_TIME_STEPS + 1):
    fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
    pos = ax.imshow(velocities[i][0, :-1, :-1, 1], cmap='RdBu', vmin=min_val, vmax=max_val)
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    fig.colorbar(pos, ax=ax)
    plt.show()
    fig.savefig(os.path.join(save_dir, 'velocity_x_' + '0' * (6 - len(str(i))) + str(i) + '.pdf'), format='pdf',
                bbox_inches='tight')

    fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
    pos = ax.imshow(pred_velocities[i][0, :-1, :-1, 1], cmap='RdBu', vmin=min_val, vmax=max_val)
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    fig.colorbar(pos, ax=ax)
    plt.show()
    fig.savefig(os.path.join(save_dir, 'pred_velocity_x_' + '0' * (6 - len(str(i))) + str(i) + '.pdf'), format='pdf',
                bbox_inches='tight')

max_val = np.abs(velocities[0][0, :-1, :-1, 0]).max()
min_val = -max_val

for i in range(NUM_TIME_STEPS + 1):
    fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
    pos = ax.imshow(velocities[i][0, :-1, :-1, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    fig.colorbar(pos, ax=ax)
    plt.show()
    fig.savefig(os.path.join(save_dir, 'velocity_y_' + '0' * (6 - len(str(i))) + str(i) + '.pdf'), format='pdf',
                bbox_inches='tight')

    fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
    pos = ax.imshow(pred_velocities[i][0, :-1, :-1, 0], cmap='RdBu', vmin=min_val, vmax=max_val)
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    fig.colorbar(pos, ax=ax)
    plt.show()
    fig.savefig(os.path.join(save_dir, 'pred_velocity_y_' + '0' * (6 - len(str(i))) + str(i) + '.pdf'),
                format='pdf',
                bbox_inches='tight')

max_val = np.abs((velocities[0][0, :-1, :-1, 0]**2 +
                 velocities[0][0, :-1, :-1, 1]**2)**0.5).max()

for i in range(NUM_TIME_STEPS + 1):
    fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
    pos = ax.imshow((velocities[i][0, :-1, :-1, 0]**2 +
                 velocities[i][0, :-1, :-1, 1]**2)**0.5, cmap='viridis', vmin=0, vmax=max_val)
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    fig.colorbar(pos, ax=ax)
    plt.show()
    fig.savefig(os.path.join(save_dir, 'velocity_total_' + '0' * (6 - len(str(i))) + str(i) + '.pdf'), format='pdf',
                bbox_inches='tight')


    fig, ax = plt.subplots(figsize=set_size(width, fraction=1, subplots=(1, 1)))
    pos = ax.imshow((pred_velocities[i][0, :-1, :-1, 0]**2 +
                     pred_velocities[i][0, :-1, :-1, 1]**2)**0.5, cmap='viridis', vmin=0, vmax=max_val)
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    fig.colorbar(pos, ax=ax)
    plt.show()
    fig.savefig(os.path.join(save_dir, 'pred_velocity_total_' + '0' * (6 - len(str(i))) + str(i) + '.pdf'), format='pdf',
                bbox_inches='tight')


plt.style.use('seaborn')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 0, 2, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 0, 2, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\Gamma_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'strength_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 0, 3, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 0, 3, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\sigma_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'sigma_p0_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 0, 0, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 0, 0, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$y_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'yloc_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 0, 1, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 0, 1, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$x_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'xloc_p0_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 1, 2, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 1, 2, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\Gamma_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'strength_p1_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 1, 3, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 1, 3, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$\sigma_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'sigma_p1_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 1, 0, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 1, 0, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$y_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'yloc_p1_N2_fit.pdf'), format='pdf')

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(opt_features[0, 1, 1, :(NUM_TIME_STEPS+1)])
ax.plot(nn_features[0, 1, 1, :(NUM_TIME_STEPS+1)])
ax.set_xlabel(r'time $t(s)$')
ax.set_ylabel(r'$x_p(t)$', rotation=0)
plt.legend([r'Vortex-Fit', r'Neural Network'])
plt.show()
fig.savefig(os.path.join(save_dir, 'xloc_p1_N2_fit.pdf'), format='pdf')