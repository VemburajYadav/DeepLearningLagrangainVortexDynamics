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
from visualisations.my_plot import set_size


parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/'
                                                     '/data/p100_gaussian_dataset_viscous_120x120_4000/train/sim_002321',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='T1_exp_weight_1.0_depth_5_100_batch_32_lr_1e-3_l2_1e-5_r120_4000_2',
                    help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--num_time_steps', type=int, default=2, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='GaussianVorticity', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

opt = parser.parse_args()

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

# location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
# strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
# sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']
# viscosity = np.load(os.path.join(case_dir, 'viscosity.npz'))['arr_0']
#
# location_1 = np.array([40.0, 50.0])
# location_2 = np.array([60.0, 50.0])
viscosity = np.linspace(0.0, 2.0, 100)
nbatch = viscosity.shape[0]
location = np.tile(np.array([60.0, 60.0]).reshape((1, 1, 2)), (nbatch, 1, 1))
sigma = np.tile(np.array([5.0]).reshape((1, 1, 1)), (nbatch, 1, 1))
strength = np.tile(np.array([100.0]).reshape((1, 1, 1)), (nbatch, 1, 1))

# location = np.stack([location_1, location_2], axis=-1)
# viscosity = np.arange(0, 0.5, 0.05)

nparticles = location.shape[1]

# velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
#               for i in range(0, (NUM_TIME_STEPS + 1) * opt.stride, opt.stride)]

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data


loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')
visc_gpu = torch.tensor(viscosity, dtype=torch.float32, device='cuda:0')

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)

points_y = torch.tensor(points_y, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(points_x, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

v0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
u0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')

logs_dir = os.path.join('../logs_p10_gauss_viscous', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

VortexNet = MultiStepViscousVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
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

loc_y, loc_x =torch.unbind(loc_gpu, dim=-1)

zeros = torch.zeros(size=(nbatch, 12), dtype=torch.float32, device='cuda:0')
input = torch.cat([tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), zeros, visc_gpu.view(-1, 1)], dim=-1)

out = VortexNet.single_step_net.net(input)

dy, dx, dtau, dsig = torch.unbind(out, dim=-1)

y_new = loc_y.view(-1) + dy * 0.1
x_new = loc_x.view(-1) + dx * 0.1
tau_new = tau_gpu.view(-1) + dtau * 0.1
sig_new = sig_gpu.view(-1) + dsig * 0.1

width = 455.24408

save_dir = '/home/vemburaj/Desktop/Ppt_Plots/Results/'

plt.style.use('tex')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# plt.style.use('seaborn')
fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(viscosity, tau_new.detach().clone().cpu().numpy())
ax.set_xlabel(r'viscosity $\nu$')
ax.set_ylabel(r'$\Gamma_p$', rotation=0)
ax.legend([r'$\Gamma_p={}$, $\sigma_p={}$'.format(strength[0,0,0], sigma[0,0,0])])
ax.grid(True)
plt.show()
# fig.savefig(os.path.join(save_dir, 'strength_p0_N2_fit.pdf'), format='pdf')


fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
ax.plot(viscosity, sig_new.detach().clone().cpu().numpy())
ax.set_xlabel(r'viscosity $\nu$')
ax.set_ylabel(r'$\sigma_p$', rotation=0)
ax.legend([r'$\Gamma_p={}$, $\sigma_p={}$'.format(strength[0,0,0], sigma[0,0,0])])
ax.grid(True)
plt.show()

