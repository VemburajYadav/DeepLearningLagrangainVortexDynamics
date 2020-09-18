import numpy as np
import torch
import torch.nn.functional as F
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from phi.flow import Domain, Fluid, OPEN
from core.networks import *
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[128, 128], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--load_weights_ex', type=str, default='T5_off_splus_weight_0.9_depth_3_512_lr_1e-4_l2_1e-5_T1_init', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=3, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=512, help='number of neurons in hidden layers')
parser.add_argument('--num_time_steps', type=int, default=5, help='number of time steps to make predictions for')
parser.add_argument('--kernel', type=str, default='offset-gaussian', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

opt = parser.parse_args()

STRENGTH_LIST = [1.6, 1.6, 1.6, 1.6, 1.6]
SIGMA_LIST = [10.0, 20.0, 30.0, 40.0, 50.0]
location = [35.8, 46.7] * len(STRENGTH_LIST)

logs_dir = os.path.join('../logs', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

VortexNet = MultiStepVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                   kernel=opt.kernel, num_steps=opt.num_time_steps, norm_mean=MEAN, norm_stddev=STDDEV)

VortexNet.single_step_net.load_state_dict(params)
VortexNet.to('cuda:0')
VortexNet.eval()

loc_np = np.array(location, dtype=np.float32).reshape((len(STRENGTH_LIST), 2))
tau_np = np.array(STRENGTH_LIST, dtype=np.float32).reshape(len(STRENGTH_LIST), 1)
sig_np = np.array(SIGMA_LIST, dtype=np.float32).reshape(len(SIGMA_LIST), 1)

loc0 = torch.tensor(loc_np, device='cuda:0')
tau0 = torch.tensor(tau_np, device='cuda:0')
sig0 = torch.tensor(sig_np, device='cuda:0')
v0 = torch.zeros((len(STRENGTH_LIST), 1), dtype=torch.float32, device='cuda:0')
u0 = torch.zeros((len(STRENGTH_LIST), 1), dtype=torch.float32, device='cuda:0')

if opt.kernel == 'gaussian':
    net_inp = torch.cat([loc0, tau0, sig0, v0, u0], dim=-1)
elif opt.kernel == 'offset-gaussian':
    off0 = torch.zeros((len(STRENGTH_LIST), 1), dtype=torch.float32, device='cuda:0')
    sigl0 = torch.zeros((len(STRENGTH_LIST), 1), dtype=torch.float32, device='cuda:0')
    net_inp = torch.cat([loc0, tau0, sig0, v0, u0, off0, sigl0], dim=-1)

with torch.no_grad():
    vortex_features = VortexNet(net_inp)

feature_tensor = torch.stack(vortex_features, dim=-1).cpu().numpy()
taus = feature_tensor[:, 2]
sigs = feature_tensor[:, 3]

legend_list = []
plt.figure()
for case in range(len(STRENGTH_LIST)):
    plt.plot(taus[case, :])
    legend_list.append('Strength: {:.2f}, Core Size: {:.2f}'.format(STRENGTH_LIST[case], SIGMA_LIST[case]))
plt.legend(legend_list)
plt.title('Evolution of Vortex Strength')
plt.show()

plt.figure()
for case in range(len(STRENGTH_LIST)):
    plt.plot(sigs[case, :])
plt.legend(legend_list)
plt.title('Evolution of Vortex Core Size')
plt.show()

if opt.kernel == 'offset-gaussian':
    sigls = feature_tensor[:, 7]
    plt.figure()
    for case in range(len(STRENGTH_LIST)):
        plt.plot(sigls[case, :])
        legend_list.append('Strength: {:.2f}, Core Size: {:.2f}'.format(STRENGTH_LIST[case], SIGMA_LIST[case]))
    plt.legend(legend_list)
    plt.title('Evolution of left Standard deviations')
    plt.show()

    sigrs = sigs - sigls
    plt.figure()
    for case in range(len(STRENGTH_LIST)):
        plt.plot(sigrs[case, :])
        legend_list.append('Strength: {:.2f}, Core Size: {:.2f}'.format(STRENGTH_LIST[case], SIGMA_LIST[case]))
    plt.legend(legend_list)
    plt.title('Evolution of right Standard deviations')
    plt.show()

    offs = feature_tensor[:, 6]
    plt.figure()
    for case in range(len(STRENGTH_LIST)):
        plt.plot(offs[case, :])
        legend_list.append('Strength: {:.2f}, Core Size: {:.2f}'.format(STRENGTH_LIST[case], SIGMA_LIST[case]))
    plt.legend(legend_list)
    plt.title('Evolution of Offsets')
    plt.show()




