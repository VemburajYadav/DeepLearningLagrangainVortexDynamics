import numpy as np
import torch
import torch.nn.functional as F
from core.custom_functions import particle_vorticity_to_velocity
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from phi.flow import Domain, Fluid, OPEN
from core.networks import SimpleNN
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[128, 128], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--load_weights_ex', type=str, default='train_demo_128x128_8000_T20_init_T5_lr_1e-4_weighted', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=3, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=512, help='number of neurons in hidden layers')
parser.add_argument('--num_time_steps', type=int, default=100, help='number of time steps to make predictions for')

opt = parser.parse_args()

STRENGTH_LIST = [1.5, 1.6, 1.7, 1.8, 1.9]
SIGMA_LIST = [40.0, 40.0, 40.0, 40.0, 40.0]

logs_dir = os.path.join('../logs', opt.load_weights_ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

checkpoints_files = os.listdir(os.path.join(ckpt_dir))
epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])

params = torch.load(ckpt_file)['model_state_dict']

network = SimpleNN(depth=opt.depth, hidden_units=opt.hidden_units)
network.load_state_dict(params)
network.to('cuda:0')
network.eval()

tau_np = np.array(STRENGTH_LIST, dtype=np.float32).reshape(len(STRENGTH_LIST), 1)
sig_np = np.array(SIGMA_LIST, dtype=np.float32).reshape(len(SIGMA_LIST), 1)

tau0 = torch.tensor(tau_np, device='cuda:0')
sig0 = torch.tensor(sig_np, device='cuda:0')
v0 = torch.zeros((len(STRENGTH_LIST), 1), dtype=torch.float32, device='cuda:0')
u0 = torch.zeros((len(STRENGTH_LIST), 1), dtype=torch.float32, device='cuda:0')

net_inp = torch.cat([tau0, sig0, v0, u0], dim=-1)

inp_list = [net_inp]

with torch.no_grad():

    for step in range(opt.num_time_steps):

        tau, sig, v, u = torch.unbind(inp_list[step], dim=-1)
        net_output = network(inp_list[step])
        dy, dx, dtau, dsig = torch.unbind(net_output, dim=-1)

        inp_list.append(torch.stack([tau + dtau, sig + F.softplus(dsig), dy * 0.1, dx * 0.1], dim=-1))

feature_tensor = torch.stack(inp_list, dim=0).cpu().numpy()

taus = feature_tensor[:, :, 0]
sigs = feature_tensor[:, :, 1]

legend_list = []
plt.figure()
for case in range(len(STRENGTH_LIST)):
    plt.plot(taus[:, case])
    legend_list.append('Strength: {:.2f}, Core Size: {:.2f}'.format(STRENGTH_LIST[case], SIGMA_LIST[case]))
plt.legend(legend_list)
plt.title('Evolution of Vortex Strength')
plt.show()

plt.figure()
for case in range(len(STRENGTH_LIST)):
    plt.plot(sigs[:, case])
plt.legend(legend_list)
plt.title('Evolution of Vortex Core Size')
plt.show()




