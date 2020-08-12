import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from core.custom_functions import particle_vorticity_to_velocity
from torch.utils.tensorboard import SummaryWriter
from core.datasets import SingleVortexDataset
import argparse
import matplotlib.pyplot as plt
from phi.flow import Domain
from core.networks import SimpleNN
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_1/eval',
                    help='path to the directory with data to make predictions')
parser.add_argument('--ex', type=str, default='train_demo_mod', help='name of the experiment')
parser.add_argument('--depth', type=int, default=3, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=1024, help='number of neurons in hidden layers')

opt = parser.parse_args()

RESOLUTION = opt.domain
domain = Domain(RESOLUTION)
sample_points = domain.center_points()

data_dir = opt.data_dir

logs_dir = os.path.join('../logs', opt.ex)
ckpt_dir = os.path.join(logs_dir, 'ckpt')

# checkpoints_files = os.listdir(os.path.join(ckpt_dir))
# epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
# ckpt_file = os.path.join(ckpt_dir, checkpoints_files[epoch_id])
#
dataset = SingleVortexDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True, shuffle=True, pin_memory=True)

model_ = SimpleNN(depth=opt.depth, hidden_units=opt.hidden_units, in_features=4, out_features=4)
model_.eval()

# params = torch.load(ckpt_file)['model_state_dict']

# model_.load_state_dict(params)
model_.to('cuda:0')
model_.eval()

dataiter = iter(dataloader)

with torch.no_grad():

    data_dict = next(dataiter)

    loc = data_dict['location']
    tau = data_dict['strength']
    sig = data_dict['sigma']

    pvel0 = torch.tensor([0.0, 0.0], dtype=torch.float32).view(1, -1).repeat(1, 1)

    print(tau.shape, sig.shape)
    inp = torch.cat([torch.squeeze(tau, dim=1),
                     torch.squeeze(sig, dim=1), pvel0], dim=-1)

    inp_gpu = inp.to('cuda:0')
    loc0_gpu = torch.squeeze(loc, dim=1).to('cuda:0')

    out_gpu = model_(inp_gpu)

    tau, sig, pu0, pv0 = torch.unbind(inp_gpu, dim=-1)
    dy, dx, dtau, dsig = torch.unbind(out_gpu, dim=-1)

    dy = dy * 0.1
    dx = dx * 0.1
    dsig = F.softplus(dsig)

    new_pos = torch.unsqueeze(torch.stack([dy, dx], dim=-1) + loc0_gpu, dim=1)
    new_tau = torch.unsqueeze(tau + dtau, dim=-1)
    new_sig = torch.unsqueeze(torch.unsqueeze(sig + dsig, dim=-1), dim=-1)

    points_gpu = torch.tensor(sample_points, dtype=torch.float32, device='cuda:0')
    pred_vel1 = particle_vorticity_to_velocity(new_pos, new_tau, new_sig, points_gpu)
    vel1 = data_dict['velocity1']

    vel0 = particle_vorticity_to_velocity(torch.unsqueeze(loc0_gpu, dim=1),
                                          torch.unsqueeze(tau, dim=-1),
                                          torch.unsqueeze(torch.unsqueeze(sig, dim=-1), dim=-1), points_gpu)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(data_dict['velocity1'].cpu().numpy()[0, :, :, 0], cmap='RdYlBu')
plt.subplot(1,2,2)
plt.imshow(vel0.cpu().numpy()[0, :, :, 0], cmap='RdYlBu')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(data_dict['velocity1'].cpu().numpy()[0, :, :, 1], cmap='RdYlBu')
plt.subplot(1,2,2)
plt.imshow(vel0.cpu().numpy()[0, :, :, 1], cmap='RdYlBu')
plt.show()
