import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from core.datasets import MultiVortexDataset
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import *
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/'
                                                    'data/p2_r_dataset_256x256_8000',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--num_time_steps', type=int, default=2, help='train the network on loss for more than 1 time step')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--batch_size', type=int, default=3, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=1e-1, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-4, help='weight for l2 regularization')
parser.add_argument('--ex', type=str, default='T3_exp_weight_1.0_depth_2_200_batch_16_c2d_lr_1e-1_l2_1e-4_r256', help='name of the experiment')
parser.add_argument('--load_weights_ex', type=str, default=None, help='name of the experiment')
parser.add_argument('--depth', type=int, default=2, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=200, help='number of neurons in hidden layers')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--kernel', type=str, default='ExpGaussian', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# MEAN = [64.0, 0.0, 27.5]
# STDDEV = [23.094, 1.52752, 12.9903]

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

mean_tensor = torch.tensor(MEAN, dtype=torch.float32)
stddev_tensor = torch.tensor(MEAN, dtype=torch.float32)

opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride
RESOLUTION = opt.domain
BATCH_SIZE = opt.batch_size
weights = [0.0] + [1.0**i for i in range(NUM_TIME_STEPS)]

delta_t = torch.tensor(opt.stride, dtype=torch.float32)

loss_weights = torch.tensor(weights, dtype=torch.float32)
print(loss_weights)
data_dir = opt.data_dir

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

logs_dir = os.path.join('../logs', opt.ex)
ckpt_save_dir = os.path.join(logs_dir, 'ckpt')
train_summaries_dir = os.path.join(logs_dir, 'train_summary')
val_summaries_dir = os.path.join(logs_dir, 'val_summary')


train_dataset = MultiVortexDataset(train_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)
# val_dataset = MultiVortexDataset(val_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=1)
# val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=1)

steps_per_epoch = int(len(train_dataset) / BATCH_SIZE)
# val_steps_per_epoch = int(len(val_dataset) / BATCH_SIZE)

train_dataiter = iter(train_dataloader)#
batch_data_dict = next(train_dataiter)

location = batch_data_dict['location']
strength = batch_data_dict['strength']
sigma = batch_data_dict['sigma']
velocities = [batch_data_dict['velocities'][i] for i in range(NUM_TIME_STEPS + 1)]

nparticles = location.shape[1]

y, x = torch.unbind(location, dim=-1)

tau, sig = strength.view(opt.batch_size, -1), sigma.view(opt.batch_size, -1)

c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32)
d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32) + 0.001

inp_vector = torch.stack([y, x, tau, sig, c, d], dim=-1)

kernel = GaussExpFalloffKernel()

pfeatures = torch.unbind(inp_vector, dim=-2)

paxes = np.arange(nparticles)

grads_list = []
vel_list = []

for b in range(BATCH_SIZE):
    batch_axes_tensor = torch.tensor([b])
    batch_features = torch.index_select(inp_vector, dim=0, index=batch_axes_tensor).view(nparticles, -1)
    batch_p_locs = torch.index_select(location, dim=0, index=batch_axes_tensor).view(nparticles, -1)

    grads_p_list = []
    vel_p_list = []

    for i in range(nparticles):
        paxes_tensor = torch.tensor([i])
        p_loc = torch.index_select(batch_p_locs, dim=0, index=paxes_tensor).view(-1)
        py, px = torch.unbind(p_loc, dim=-1)
        py.requires_grad_(True)
        px.requires_grad_(True)
        p_loc_inp = torch.stack([py, px]).view(1, 1, 1, 2)
        other_p_axes = np.delete(paxes, i)
        other_paxes_tensor = torch.tensor(other_p_axes)
        other_p_features = torch.index_select(batch_features, dim=0, index=other_paxes_tensor).view(1, nparticles-1, -1)
        vel_by_other_ps = kernel(other_p_features, p_loc_inp).view(-1)
        vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)
        dv_dy, dv_dx = torch.autograd.grad(vel_y, [py, px], retain_graph=True)
        du_dy, du_dx = torch.autograd.grad(vel_x, [py, px])
        grads = torch.stack([dv_dy, dv_dx, du_dy, du_dx], dim=-1)
        vels = torch.stack([vel_y, vel_x], dim=-1)
        grads_p_list.append(grads)
        vel_p_list.append(vels)

    grads_p_tensor = torch.stack(grads_p_list, dim=0)
    vel_p_tensor = torch.stack(vel_p_list, dim=0)
    grads_list.append(grads_p_tensor)
    vel_list.append(vel_p_tensor)

grads_tensor = torch.stack(grads_list, dim=0)
vel_tensor = torch.stack(vel_list, dim=0)

grad_list_ = []
hess_list_ = []
vel_list_ = []

for i in range(nparticles):
    paxes_tensor_ = torch.tensor([i])
    p_loc_ = torch.index_select(location, dim=-2, index=paxes_tensor_).view(-1, 2)
    py_, px_ = torch.unbind(p_loc_, dim=-1)
    py_.requires_grad_(True)
    px_.requires_grad_(True)
    p_loc_inp_ = torch.stack([py_, px_], dim=-1).view(-1, 1, 1, 2)
    other_p_axes_ = np.delete(paxes, i)
    other_paxes_tensor_ = torch.tensor(other_p_axes_)
    other_p_features_ = torch.index_select(inp_vector, dim=-2, index=other_paxes_tensor_)
    vel_by_other_ps_ = kernel(other_p_features_, p_loc_inp_).view(-1, 2)
    vel_y_, vel_x_ = torch.unbind(vel_by_other_ps_, dim=-1)
    dv_dy_ = torch.autograd.grad(torch.unbind(vel_y_, dim=-1), py_, create_graph=True, retain_graph=True, allow_unused=True)[0]
    dv_dx_ = torch.autograd.grad(torch.unbind(vel_y_, dim=-1), px_, create_graph=True, retain_graph=True, allow_unused=True)[0]
    du_dy_ = torch.autograd.grad(torch.unbind(vel_x_, dim=-1), py_, create_graph=True, retain_graph=True, allow_unused=True)[0]
    du_dx_ = torch.autograd.grad(torch.unbind(vel_x_, dim=-1), px_, create_graph=True, retain_graph=True, allow_unused=True)[0]

    d2u_dx2_ = torch.autograd.grad(torch.unbind(du_dx_, dim=-1), px_, retain_graph=True, allow_unused=True)[0]
    d2u_dy2_ = torch.autograd.grad(torch.unbind(du_dy_, dim=-1), py_, retain_graph=True, allow_unused=True)[0]
    d2u_dydx_ = torch.autograd.grad(torch.unbind(du_dx_, dim=-1), py_, retain_graph=True, allow_unused=True)[0]
    d2u_dxdy_ = torch.autograd.grad(torch.unbind(du_dy_, dim=-1), px_, retain_graph=True, allow_unused=True)[0]

    d2v_dy2_ = torch.autograd.grad(torch.unbind(dv_dy_, dim=-1), py_, retain_graph=True, allow_unused=True)[0]
    d2v_dx2_ = torch.autograd.grad(torch.unbind(dv_dx_, dim=-1), px_, retain_graph=True, allow_unused=True)[0]
    d2v_dxdy_ = torch.autograd.grad(torch.unbind(dv_dy_, dim=-1), px_, retain_graph=True, allow_unused=True)[0]
    d2v_dydx_ = torch.autograd.grad(torch.unbind(dv_dx_, dim=-1), py_, retain_graph=True, allow_unused=True)[0]

    grads_ = torch.stack([dv_dy_, dv_dx_, du_dy_, du_dx_], dim=-1)
    hess_ = torch.stack([d2v_dy2_.detach().clone(), d2v_dx2_.detach().clone(),
                         d2v_dydx_.detach().clone(), d2v_dxdy_.detach().clone(),
                         d2u_dy2_.detach().clone(), d2u_dx2_.detach().clone(),
                         d2u_dydx_.detach().clone(), d2u_dxdy_.detach().clone()], dim=-1)
    vels_ = torch.stack([vel_y_.detach().clone(), vel_x_.detach().clone()], dim=-1)
    grad_list_.append(grads_)
    hess_list_.append(hess_)
    vel_list_.append(vels_)

grads_tensor_ = torch.stack(grad_list_, dim=1)
hess_tensor_ = torch.stack(hess_list_, dim=1)
vel_tensor_ = torch.stack(vel_list_, dim=1)

p1, p2 = torch.unbind(location, dim=-2)
dist = torch.sum((p1 - p2)**2, dim=-1)**0.5

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(batch_data_dict['velocities'][0].numpy()[0, :, :, 1])
# plt.subplot(1, 2, 2)
# plt.imshow(batch_data_dict['velocities'][1].numpy()[0, :, :, 1])
# plt.show()
#



