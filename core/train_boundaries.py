import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from core.datasets import VortexBoundariesDataset
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import *
import os
from core.velocity_derivs import *
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[100, 100], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/'
                                                    'data/p10_b_sb_dataset_100x100_4000',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--num_time_steps', type=int, default=1, help='train the network on loss for more than 1 time step')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=1e-2, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='weight for l2 regularization')
parser.add_argument('--ex', type=str, default='p10_b_sb_T1_exp_red_BC_weight_1.0_depth_5_100_batch_16_lr_1e-2_l2_1e-5_r100_4000_2', help='name of the experiment')
parser.add_argument('--load_weights_ex', type=str, default='p10_b_sb_T1_exp_red_weight_1.0_depth_5_100_batch_16_lr_1e-2_l2_1e-5_r100_4000_2', help='name of the experiment')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--loss_scaling', type=float, default=1.0, help='scaling of loss for training to predict to more than one time stepo')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--kernel', type=str, default='ExpGaussianRed', help='kernel representing vorticity strength filed. options:'
                                                                   ' "guassian" or "offset-gaussian" ')

# MEAN = [64.0, 0.0, 27.5]
# STDDEV = [23.094, 1.52752, 12.9903]

MEAN = [0.0, 0.0, 0.0]
STDDEV = [1.0, 1.0, 1.0]

mean_tensor = torch.tensor(MEAN, dtype=torch.float32, device='cuda:0')
stddev_tensor = torch.tensor(MEAN, dtype=torch.float32, device='cuda:0')

opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride
RESOLUTION = opt.domain
BATCH_SIZE = opt.batch_size
weights = [0.0] + [opt.loss_scaling**i for i in range(NUM_TIME_STEPS)]

delta_t = torch.tensor(opt.stride, dtype=torch.float32, device='cuda:0')

loss_weights = torch.tensor(weights, dtype=torch.float32, device=('cuda:0'))
print(loss_weights)
data_dir = opt.data_dir

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

logs_dir = os.path.join('../logs', opt.ex)
ckpt_save_dir = os.path.join(logs_dir, 'ckpt')
train_summaries_dir = os.path.join(logs_dir, 'train_summary')
val_summaries_dir = os.path.join(logs_dir, 'val_summary')

for dir in [ckpt_save_dir, train_summaries_dir, val_summaries_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

with open(os.path.join(logs_dir, 'train_config'), 'w') as configfile:
    json.dump(vars(opt), configfile, indent=2)

train_dataset = VortexBoundariesDataset(train_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)
val_dataset = VortexBoundariesDataset(val_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                   pin_memory=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=4)

steps_per_epoch = int(len(train_dataset) / BATCH_SIZE)
val_steps_per_epoch = int(len(val_dataset) / BATCH_SIZE)

train_dataiter = iter(train_dataloader)#
batch_data_dict = next(train_dataiter)

location = batch_data_dict['location'].to('cuda:0')
strength = batch_data_dict['strength'].to('cuda:0')
sigma = batch_data_dict['sigma'].to('cuda:0')
velocities = [batch_data_dict['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]

nparticles = location.shape[1]

y, x = torch.unbind(location, dim=-1)

tau, sig = strength.view(opt.batch_size, -1), sigma.view(opt.batch_size, -1)

c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001

inp_vector = torch.stack([y, x, tau, sig, c, d], dim=-1)

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW_REF = Fluid(domain=domain)
points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')


start_epoch = 0

val_best = 10000000.0

@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', a=0.1)
        torch.nn.init.zeros_(m.bias.data)

VortexNet = MultiStepMultiVortexNetworkBC(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                          kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV, order=opt.order,
                                          num_steps=opt.num_time_steps, distinct_nets=opt.distinct_nets)
BCNet = BoundaryConditionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True, order=opt.order)
# VortexNet = InteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
#                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV)
#
#
if opt.load_weights_ex is not None:
    init_weights_log_dir = os.path.join('../logs', opt.load_weights_ex)
    init_weights_ckpt_dir = os.path.join(init_weights_log_dir, 'ckpt')

    checkpoints_files = os.listdir(os.path.join(init_weights_ckpt_dir))
    epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
    init_weights_ckpt_file = os.path.join(init_weights_ckpt_dir, checkpoints_files[epoch_id])
    params = torch.load(init_weights_ckpt_file)['model_state_dict']
    VortexNet.single_step_net.load_state_dict(params)
    if opt.num_time_steps > 1 and opt.distinct_nets:
        params2 = torch.load(init_weights_ckpt_file)['model_state_dict2']
        VortexNet.single_step_net2.load_state_dict(params2)
else:
    VortexNet.apply(init_weights)

VortexNet.to('cuda:0')
VortexNet.requires_grad_(requires_grad=False).eval()

BCNet.to('cuda:0')
BCNet.requires_grad_(requires_grad=True).train()

optimizer = Adam(params=BCNet.parameters(), lr=opt.lr, weight_decay=opt.l2)
# optimizer = SGD(params=VortexNet.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.l2)
#
if opt.ex == opt.load_weights_ex:
    optimizer.load_state_dict(torch.load(init_weights_ckpt_file)['optimizer_state_dict'])
    start_epoch = torch.load(init_weights_ckpt_file)['epoch']

lambda1 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lambda1)
# scheduler = StepLR(optimizer, 5, gamma=0.95)
# scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=25)

if opt.kernel == 'ExpGaussianRed':
    falloff_kernel = GaussExpFalloffKernelReduced()
elif opt.kernel == 'ExpGaussian':
    falloff_kernel =  GaussExpFalloffKernel()
elif opt.kernel == 'gaussian':
    falloff_kernel =  GaussianFalloffKernel()

train_summary = SummaryWriter(log_dir=train_summaries_dir)
val_summary = SummaryWriter(log_dir=val_summaries_dir)
#
train_loss_module = MultiStepLoss(kernel=opt.kernel, resolution=opt.domain,
                                  num_steps=opt.num_time_steps, batch_size=opt.batch_size, dt=delta_t)
val_loss_module = MultiStepLoss(kernel=opt.kernel, resolution=opt.domain,
                                num_steps=opt.num_time_steps, batch_size=opt.batch_size, dt=delta_t)

# epoch = 0
# train_dataiter = iter(train_dataloader)
# VortexNet.eval()
#
# print('============================== Starting Epoch; {}/{} ========================================='.format(epoch + 1,
#                                                                                                               opt.epochs))
# print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))
#
# batch_data_dict = next(train_dataiter)
#
# location = batch_data_dict['location'].to('cuda:0')
# strength = batch_data_dict['strength'].to('cuda:0')
# sigma = batch_data_dict['sigma'].to('cuda:0')
# velocities = [batch_data_dict['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]
#
# nparticles = location.shape[1]

# optimizer.zero_grad()

# y, x = torch.unbind(location, dim=-1)
# tau, sig = strength.view(opt.batch_size, -1), sigma.view(opt.batch_size, -1)
#
# c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
# d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
#
# if opt.kernel == 'gaussian':
#     inp_vector = torch.stack([y, x, tau, sig, ], dim=-1)
# elif opt.kernel == 'offset-gaussian':
#     off = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
#     sig_l = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
#     inp_vector = torch.stack([y, x, tau, sig, off, sig_l], dim=-1)
# elif opt.kernel == 'ExpGaussian':
#     c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
#     d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
#     inp_vector = torch.stack([y, x, tau, sig, c, d], dim=-1)
# elif opt.kernel == 'ExpGaussianRed':
#     d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
#     inp_vector = torch.stack([y, x, tau, sig, d], dim=-1)
#
# vortex_features_out = VortexNet(inp_vector)
# vortex_features = [vortex_features_out[i].clone().detach() for i in range(len(vortex_features_out))]
# mse_loss_list, max_loss_list = train_loss_module(vortex_features, velocities)
# mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
# loss = torch.sum(mse_loss_tensor)
# loss.backward()
# optimizer.step()

# deriv_module = VelocityDerivatives(order=opt.order, kernel=opt.kernel)
# y_ph = torch.rand(BATCH_SIZE * 1000, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE , -1) * RESOLUTION[0]
# x_ph = torch.rand(BATCH_SIZE * 1000, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE , -1) * RESOLUTION[1]
#
# y_b1 = torch.rand(BATCH_SIZE * 20, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE , -1) * RESOLUTION[0]
# x_b1 = torch.tensor(np.concatenate([np.array([0] * 10), np.array([RESOLUTION[1]] * 10)]),
#                     dtype=torch.float32, device='cuda:0').view(1 , -1).repeat(BATCH_SIZE, 1)
#
# x_b2 = torch.rand(BATCH_SIZE * 20, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE , -1) * RESOLUTION[1]
# y_b2 = torch.tensor(np.concatenate([np.array([0] * 10), np.array([RESOLUTION[0]] * 10)]),
#                     dtype=torch.float32, device='cuda:0').view(1 , -1).repeat(BATCH_SIZE, 1)
#
# points_ph = torch.stack([y_ph, x_ph], dim=-1)
# points_b1 = torch.stack([y_b1, x_b1], dim=-1)
# points_b2 = torch.stack([y_b2, x_b2], dim=-1)
#

# derivs_list = []
# derivs_list_b1 = []
# derivs_list_b2 = []
# velocity_field_by_vortex = []
# velocity_field_by_vortex_b1 = []
# velocity_field_by_vortex_b2 = []
#
# for i in range(NUM_TIME_STEPS + 1):
#     derivs_list.append(deriv_module(vortex_features[i], points_ph))
#     derivs_list_b1.append(deriv_module(vortex_features[i], points_b1))
#     derivs_list_b2.append(deriv_module(vortex_features[i], points_b2))
#     velocity_field_by_vortex.append(falloff_kernel(vortex_features[i], points_ph.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
#     velocity_field_by_vortex_b1.append(falloff_kernel(vortex_features[i], points_b1.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
#     velocity_field_by_vortex_b2.append(falloff_kernel(vortex_features[i], points_b2.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
#
#
# Compute divergence loss
# div_dv_dy_list = []
# div_du_dx_list = []
#
# for b in range(BATCH_SIZE):
#     div_b_dv_dy_list = []
#     div_b_du_dx_list = []
#
#     for t in range(NUM_TIME_STEPS + 1):
#         paxes_tensor = torch.tensor([b], device='cuda:0')
#         y_b_ph = torch.index_select(y_ph, dim=0, index=paxes_tensor).view(-1)
#         x_b_ph = torch.index_select(x_ph, dim=0, index=paxes_tensor).view(-1)
#         y_b_ph.requires_grad_(True)
#         x_b_ph.requires_grad_(True)
#         loc_b_ph = torch.stack([y_b_ph, x_b_ph], dim=-1)
#         deriv_b_ph = torch.index_select(derivs_list[t], dim=0, index=paxes_tensor)
#         inp_b_ph = torch.cat([loc_b_ph.view(1, -1, 2), deriv_b_ph], dim=-1)
#         out_b_ph = BCNet(inp_b_ph)
#         corr_vel_y_bh, corr_vel_x_bh = torch.unbind(out_b_ph.view(-1, 2), dim=-1)
#         div_dv_dy = torch.autograd.grad(torch.unbind(corr_vel_y_bh, dim=-1), y_b_ph, retain_graph=True, allow_unused=True)[0]
#         div_du_dx = torch.autograd.grad(torch.unbind(corr_vel_x_bh, dim=-1), x_b_ph, allow_unused=True)[0]
#         div_b_dv_dy_list.append(div_dv_dy)
#         div_b_du_dx_list.append(div_du_dx)
#     div_dv_dy_list.append(torch.stack(div_b_dv_dy_list, dim=-1))
#     div_du_dx_list.append(torch.stack(div_b_du_dx_list, dim=-1))
#
# div_dv_dy_all = torch.stack(div_dv_dy_list, dim=0)
# div_du_dx_all = torch.stack(div_du_dx_list, dim=0)
#
# div_ph_loss = torch.sum((div_du_dx_all + div_dv_dy_all)**2) / (BATCH_SIZE * div_du_dx_all.shape[-1])
#
# Compute boundary condition loss
# out_b1_list = []
# out_b2_list = []

# total_field_b1_list = []
# total_field_b2_list = []
#
# for t in range(NUM_TIME_STEPS + 1):
#     inp_b1 = torch.cat([points_b1, derivs_list_b1[t]], dim=-1)
#     inp_b2 = torch.cat([points_b2, derivs_list_b2[i]], dim=-1)
#     out_b1 = BCNet(inp_b1)
#     out_b2 = BCNet(inp_b2)
#     out_b1_list.append(out_b1)
#     out_b2_list.append(out_b2)
#     total_field_b1_list.append(velocity_field_by_vortex_b1[t] + out_b1_list[t])
#     total_field_b2_list.append(velocity_field_by_vortex_b2[t] + out_b2_list[t])
#
# total_vel_b1 = torch.stack(total_field_b1_list, dim=-1)
# total_vel_b2 = torch.stack(total_field_b2_list, dim=-1)
#
# _, total_vel_b1_x = torch.unbind(total_vel_b1, dim=-2)
# total_vel_b2_y, _ = torch.unbind(total_vel_b2, dim=-2)
#
# bc_loss = torch.sum((total_vel_b1_x ** 2 + total_vel_b2_y ** 2)) / (BATCH_SIZE * total_vel_b1_x.shape[-1])
#
# total_bc_net_loss = bc_loss + div_ph_loss
#
#







# correction_fields = BCNet(derivs_list)
#
# total_field = []
#
# for i in range(NUM_TIME_STEPS + 1):
#     total_field.append(velocity_field_by_vortex[i] + correction_fields[i])


# print('Epoch: {}, Step: {}/{}, loss: {:.4f}, Max_loss: {:.4f}'.
#       format(epoch, step, steps_per_epoch, loss.item(), max_loss_list[-1].item()))

deriv_module = VelocityDerivatives(order=opt.order, kernel=opt.kernel).to('cuda:0')

for epoch in range(start_epoch, opt.epochs):

    train_dataiter = iter(train_dataloader)
    VortexNet.eval()
    BCNet.train()

    print('============================== Starting Epoch; {}/{} ========================================='.format(epoch+1, opt.epochs))
    print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))
    for step in range(steps_per_epoch):

        batch_data_dict = next(train_dataiter)

        location = batch_data_dict['location'].to('cuda:0')
        strength = batch_data_dict['strength'].to('cuda:0')
        sigma = batch_data_dict['sigma'].to('cuda:0')
        velocities = [batch_data_dict['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]

        nparticles = location.shape[1]

        # optimizer.zero_grad()

        y, x = torch.unbind(location, dim=-1)
        tau, sig = strength.view(opt.batch_size, -1), sigma.view(opt.batch_size, -1)

        c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
        d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001

        if opt.kernel == 'gaussian':
            inp_vector = torch.stack([y, x, tau, sig,], dim=-1)
        elif opt.kernel == 'offset-gaussian':
            off = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
            sig_l = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
            inp_vector = torch.stack([y, x, tau, sig, off, sig_l], dim=-1)
        elif opt.kernel == 'ExpGaussian':
            c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
            d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
            inp_vector = torch.stack([y, x, tau, sig, c, d], dim=-1)
        elif opt.kernel == 'ExpGaussianRed':
            d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
            inp_vector = torch.stack([y, x, tau, sig, d], dim=-1)

        vortex_features_out = VortexNet(inp_vector)
        vortex_features = [vortex_features_out[i].clone().detach() for i in range(len(vortex_features_out))]

        y_ph = torch.rand(BATCH_SIZE * 100, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[0]
        x_ph = torch.rand(BATCH_SIZE * 100, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[1]

        y_b1 = torch.rand(BATCH_SIZE * 10, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[0]
        x_b1 = torch.tensor(np.concatenate([np.array([0] * 10), np.array([RESOLUTION[1]] * 10)]),
                            dtype=torch.float32, device='cuda:0').view(1, -1).repeat(BATCH_SIZE, 1)

        x_b2 = torch.rand(BATCH_SIZE * 10, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[1]
        y_b2 = torch.tensor(np.concatenate([np.array([0] * 10), np.array([RESOLUTION[0]] * 10)]),
                            dtype=torch.float32, device='cuda:0').view(1, -1).repeat(BATCH_SIZE, 1)

        points_ph = torch.stack([y_ph, x_ph], dim=-1)
        points_b1 = torch.stack([y_b1, x_b1], dim=-1)
        points_b2 = torch.stack([y_b2, x_b2], dim=-1)

        derivs_list = []
        derivs_list_b1 = []
        derivs_list_b2 = []
        velocity_field_by_vortex = []
        velocity_field_by_vortex_b1 = []
        velocity_field_by_vortex_b2 = []

        for i in range(NUM_TIME_STEPS + 1):
            derivs_list.append(deriv_module(vortex_features[i], points_ph))
            derivs_list_b1.append(deriv_module(vortex_features[i], points_b1))
            derivs_list_b2.append(deriv_module(vortex_features[i], points_b2))
            velocity_field_by_vortex.append(
                falloff_kernel(vortex_features[i], points_ph.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
            velocity_field_by_vortex_b1.append(
                falloff_kernel(vortex_features[i], points_b1.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
            velocity_field_by_vortex_b2.append(
                falloff_kernel(vortex_features[i], points_b2.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))

        optimizer.zero_grad()

        # Compute divergence loss
        div_dv_dy_list = []
        div_du_dx_list = []

        for b in range(BATCH_SIZE):
            div_b_dv_dy_list = []
            div_b_du_dx_list = []

            for t in range(NUM_TIME_STEPS + 1):
                paxes_tensor = torch.tensor([b], device='cuda:0')
                y_b_ph = torch.index_select(y_ph, dim=0, index=paxes_tensor).view(-1)
                x_b_ph = torch.index_select(x_ph, dim=0, index=paxes_tensor).view(-1)
                y_b_ph.requires_grad_(True)
                x_b_ph.requires_grad_(True)
                loc_b_ph = torch.stack([y_b_ph, x_b_ph], dim=-1)
                deriv_b_ph = torch.index_select(derivs_list[t], dim=0, index=paxes_tensor)
                inp_b_ph = torch.cat([loc_b_ph.view(1, -1, 2), deriv_b_ph], dim=-1)
                out_b_ph = BCNet(inp_b_ph)
                corr_vel_y_bh, corr_vel_x_bh = torch.unbind(out_b_ph.view(-1, 2), dim=-1)
                div_dv_dy = \
                torch.autograd.grad(torch.unbind(corr_vel_y_bh, dim=-1), y_b_ph, retain_graph=True, allow_unused=True)[
                    0]
                div_du_dx = torch.autograd.grad(torch.unbind(corr_vel_x_bh, dim=-1), x_b_ph, allow_unused=True)[0]
                div_b_dv_dy_list.append(div_dv_dy)
                div_b_du_dx_list.append(div_du_dx)
            div_dv_dy_list.append(torch.stack(div_b_dv_dy_list, dim=-1))
            div_du_dx_list.append(torch.stack(div_b_du_dx_list, dim=-1))

        div_dv_dy_all = torch.stack(div_dv_dy_list, dim=0)
        div_du_dx_all = torch.stack(div_du_dx_list, dim=0)

        div_ph_loss = torch.sum((div_du_dx_all + div_dv_dy_all) ** 2) / (BATCH_SIZE * div_du_dx_all.shape[-1])

        # Compute boundary condition loss
        out_b1_list = []
        out_b2_list = []

        total_field_b1_list = []
        total_field_b2_list = []

        for t in range(NUM_TIME_STEPS + 1):
            inp_b1 = torch.cat([points_b1, derivs_list_b1[t]], dim=-1)
            inp_b2 = torch.cat([points_b2, derivs_list_b2[i]], dim=-1)
            out_b1 = BCNet(inp_b1)
            out_b2 = BCNet(inp_b2)
            out_b1_list.append(out_b1)
            out_b2_list.append(out_b2)
            total_field_b1_list.append(velocity_field_by_vortex_b1[t] + out_b1_list[t])
            total_field_b2_list.append(velocity_field_by_vortex_b2[t] + out_b2_list[t])

        total_vel_b1 = torch.stack(total_field_b1_list, dim=-1)
        total_vel_b2 = torch.stack(total_field_b2_list, dim=-1)

        _, total_vel_b1_x = torch.unbind(total_vel_b1, dim=-2)
        total_vel_b2_y, _ = torch.unbind(total_vel_b2, dim=-2)

        bc_loss = torch.sum((total_vel_b1_x ** 2 + total_vel_b2_y ** 2)) / (BATCH_SIZE * total_vel_b1_x.shape[-1])

        total_bc_net_loss = bc_loss + div_ph_loss

        # mse_loss_list, max_loss_list = train_loss_module(vortex_features, velocities)
        # mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
        # loss = torch.sum(mse_loss_tensor)
        total_bc_net_loss.backward()
        optimizer.step()

        print('Epoch: {}, Step: {}/{}, loss: {:.4f}'.
              format(epoch, step, steps_per_epoch, total_bc_net_loss.item()))

    BCNet.eval()
    #
    # with torch.no_grad():

    val_dataiter = iter(val_dataloader)

    val_loss = 0.0

    for val_step in range(val_steps_per_epoch):

        val_batch = next(val_dataiter)

        location = val_batch['location'].to('cuda:0')
        strength = val_batch['strength'].to('cuda:0')
        sigma = val_batch['sigma'].to('cuda:0')
        velocities = [val_batch['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]

        v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
        u = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')

        nparticles = location.shape[1]

        y, x = torch.unbind(location, dim=-1)
        tau, sig = strength.view(opt.batch_size, -1), sigma.view(opt.batch_size, -1)

        c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
        d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001

        if opt.kernel == 'gaussian':
            inp_vector = torch.stack([y, x, tau, sig, v, u], dim=-1)
        elif opt.kernel == 'offset-gaussian':
            off = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
            sig_l = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
            inp_vector = torch.stack([y, x, tau, sig, v, u, off, sig_l], dim=-1)
        elif opt.kernel == 'ExpGaussian':
            c = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0')
            d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
            inp_vector = torch.stack([y, x, tau, sig, c, d], dim=-1)
        elif opt.kernel == 'ExpGaussianRed':
            d = torch.zeros((BATCH_SIZE, nparticles), dtype=torch.float32, device='cuda:0') + 0.001
            inp_vector = torch.stack([y, x, tau, sig, d], dim=-1)

        vortex_features_out = VortexNet(inp_vector)

        vortex_features = [vortex_features_out[i].clone().detach() for i in range(len(vortex_features_out))]

        y_ph = torch.rand(BATCH_SIZE * 100, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[0]
        x_ph = torch.rand(BATCH_SIZE * 100, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[1]

        y_b1 = torch.rand(BATCH_SIZE * 10, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[0]
        x_b1 = torch.tensor(np.concatenate([np.array([0] * 5), np.array([RESOLUTION[1]] * 5)]),
                            dtype=torch.float32, device='cuda:0').view(1, -1).repeat(BATCH_SIZE, 1)

        x_b2 = torch.rand(BATCH_SIZE * 10, dtype=torch.float32, device='cuda:0').view(BATCH_SIZE, -1) * RESOLUTION[1]
        y_b2 = torch.tensor(np.concatenate([np.array([0] * 5), np.array([RESOLUTION[0]] * 5)]),
                            dtype=torch.float32, device='cuda:0').view(1, -1).repeat(BATCH_SIZE, 1)

        points_ph = torch.stack([y_ph, x_ph], dim=-1)
        points_b1 = torch.stack([y_b1, x_b1], dim=-1)
        points_b2 = torch.stack([y_b2, x_b2], dim=-1)

        derivs_list = []
        derivs_list_b1 = []
        derivs_list_b2 = []
        velocity_field_by_vortex = []
        velocity_field_by_vortex_b1 = []
        velocity_field_by_vortex_b2 = []

        for i in range(NUM_TIME_STEPS + 1):
            derivs_list.append(deriv_module(vortex_features[i], points_ph))
            derivs_list_b1.append(deriv_module(vortex_features[i], points_b1))
            derivs_list_b2.append(deriv_module(vortex_features[i], points_b2))
            velocity_field_by_vortex.append(
                falloff_kernel(vortex_features[i], points_ph.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
            velocity_field_by_vortex_b1.append(
                falloff_kernel(vortex_features[i], points_b1.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))
            velocity_field_by_vortex_b2.append(
                falloff_kernel(vortex_features[i], points_b2.view(BATCH_SIZE, -1, 1, 2)).view(BATCH_SIZE, -1, 2))

        # Compute divergence loss
        div_dv_dy_list = []
        div_du_dx_list = []

        for b in range(BATCH_SIZE):
            div_b_dv_dy_list = []
            div_b_du_dx_list = []

            for t in range(NUM_TIME_STEPS + 1):
                paxes_tensor = torch.tensor([b], device='cuda:0')
                y_b_ph = torch.index_select(y_ph, dim=0, index=paxes_tensor).view(-1)
                x_b_ph = torch.index_select(x_ph, dim=0, index=paxes_tensor).view(-1)
                y_b_ph.requires_grad_(True)
                x_b_ph.requires_grad_(True)
                loc_b_ph = torch.stack([y_b_ph, x_b_ph], dim=-1)
                deriv_b_ph = torch.index_select(derivs_list[t], dim=0, index=paxes_tensor)
                inp_b_ph = torch.cat([loc_b_ph.view(1, -1, 2), deriv_b_ph], dim=-1)
                out_b_ph = BCNet(inp_b_ph)
                corr_vel_y_bh, corr_vel_x_bh = torch.unbind(out_b_ph.view(-1, 2), dim=-1)
                div_dv_dy = \
                torch.autograd.grad(torch.unbind(corr_vel_y_bh, dim=-1), y_b_ph, retain_graph=True, allow_unused=True)[
                    0]
                div_du_dx = torch.autograd.grad(torch.unbind(corr_vel_x_bh, dim=-1), x_b_ph, allow_unused=True)[0]
                div_b_dv_dy_list.append(div_dv_dy)
                div_b_du_dx_list.append(div_du_dx)
            div_dv_dy_list.append(torch.stack(div_b_dv_dy_list, dim=-1))
            div_du_dx_list.append(torch.stack(div_b_du_dx_list, dim=-1))

        div_dv_dy_all = torch.stack(div_dv_dy_list, dim=0)
        div_du_dx_all = torch.stack(div_du_dx_list, dim=0)

        div_ph_loss = torch.sum((div_du_dx_all + div_dv_dy_all) ** 2) / (BATCH_SIZE * div_du_dx_all.shape[-1])

        # Compute boundary condition loss
        out_b1_list = []
        out_b2_list = []

        total_field_b1_list = []
        total_field_b2_list = []

        for t in range(NUM_TIME_STEPS + 1):
            inp_b1 = torch.cat([points_b1, derivs_list_b1[t]], dim=-1)
            inp_b2 = torch.cat([points_b2, derivs_list_b2[i]], dim=-1)
            out_b1 = BCNet(inp_b1)
            out_b2 = BCNet(inp_b2)
            out_b1_list.append(out_b1)
            out_b2_list.append(out_b2)
            total_field_b1_list.append(velocity_field_by_vortex_b1[t] + out_b1_list[t])
            total_field_b2_list.append(velocity_field_by_vortex_b2[t] + out_b2_list[t])

        total_vel_b1 = torch.stack(total_field_b1_list, dim=-1)
        total_vel_b2 = torch.stack(total_field_b2_list, dim=-1)

        _, total_vel_b1_x = torch.unbind(total_vel_b1, dim=-2)
        total_vel_b2_y, _ = torch.unbind(total_vel_b2, dim=-2)

        bc_loss = torch.sum((total_vel_b1_x ** 2 + total_vel_b2_y ** 2)) / (BATCH_SIZE * total_vel_b1_x.shape[-1])

        total_bc_net_loss = bc_loss + div_ph_loss
        val_loss = val_loss + total_bc_net_loss
        # with torch.no_grad():
        #     mse_loss_list, max_loss_list = val_loss_module(vortex_features, velocities)
        #     mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
        #     val_loss = val_loss + torch.sum(mse_loss_tensor * loss_weights)
    #
    val_loss = val_loss / val_steps_per_epoch

    print('Epoch; {}, val_loss: {:.4f}'.format(epoch, val_loss.item()))
    #
    val_summary.add_scalar('val_l2_loss', val_loss.item(), (epoch) * steps_per_epoch)

    # if val_loss.item() < val_best:
    #     if opt.num_time_steps > 1 and opt.distinct_nets:
    #         save_state_dict = {
    #             'model_state_dict': VortexNet.single_step_net.state_dict(),
    #             'model_state_dict2': VortexNet.single_step_net2.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'val_loss': val_loss,
    #         }
    #     else:
    #         save_state_dict = {
    #             'model_state_dict': VortexNet.single_step_net.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'val_loss': val_loss,
    #         }

    if val_loss.item() < val_best:
        if opt.num_time_steps > 1 and opt.distinct_nets:
            save_state_dict = {
                'model_state_dict': BCNet.state_dict(),
                'model_state_dict2': BCNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }
        else:
            save_state_dict = {
                'model_state_dict': BCNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }


        ckpt_filename = 'ckpt_{:02d}_val_loss_{:.4f}.pytorch'.format(epoch, val_loss.item())
        ckpt_path = os.path.join(ckpt_save_dir, ckpt_filename)
        torch.save(save_state_dict, ckpt_path)

        val_best = val_loss.item()

    # scheduler.step(metrics=val_loss, epoch=epoch)
    scheduler.step(epoch=epoch)