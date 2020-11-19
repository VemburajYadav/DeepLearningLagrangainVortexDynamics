import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from core.datasets import MultiVortexDataset
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import *
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/'
                                                    'data/p2_r_dataset_256x256_32000',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--num_time_steps', type=int, default=2, help='train the network on loss for more than 1 time step')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='weight for l2 regularization')
parser.add_argument('--ex', type=str, default='p2_r_T2_exp_weight_1.0_depth_2_100_batch_64_c3d_lr_1e-3_l2_1e-5_r256_32000_1', help='name of the experiment')
parser.add_argument('--load_weights_ex', type=str, default=None, help='name of the experiment')
parser.add_argument('--depth', type=int, default=1, help='number of hidden layers')
parser.add_argument('--order', type=int, default=1, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--loss_scaling', type=float, default=1.0, help='scaling of loss for training to predict to more than one time stepo')
parser.add_argument('--distinct_nets', type=bool, default=False, help='True for two networks for multi step training and False for single network')
parser.add_argument('--kernel', type=str, default='ExpGaussian', help='kernel representing vorticity strength filed. options:'
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

train_dataset = MultiVortexDataset(train_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)
val_dataset = MultiVortexDataset(val_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)

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

VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV, order=opt.order,
                                        num_steps=opt.num_time_steps, distinct_nets=opt.distinct_nets)
# VortexNet = InteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, batch_norm=True,
#                                        kernel=opt.kernel, norm_mean=MEAN, norm_stddev=STDDEV)


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
#
VortexNet.to('cuda:0')
VortexNet.requires_grad_(requires_grad=True).train()

optimizer = Adam(params=VortexNet.parameters(), lr=opt.lr, weight_decay=opt.l2)
# optimizer = SGD(params=VortexNet.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.l2)
#
if opt.ex == opt.load_weights_ex:
    optimizer.load_state_dict(torch.load(init_weights_ckpt_file)['optimizer_state_dict'])
    start_epoch = torch.load(init_weights_ckpt_file)['epoch']

lambda1 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lambda1)
# scheduler = StepLR(optimizer, 100, gamma=0.1)
# scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
#
train_summary = SummaryWriter(log_dir=train_summaries_dir)
val_summary = SummaryWriter(log_dir=val_summaries_dir)

train_loss_module = MultiStepLoss(kernel=opt.kernel, resolution=opt.domain,
                                  num_steps=opt.num_time_steps, batch_size=opt.batch_size, dt=delta_t)
val_loss_module = MultiStepLoss(kernel=opt.kernel, resolution=opt.domain,
                                num_steps=opt.num_time_steps, batch_size=opt.batch_size, dt=delta_t)

for epoch in range(start_epoch, opt.epochs):

    train_dataiter = iter(train_dataloader)
    VortexNet.train()

    print('============================== Starting Epoch; {}/{} ========================================='.format(epoch+1, opt.epochs))
    print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))
    for step in range(steps_per_epoch):

        batch_data_dict = next(train_dataiter)

        location = batch_data_dict['location'].to('cuda:0')
        strength = batch_data_dict['strength'].to('cuda:0')
        sigma = batch_data_dict['sigma'].to('cuda:0')
        velocities = [batch_data_dict['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]

        nparticles = location.shape[1]

        optimizer.zero_grad()

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

        vortex_features = VortexNet(inp_vector)
        mse_loss_list, max_loss_list = train_loss_module(vortex_features, velocities)
        mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
        loss = torch.sum(mse_loss_tensor)
        loss.backward()
        optimizer.step()

        print('Epoch: {}, Step: {}/{}, loss: {:.4f}, Max_loss: {:.4f}'.
              format(epoch, step, steps_per_epoch, loss.item(), max_loss_list[-1].item()))

    VortexNet.eval()

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

        vortex_features = VortexNet(inp_vector)

        with torch.no_grad():
            mse_loss_list, max_loss_list = val_loss_module(vortex_features, velocities)
            mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
            val_loss = val_loss + torch.sum(mse_loss_tensor * loss_weights)

    val_loss = val_loss / val_steps_per_epoch

    print('Epoch; {}, val_loss: {:.4f}'.format(epoch, val_loss.item()))

    val_summary.add_scalar('val_l2_loss', val_loss.item(), (epoch) * steps_per_epoch)

    if val_loss.item() < val_best:
        if opt.num_time_steps > 1 and opt.distinct_nets:
            save_state_dict = {
                'model_state_dict': VortexNet.single_step_net.state_dict(),
                'model_state_dict2': VortexNet.single_step_net2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }
        else:
            save_state_dict = {
                'model_state_dict': VortexNet.single_step_net.state_dict(),
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