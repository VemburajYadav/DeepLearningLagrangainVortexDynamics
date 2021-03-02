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

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/media/vemburaj/9d072277-d226-41f6-a38d-1db833dca2bd/Datasets_LVD/'
                                                    'data/p10_gaussian_dataset_120x120_4000',
                    help='path to the directory of the dataset')
parser.add_argument('--network', type=str, default='Vortex',
                    help='type of neural network: Vortex or Interaction')
parser.add_argument('--num_time_steps', type=int, default=1, help='train the network on loss for more than 1 time step')
parser.add_argument('--stride', type=int, default=5, help='skip intermediate time frames corresponding to stride in the dataset '
                                                          'for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='weight for l2 regularization')
parser.add_argument('--logs_dir', type=str, default='../logs_temp', help='directory to save checkpoints and training summaries')
parser.add_argument('--ex', type=str, default='T1_p10_gauss_weight_1.0_depth_5_100_batch_32_lr_1e-3_l2_1e-5_r120_4000_2', help='name of the experiment')
parser.add_argument('--load_weights_ex', type=str, default=None, help='name of the experiment')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--loss_scaling', type=float, default=1.0, help='scaling of loss for training to predict to more than one time stepo')



# Parse Input arguments
opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride
RESOLUTION = opt.domain
BATCH_SIZE = opt.batch_size
NETWORK = opt.network
data_dir = opt.data_dir


# weights for loss for training with more than 1 time step
weights = [0.0] + [opt.loss_scaling**i for i in range(NUM_TIME_STEPS)]
loss_weights = torch.tensor(weights, dtype=torch.float32, device=('cuda:0'))


# get the directories with the data
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')


# directories to save training summaries
logs_dir = os.path.join(opt.logs_dir, opt.ex)
ckpt_save_dir = os.path.join(logs_dir, 'ckpt')
train_summaries_dir = os.path.join(logs_dir, 'train_summary')
val_summaries_dir = os.path.join(logs_dir, 'val_summary')

for dir in [ckpt_save_dir, train_summaries_dir, val_summaries_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

with open(os.path.join(logs_dir, 'train_config'), 'w') as configfile:
    json.dump(vars(opt), configfile, indent=2)


# create torch dataset
train_dataset = MultiVortexDataset(train_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)
val_dataset = MultiVortexDataset(val_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)

steps_per_epoch = int(len(train_dataset) / BATCH_SIZE)
val_steps_per_epoch = int(len(val_dataset) / BATCH_SIZE)


# create torch dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                   pin_memory=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=4)



# define domain and resolution of the grid
domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW_REF = Fluid(domain=domain)

# points in the staggered grid
points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')


start_epoch = 0
val_best = 10000000.0


# weight initialization
@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', a=0.1)
        torch.nn.init.zeros_(m.bias.data)


# Neural network for Vortex Particle Dynamics
if NETWORK == 'Vortex':
    VortexNet = MultiStepMultiVortexNetwork(depth=opt.depth, hidden_units=opt.hidden_units,
                                            order=opt.order, num_steps=opt.num_time_steps)
else:
    VortexNet = MultiStepInteractionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, num_steps=opt.num_time_steps)


# Load weights weights from other experiments or resume training for the current experiment (if applicable)
if opt.load_weights_ex is not None:
    init_weights_log_dir = os.path.join(opt.logs_dir, opt.load_weights_ex)
    init_weights_ckpt_dir = os.path.join(init_weights_log_dir, 'ckpt')

    checkpoints_files = os.listdir(os.path.join(init_weights_ckpt_dir))
    epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
    init_weights_ckpt_file = os.path.join(init_weights_ckpt_dir, checkpoints_files[epoch_id])
    params = torch.load(init_weights_ckpt_file)['model_state_dict']
    VortexNet.single_step_net.load_state_dict(params)
else:
    VortexNet.apply(init_weights)


# Neural network to gpu
VortexNet.to('cuda:0')
VortexNet.requires_grad_(requires_grad=True).train()


# Optimizer
optimizer = Adam(params=VortexNet.parameters(), lr=opt.lr, weight_decay=opt.l2)

# Restore optimizer state for resuming training (if applicable)
if opt.ex == opt.load_weights_ex:
    optimizer.load_state_dict(torch.load(init_weights_ckpt_file)['optimizer_state_dict'])
    start_epoch = torch.load(init_weights_ckpt_file)['epoch']


# learning rate scheduler
scheduler = StepLR(optimizer, 150, gamma=0.1)


# Training and validation summary writer
train_summary = SummaryWriter(log_dir=train_summaries_dir)
val_summary = SummaryWriter(log_dir=val_summaries_dir)


# Modules to compute loss
train_loss_module = MultiStepLoss(resolution=opt.domain, num_steps=opt.num_time_steps, batch_size=opt.batch_size)
val_loss_module = MultiStepLoss(resolution=opt.domain, num_steps=opt.num_time_steps, batch_size=opt.batch_size)


# Training epochs
for epoch in range(start_epoch, opt.epochs):

    train_dataiter = iter(train_dataloader)
    VortexNet.train()

    print('============================== Starting Epoch; {}/{} ========================================='.format(epoch+1, opt.epochs))
    print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))

    # mini-batch training
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

        inp_vector = torch.stack([y, x, tau, sig], dim=-1)
        vortex_features = VortexNet(inp_vector)

        mse_loss_list, max_loss_list = train_loss_module(vortex_features, velocities)
        mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
        loss = torch.sum(mse_loss_tensor)

        loss.backward()
        optimizer.step()

        print('Epoch: {}, Step: {}/{}, loss: {:.4f}, Max_loss: {:.4f}'.
              format(epoch, step, steps_per_epoch, loss.item(), max_loss_list[-1].item()))


    # validation after an epoch
    VortexNet.eval()

    val_dataiter = iter(val_dataloader)

    val_loss = 0.0

    for val_step in range(val_steps_per_epoch):

        val_batch = next(val_dataiter)
        location = val_batch['location'].to('cuda:0')
        strength = val_batch['strength'].to('cuda:0')
        sigma = val_batch['sigma'].to('cuda:0')
        velocities = [val_batch['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]

        nparticles = location.shape[1]

        y, x = torch.unbind(location, dim=-1)
        tau, sig = strength.view(opt.batch_size, -1), sigma.view(opt.batch_size, -1)

        inp_vector = torch.stack([y, x, tau, sig], dim=-1)
        vortex_features = VortexNet(inp_vector)

        with torch.no_grad():
            mse_loss_list, max_loss_list = val_loss_module(vortex_features, velocities)
            mse_loss_tensor = torch.stack(mse_loss_list, dim=0)
            val_loss = val_loss + torch.sum(mse_loss_tensor * loss_weights)

    val_loss = val_loss / val_steps_per_epoch

    print('Epoch; {}, val_loss: {:.4f}'.format(epoch, val_loss.item()))

    val_summary.add_scalar('val_l2_loss', val_loss.item(), (epoch) * steps_per_epoch)

    # save the best checkpoints
    if val_loss.item() < val_best:
        save_state_dict = {
                'model_state_dict': VortexNet.single_step_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
        }


        ckpt_filename = 'ckpt_{:02d}_val_loss_{:.4f}.pytorch'.format(epoch, val_loss.item())
        ckpt_path = os.path.join(ckpt_save_dir, ckpt_filename)
        torch.save(save_state_dict, ckpt_path)

        val_best = val_loss.item()

    scheduler.step(epoch=epoch)