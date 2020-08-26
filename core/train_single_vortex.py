import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from core.custom_functions import particle_vorticity_to_velocity
from torch.utils.tensorboard import SummaryWriter
from core.datasets import SingleVortexDataset
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import SimpleNN, MultiStepLossModule
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[128, 128], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_128x128_8000',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--num_time_steps', type=int, default=20, help='train the network on loss for more than 1 time step')
parser.add_argument('--stride', type=int, default=1, help='skip intermediate time frames corresponding to stride during training f'
                                                          'or multiple time steps')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='weight for l2 regularization')
parser.add_argument('--ex', type=str, default='train_demo_128x128_8000_T20_init_T5_lr_1e-4_weighted', help='name of the experiment')
parser.add_argument('--load_weights_ex', type=str, default='train_demo_128x128_8000_T5', help='name of the experiment')
parser.add_argument('--depth', type=int, default=3, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=512, help='number of neurons in hidden layers')

opt = parser.parse_args()

NUM_TIME_STEPS = opt.num_time_steps
STRIDE = opt.stride
RESOLUTION = opt.domain
BATCH_SIZE = opt.batch_size
loss_weights = torch.tensor([0.9**i for i in range(NUM_TIME_STEPS)], dtype=torch.float32, device=('cuda:0'))

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


train_dataset = SingleVortexDataset(train_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)
val_dataset = SingleVortexDataset(val_dir, num_steps=NUM_TIME_STEPS, stride=STRIDE)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                   pin_memory=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=2)

steps_per_epoch = int(len(train_dataset) / BATCH_SIZE)
val_steps_per_epoch = int(len(val_dataset) / BATCH_SIZE)

start_epoch = 0

val_best = 10000000.0

@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', a=0.1)
        torch.nn.init.zeros_(m.bias.data)


loss_module = MultiStepLossModule(depth=opt.depth, hidden_units=opt.hidden_units, in_features=4,
                                  out_features=4, num_steps=opt.num_time_steps, batch_size=opt.batch_size,
                                  resolution=(RESOLUTION[0], RESOLUTION[1]), batch_norm=True)

if opt.load_weights_ex is not None:
    init_weights_log_dir = os.path.join('../logs', opt.load_weights_ex)
    init_weights_ckpt_dir = os.path.join(init_weights_log_dir, 'ckpt')

    checkpoints_files = os.listdir(os.path.join(init_weights_ckpt_dir))
    epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
    init_weights_ckpt_file = os.path.join(init_weights_ckpt_dir, checkpoints_files[epoch_id])
    params = torch.load(init_weights_ckpt_file)['model_state_dict']
    loss_module.net.load_state_dict(params)
else:
    loss_module.apply(init_weights)

loss_module.to('cuda:0')
loss_module.requires_grad_(requires_grad=True).train()

optimizer = Adam(params=loss_module.parameters(), lr=opt.lr, weight_decay=opt.l2)

if opt.ex == opt.load_weights_ex:
    optimizer.load_state_dict(torch.load(init_weights_ckpt_file)['optimizer_state_dict'])
    start_epoch = torch.load(init_weights_ckpt_file)['epoch']

scheduler = StepLR(optimizer, 20, gamma=0.1)
train_summary = SummaryWriter(log_dir=train_summaries_dir)
val_summary = SummaryWriter(log_dir=val_summaries_dir)


for epoch in range(start_epoch, opt.epochs):

    train_dataiter = iter(train_dataloader)
    loss_module.train()

    print('============================== Starting Epoch; {}/{} ========================================='.format(epoch+1, opt.epochs))
    print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))
    for step in range(steps_per_epoch):

        batch_data_dict = next(train_dataiter)
        location = batch_data_dict['location'].to('cuda:0')
        strength = batch_data_dict['strength'].to('cuda:0')
        sigma = batch_data_dict['sigma'].to('cuda:0')
        velocities = [batch_data_dict['velocities'][i].to('cuda:0') for i in range(NUM_TIME_STEPS + 1)]
#
        optimizer.zero_grad()

        v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')
        u = torch.zeros(BATCH_SIZE, dtype=torch.float32, device='cuda:0')

        y, x = torch.unbind(location.view(-1, 2), dim=-1)
        tau, sig = strength.view(-1), sigma.view(-1)

        inp_vector = torch.stack([y, x, tau, sig, v, u], dim=-1)
        loss_tensor = loss_module(inp_vector, velocities) * loss_weights
        loss = torch.sum(loss_tensor)
        loss.backward()
        optimizer.step()

        print('Epoch: {}, Step: {}/{}, loss: {:.4f}'.format(epoch, step, steps_per_epoch, loss.item()))

    loss_module.eval()

    with torch.no_grad():

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

            y, x = torch.unbind(location.view(-1, 2), dim=-1)
            tau, sig = strength.view(-1), sigma.view(-1)

            inp_vector = torch.stack([y, x, tau, sig, v, u], dim=-1)
            loss_tensor = loss_module(inp_vector, velocities) * loss_weights
            val_loss = val_loss + torch.sum(loss_tensor)

        val_loss = val_loss / val_steps_per_epoch

        print('After epoch; {}, val_loss: {:.4f}'.format(epoch+1, val_loss.item()))

        val_summary.add_scalar('val_l2_loss', val_loss.item(), (epoch + 1) * steps_per_epoch)

        if val_loss.item() < val_best:
            save_state_dict = {
                'model_state_dict': loss_module.net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
                'val_loss': val_loss,
            }

            ckpt_filename = 'ckpt_{:02d}_val_loss_{:.4f}.pytorch'.format(epoch+1, val_loss.item())
            ckpt_path = os.path.join(ckpt_save_dir, ckpt_filename)
            torch.save(save_state_dict, ckpt_path)

            val_best = val_loss.item()

    scheduler.step()
