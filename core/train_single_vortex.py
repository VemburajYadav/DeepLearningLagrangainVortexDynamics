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
from phi.flow import Domain
from core.networks import SimpleNN
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[256, 256], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_1',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='weight for l2 regularization')
parser.add_argument('--ex', type=str, default='train_demo_4', help='name of the experiment')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=1024, help='number of neurons in hidden layers')

opt = parser.parse_args()

RESOLUTION = opt.domain
domain = Domain(RESOLUTION)
sample_points = domain.center_points()

data_dir = opt.data_dir

train_dir = os.path.join(data_dir, 'train')
eval_dir = os.path.join(data_dir, 'eval')
test_dir = os.path.join(data_dir, 'test')

logs_dir = os.path.join('../logs', opt.ex)
ckpt_save_dir = os.path.join(logs_dir, 'ckpt')
train_summaries_dir = os.path.join(logs_dir, 'train_summary')
val_summaries_dir = os.path.join(logs_dir, 'val_summary')

for dir in [ckpt_save_dir, train_summaries_dir, val_summaries_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

train_dataset = SingleVortexDataset(train_dir)
val_dataset = SingleVortexDataset(eval_dir)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, pin_memory=True)

steps_per_epoch = int(len(train_dataset) / opt.batch_size)
val_steps_per_epoch = int(len(val_dataset) / opt.batch_size)

@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', a=0.1)
        torch.nn.init.zeros_(m.bias.data)


model_ = SimpleNN(depth=opt.depth, hidden_units=opt.hidden_units, in_features=4, out_features=4)
model_.apply(init_weights)

model_.to('cuda:0')
model_.requires_grad_(requires_grad=True).train()

optimizer = Adam(params=model_.parameters(), lr=opt.lr, weight_decay=opt.l2)
scheduler = StepLR(optimizer, 100, gamma=0.1)
train_summary = SummaryWriter(log_dir=train_summaries_dir)
val_summary = SummaryWriter(log_dir=val_summaries_dir)


def execute_batch(data_dict, model, points, return_loss=None):

    loc = data_dict['location']
    tau = data_dict['strength']
    sig = data_dict['sigma']

    pvel0 = torch.tensor([0.0, 0.0], dtype=torch.float32).view(1, -1).repeat(opt.batch_size, 1)

    inp = torch.cat([torch.squeeze(tau, dim=1),
                     torch.squeeze(sig, dim=1), pvel0], dim=-1)

    inp_gpu = inp.to('cuda:0')
    loc0_gpu = torch.squeeze(loc, dim=1).to('cuda:0')

    out_gpu = model(inp_gpu)

    tau, sig, pu0, pv0 = torch.unbind(inp_gpu, dim=-1)
    dy, dx, dtau, dsig = torch.unbind(out_gpu, dim=-1)

    dy = dy * 0.1
    dx = dx * 0.1
    dsig = F.softplus(dsig)

    new_pos = torch.unsqueeze(torch.stack([dy, dx], dim=-1) + loc0_gpu, dim=1)
    new_tau = torch.unsqueeze(tau + dtau, dim=-1)
    new_sig = torch.unsqueeze(torch.unsqueeze(sig + dsig, dim=-1), dim=-1)

    points_gpu = torch.tensor(points, dtype=torch.float32, device='cuda:0')
    pred_vel1 = particle_vorticity_to_velocity(new_pos, new_tau, new_sig, points_gpu)

    if return_loss is None:
        return pred_vel1
    else:
        vel1 = data_dict['velocity1']
        vel1_gpu = vel1.to('cuda:0')
        loss = F.l1_loss(pred_vel1, vel1_gpu, reduction='sum') / opt.batch_size
        return pred_vel1, loss


val_best = 100000.0

for epoch in range(opt.epochs):
    train_dataiter = iter(train_dataloader)

    model_.train()
    print('====================Starting Epoch: {} ================================='.format(epoch+1))

    for step in range(steps_per_epoch):
        batch_data_dict = next(train_dataiter)
        optimizer.zero_grad()
        _, loss_ = execute_batch(batch_data_dict, model_, sample_points, return_loss=True)
        loss_.backward()
        optimizer.step()

        print('Epoch: {} Step {}/{}, loss: {:.4f}'.format(epoch+1, step+1, steps_per_epoch, loss_.item()))

        train_summary.add_scalar('l2_loss', loss_.item(), epoch * steps_per_epoch + step)

    model_.eval()
    val_dataiter = iter(val_dataloader)

    with torch.no_grad():

        val_loss = 0.0

        for val_step in range(val_steps_per_epoch):
            val_data_dict = next(val_dataiter)
            val_pred_vel, val_loss_batch = execute_batch(val_data_dict, model_, sample_points, return_loss=True)
            val_loss = val_loss + val_loss_batch

        val_loss = val_loss / val_steps_per_epoch

        print('After epoch; {}, val_loss: {:.4f}'.format(epoch+1, val_loss.item()))

    if val_loss.item() < val_best:
        save_state_dict = {
            'model_state_dict': model_.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch+1,
            'val_loss': val_loss,
        }

        train_summary.add_scalar('val_l2_loss', val_loss, (epoch+1) * steps_per_epoch)

        ckpt_filename = 'ckpt_{:02d}_val_loss_{:.4f}.pytorch'.format(epoch+1, val_loss.item())
        ckpt_path = os.path.join(ckpt_save_dir, ckpt_filename)
        torch.save(save_state_dict, ckpt_path)

        val_best = val_loss.item()

    scheduler.step()