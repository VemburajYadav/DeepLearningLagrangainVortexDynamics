import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from core.custom_functions import particle_vorticity_to_velocity
from torch.utils.tensorboard import SummaryWriter
from core.datasets import SingleVortexDataset
import argparse
import matplotlib.pyplot as plt
from core.networks import SimpleNN
import os

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_small', help='path to save training '
                                                                  'summaries and checkpoints')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-4, help='weight for l2 regularization')
parser.add_argument('--log_dir', type=str, default='./logs_10', help='path to save training '
                                                                  'summaries and checkpoints')
parser.add_argument('--depth', type=int, default=3, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=1024, help='number of neurons in hidden layers')
parser.add_argument('--dataset_path', type=str, default='data/VWData.npz', help='Path to the dataset in .npz format')

opt = parser.parse_args()

data_dir = opt.data_dir

train_dir = os.path.join(data_dir, 'train')
eval_dir = os.path.join(data_dir, 'eval')
test_dir = os.path.join(data_dir, 'test')

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


model = SimpleNN(depth=opt.depth, hidden_units=opt.hidden_units, in_features=4, out_features=4)
model.apply(init_weights)
#
model.requires_grad_(requires_grad=True).train()
model.to('cuda:0')
#
optimizer = Adam(params=model.parameters(), lr=0.0001, weight_decay=opt.l2)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
val_best = 100000.0
# train_summary = SummaryWriter(log_dir=train_summaries_dir)
# val_summary = SummaryWriter(log_dir=val_summaries_dir)
#
for epoch in range(opt.epochs):
    train_dataiter = iter(train_dataloader)

    model.train()
    print('Starting Epoch: {}'.format(epoch+1))
    for step in range(steps_per_epoch):
        data_dict = next(train_dataiter)

        loc = data_dict['location']
        tau = data_dict['strength']
        sig = data_dict['sigma']
        vel1 = data_dict['velocity1']

        pvel0 = torch.tensor([0.0, 0.0], dtype=torch.float32).view(1, -1).repeat(opt.batch_size, 1)

        inp = torch.cat([torch.squeeze(tau, dim=1),
                         torch.squeeze(sig, dim=1), pvel0], dim=-1)

        inp_gpu = inp.to('cuda:0')
        loc0_gpu = torch.squeeze(loc, dim=1).to('cuda:0')
        vel1_gpu = vel1.to('cuda:0')

        optimizer.zero_grad()
        out_gpu = model(inp_gpu)

        tau, sig, pu0, pv0 = torch.unbind(out_gpu, dim=-1)
        dy, dx, dtau, dsig = torch.unbind(out_gpu, dim=-1)

        new_pos = torch.unsqueeze(torch.stack([dy, dx], dim=-1) + loc0_gpu, dim=1)
        new_tau = torch.unsqueeze(tau + dtau, dim=-1)
        new_sig = torch.unsqueeze(torch.unsqueeze(sig + dsig, dim=-1), dim=-1)

        optimizer.step()
        #
        # train_summary.add_scalar('l1_loss', loss, epoch * steps_per_epoch + step)
#         print('Step {}/{}: loss: {:.4f}'.format(step, steps_per_epoch, loss.item()))
    #
    # model.eval()
    # val_dataiter = iter(val_dataloader)
    #
    # with torch.no_grad():
    #
    #     val_loss = 0.0
    #
    #     for val_step in range(val_steps_per_epoch):
    #         val_batch_x, val_batch_y = next(val_dataiter)
    #         val_batch_x_gpu = val_batch_x.to('cuda:0')
    #         val_batch_y_gpu = val_batch_y.to('cuda:0')
    #
    #         pred_y = model(val_batch_x_gpu)
    #         val_loss = val_loss + torch.nn.functional.l1_loss(pred_y, val_batch_y_gpu)
    #
    #     val_loss = val_loss / val_steps_per_epoch
    #
    #     print('After epoch; {}, val_loss: {:.4f}'.format(epoch+1, val_loss))
    #
    # if val_loss.item() < val_best:
    #     save_state_dict = {
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'epoch': epoch+1,
    #         'val_loss': val_loss,
    #     }
    #
    #     train_summary.add_scalar('val_l1_loss', val_loss, (epoch+1) * steps_per_epoch)
    #
    #     ckpt_filename = 'ckpt_{:02d}_val_loss_{:.4f}.pytorch'.format(epoch+1, val_loss.item())
    #     ckpt_path = os.path.join(ckpt_dir_path, ckpt_filename)
    #     torch.save(save_state_dict, ckpt_path)
    #
    #     val_best = val_loss.item()