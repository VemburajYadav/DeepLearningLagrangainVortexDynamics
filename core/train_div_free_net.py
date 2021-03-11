import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from core.datasets import DivFreeNetDataset
from core.custom_functions import *
import argparse
import matplotlib.pyplot as plt
from phi.flow import *
from core.networks import *
import os
from core.velocity_derivs import *
import json

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[120, 120], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='../'
                                                    'data/bc_net_dataset_p10_gaussian_visc_8000',
                    help='path to save training summaries and checkpoints')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for training')
parser.add_argument('--n_domain_pts', type=int, default=500, help='Number of points to sample for divegence loss')
parser.add_argument('--n_boundary_pts', type=int, default=50, help='Number of points to sample for boundary condition loss')
parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='weight for l2 regularization')
parser.add_argument('--logs_dir', type=str, default='../logs', help='directory to save checkpoints and training summaries')
parser.add_argument('--ex', type=str, default='BCNet_2', help='name of the experiment')
parser.add_argument('--load_weights_ex', type=str, default=None, help='name of the experiment')
parser.add_argument('--sampling_type', type=str, default='both',
                    help='strategy to sample points for PINN training. '
                         'Options: both, grid-only, non-grid-only')
parser.add_argument('--depth', type=int, default=5, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=100, help='number of neurons in hidden layers')
parser.add_argument('--order', type=int, default=2, help='derivatives of velocity fields for interaction. Either 0, 1 or 2')




# Parse input arguments
opt = parser.parse_args()

RESOLUTION = opt.domain
BATCH_SIZE = opt.batch_size
N_DOMAIN_PTS = opt.n_domain_pts
N_BOUNDARY_PTS = opt.n_boundary_pts
SAMPLING_TYPE = opt.sampling_type
data_dir = opt.data_dir


# get the directories with the data
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
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


# define domain and resolution of the grid
domain = Domain(resolution=opt.domain, boundaries=CLOSED)
FLOW_REF = Fluid(domain=domain)

# points in the staggered grid
points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')


# create torch dataset
train_dataset = DivFreeNetDataset(train_dir, n_dom_points=N_DOMAIN_PTS, n_b_points=N_BOUNDARY_PTS,
                                  use_frac=1.0, sampling_type=SAMPLING_TYPE)
val_dataset = DivFreeNetDataset(val_dir, n_dom_points=N_DOMAIN_PTS, n_b_points=N_BOUNDARY_PTS, use_frac=1.0,
                                sampling_type=SAMPLING_TYPE)


# create torch dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                   pin_memory=True, num_workers=4)

VAL_BATCH_SIZE = 8
val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=4)


# Training and validation summary writer
train_summary = SummaryWriter(log_dir=train_summaries_dir)
val_summary = SummaryWriter(log_dir=val_summaries_dir)


start_epoch = 0
val_best = 10000000.0


# weight initialization
@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', a=0.1)
        torch.nn.init.zeros_(m.bias.data)


# Indices for different values in the feature vector from the dataset
b_vel_index = torch.tensor([2, 3], device='cuda:0')  # velocity inidces
loc_index = torch.tensor([0, 1], device='cuda:0')  # location indices

if opt.order == 0:
    feat_index = torch.arange(4).to('cuda:0')  # indices for the input to neural network
    deriv_index = torch.arange(2, 4).to('cuda:0')  # indices for velocity and higher order derivatives of velocity
    b_normal_index = torch.tensor([4, 5], device='cuda:0')  # indices for the normal vectors for boundary features
elif opt.order == 1:
    feat_index = torch.arange(8).to('cuda:0')  # indices for the input to neural network
    deriv_index = torch.arange(2, 8).to('cuda:0')  # indices for velocity and higher order derivatives of velocity
    b_normal_index = torch.tensor([8, 9], device='cuda:0')  # indices for the normal vectors for boundary features
elif opt.order == 2:
    feat_index = torch.arange(14).to('cuda:0')  # indices for the input to neural network
    deriv_index = torch.arange(2, 14).to('cuda:0')  # indices for velocity and higher order derivatives of velocity
    b_normal_index = torch.tensor([14, 15], device='cuda:0')  # indices for the normal vectors for boundary features
elif opt.order == 3:
    feat_index = torch.arange(22).to('cuda:0')  # indices for the input to neural network
    deriv_index = torch.arange(2, 22).to('cuda:0')  # indices for velocity and higher order derivatives of velocity
    b_normal_index = torch.tensor([22, 23], device='cuda:0')  # indices for the normal vectors for boundary features


# Boundary condition neural network
BCNet = BoundaryConditionNetwork(depth=opt.depth, hidden_units=opt.hidden_units, order=opt.order)


# Load weights weights from other experiments or resume training for the current experiment (if applicable)
if opt.load_weights_ex is not None:
    init_weights_log_dir = os.path.join(opt.logs_dir, opt.load_weights_ex)
    init_weights_ckpt_dir = os.path.join(init_weights_log_dir, 'ckpt')

    checkpoints_files = os.listdir(os.path.join(init_weights_ckpt_dir))
    epoch_id = np.argmax(np.array([int(i.split('_')[1]) for i in checkpoints_files]))
    init_weights_ckpt_file = os.path.join(init_weights_ckpt_dir, checkpoints_files[epoch_id])
    params = torch.load(init_weights_ckpt_file)['model_state_dict']
    BCNet.load_state_dict(params)
else:
    BCNet.apply(init_weights)


# Neural network to gpu
BCNet.to('cuda:0')
BCNet.requires_grad_(requires_grad=True).train()


# Optimizer
optimizer = Adam(params=BCNet.parameters(), lr=opt.lr, weight_decay=opt.l2)



# Restore optimizer state for resuming training (if applicable)
if opt.ex == opt.load_weights_ex:
    optimizer.load_state_dict(torch.load(init_weights_ckpt_file)['optimizer_state_dict'])
    start_epoch = torch.load(init_weights_ckpt_file)['epoch']


# learning rate scheduler
scheduler = StepLR(optimizer, 150, gamma=0.1)


# number of batches in an epoch
steps_per_epoch = int(len(train_dataset) / BATCH_SIZE)
val_steps_per_epoch = int(len(val_dataset) / VAL_BATCH_SIZE)


# Training epochs
for epoch in range(start_epoch, opt.epochs):

    train_dataiter = iter(train_dataloader)
    BCNet.train()

    print('============================== Starting Epoch; {}/{} ========================================='.format(epoch+1, opt.epochs))
    print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))

    # mini-batch training
    for step in range(steps_per_epoch):

        batch_data_dict = next(train_dataiter)

        # features from dataset for domain points (grid  +  non-grid)
        dom_points_feat = batch_data_dict['domain_points'].to('cuda:0')
        # features from dataset for boundary points
        b_points_feat = batch_data_dict['b_points'].to('cuda:0')

        # neural network input for boundary points
        b_features = torch.index_select(b_points_feat, dim=-1, index=feat_index)

        # normal vectors for boundary points
        b_normals = torch.index_select(b_points_feat, dim=-1, index=b_normal_index)

        # velocity (due to vortex particles) at boundary points
        b_vortex_vel = torch.index_select(b_points_feat, dim=-1, index=b_vel_index)

        # location for domain points (grid  +  non-grid)
        dom_points = torch.index_select(dom_points_feat, dim=-1, index=loc_index)
        dom_pts_y, dom_pts_x = torch.unbind(dom_points, dim=-1)
        dom_pts_y_re = dom_pts_y.view(-1)
        dom_pts_x_re = dom_pts_x.view(-1)

        dom_pts_y_re.requires_grad_(True)
        dom_pts_x_re.requires_grad_(True)

        dom_pts_loc = torch.stack([dom_pts_y_re.view(BATCH_SIZE, -1), dom_pts_x_re.view(BATCH_SIZE, -1)], dim=-1)
        dom_pts_deriv = torch.index_select(dom_points_feat, dim=-1, index=deriv_index)


        # neural network input for domain points (grid  +  non-grid)
        dom_features = torch.cat([dom_pts_loc, dom_pts_deriv], dim=-1)
        # correction velocity from BCNet for domain points (grid  +  non-grid)
        dom_corr_vel = BCNet(dom_features)
#
        dom_corr_vel_y, dom_corr_vel_x = torch.unbind(dom_corr_vel, dim=-1)
        dom_corr_vel_y_re = dom_corr_vel_y.view(-1)
        dom_corr_vel_x_re = dom_corr_vel_x.view(-1)


        # compute divergence for domain points (grid  +  non-grid)
        div_du_dx = torch.autograd.grad(torch.unbind(dom_corr_vel_x_re, dim=-1), dom_pts_x_re, create_graph=True,
                                        allow_unused=True)[0]
        div_dv_dy = torch.autograd.grad(torch.unbind(dom_corr_vel_y_re, dim=-1), dom_pts_y_re, create_graph=True,
                                        allow_unused=True)[0]

        div_du_dx = div_du_dx.view(BATCH_SIZE, -1)
        div_dv_dy = div_dv_dy.view(BATCH_SIZE, -1)

        # divergence loss for domain points (grid  +  non-grid)
        total_div = div_du_dx + div_dv_dy
        div_loss = torch.sum(total_div ** 2) / total_div.nelement()

        # correction velocity from BCNet for boundary points
        b_corr_vel = BCNet(b_features)

        # total velocity for boundary points: velocity (due to vortex particles) + correction velocity
        b_total_vel = b_corr_vel + b_vortex_vel
        b_normal_vel = b_total_vel * b_normals

        # boundary condition loss
        bc_loss = torch.sum(b_normal_vel ** 2) / b_normal_vel.nelement()


        # computing mse loss for domain points (grid only)
        if SAMPLING_TYPE == 'grid-only' or SAMPLING_TYPE == 'both':
            grid_vel_y = batch_data_dict['grid_y_vel'].to('cuda:0')
            grid_vel_x = batch_data_dict['grid_x_vel'].to('cuda:0')
            n_grid_y_pts = grid_vel_y.shape[1]
            n_grid_x_pts = grid_vel_x.shape[1]

            grid_y_index = torch.arange(n_grid_y_pts, device='cuda:0')
            grid_x_index = torch.arange(n_grid_y_pts, (n_grid_y_pts + n_grid_x_pts), device='cuda:0')

            pred_grid_corr_vel_y = torch.index_select(dom_corr_vel_y, dim=-1, index=grid_y_index)
            pred_grid_corr_vel_x = torch.index_select(dom_corr_vel_x, dim=-1, index=grid_x_index)

            dom_vortex_vel = torch.index_select(dom_points_feat, dim=-1, index=b_vel_index)
            dom_vortex_vel_y, dom_vortex_vel_x = torch.unbind(dom_vortex_vel, dim=-1)

            dom_vortex_vel_grid_y = torch.index_select(dom_vortex_vel_y, dim=-1, index=grid_y_index)
            dom_vortex_vel_grid_x = torch.index_select(dom_vortex_vel_x, dim=-1, index=grid_x_index)

            pred_grid_vel_y = dom_vortex_vel_grid_y + pred_grid_corr_vel_y
            pred_grid_vel_x = dom_vortex_vel_grid_x + pred_grid_corr_vel_x

            loss_grid_y = torch.sum((grid_vel_y - pred_grid_vel_y)**2) / grid_vel_y.nelement()
            loss_grid_x = torch.sum((grid_vel_x - pred_grid_vel_x)**2) / grid_vel_x.nelement()

            mse_loss = loss_grid_y + loss_grid_x
            loss = mse_loss + div_loss + bc_loss

            print('Epoch: {}, Step: {}/{}, MSE Loss: {:.4f}, Div Loss: {:.4f}, BC Loss: {:.4f}, Loss: {:.4f}'.
                  format(epoch, step, steps_per_epoch, mse_loss.item(), div_loss.item(), bc_loss.item(), loss.item()))

        else:
            loss = div_loss + bc_loss

            print('Epoch: {}, Step: {}/{}, Div Loss: {:.4f}, BC Loss: {:.4f}, Loss: {:.4f}'.
                  format(epoch, step, steps_per_epoch, div_loss.item(), bc_loss.item(), loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # validation after an epoch
    BCNet.eval()

    val_dataiter = iter(val_dataloader)

    val_loss = 0.0
    val_div_loss = 0.0
    val_bc_loss = 0.0
    val_mse_loss = 0.0

    for val_step in range(val_steps_per_epoch):
        val_batch_data_dict = next(val_dataiter)
        dom_points_feat = val_batch_data_dict['domain_points'].to('cuda:0')
        b_points_feat = val_batch_data_dict['b_points'].to('cuda:0')

        b_features = torch.index_select(b_points_feat, dim=-1, index=feat_index)
        b_normals = torch.index_select(b_points_feat, dim=-1, index=b_normal_index)
        b_vortex_vel = torch.index_select(b_points_feat, dim=-1, index=b_vel_index)

        dom_points = torch.index_select(dom_points_feat, dim=-1, index=loc_index)
        dom_pts_y, dom_pts_x = torch.unbind(dom_points, dim=-1)
        dom_pts_y_re = dom_pts_y.view(-1)
        dom_pts_x_re = dom_pts_x.view(-1)

        dom_pts_y_re.requires_grad_(True)
        dom_pts_x_re.requires_grad_(True)

        dom_pts_loc = torch.stack([dom_pts_y_re.view(VAL_BATCH_SIZE, -1), dom_pts_x_re.view(VAL_BATCH_SIZE, -1)], dim=-1)
        dom_pts_deriv = torch.index_select(dom_points_feat, dim=-1, index=deriv_index)

        dom_features = torch.cat([dom_pts_loc, dom_pts_deriv], dim=-1)
        dom_corr_vel = BCNet(dom_features)

        dom_corr_vel_y, dom_corr_vel_x = torch.unbind(dom_corr_vel, dim=-1)
        dom_corr_vel_y_re = dom_corr_vel_y.view(-1)
        dom_corr_vel_x_re = dom_corr_vel_x.view(-1)

        div_du_dx = torch.autograd.grad(torch.unbind(dom_corr_vel_x_re, dim=-1), dom_pts_x_re, create_graph=True,
                                        allow_unused=True)[0]
        div_dv_dy = torch.autograd.grad(torch.unbind(dom_corr_vel_y_re, dim=-1), dom_pts_y_re, create_graph=True,
                                        allow_unused=True)[0]

        div_du_dx = div_du_dx.view(VAL_BATCH_SIZE, -1).detach().clone()
        div_dv_dy = div_dv_dy.view(VAL_BATCH_SIZE, -1).detach().clone()

        with torch.no_grad():
            total_div = div_du_dx + div_dv_dy
            div_loss = torch.sum(total_div ** 2) / total_div.nelement()

            b_corr_vel = BCNet(b_features)
            b_total_vel = b_corr_vel + b_vortex_vel
            b_normal_vel = b_total_vel * b_normals

            bc_loss = torch.sum(b_normal_vel ** 2) / b_normal_vel.nelement()

            if SAMPLING_TYPE == 'grid-only' or SAMPLING_TYPE == 'both':
                grid_vel_y = val_batch_data_dict['grid_y_vel'].to('cuda:0')
                grid_vel_x = val_batch_data_dict['grid_x_vel'].to('cuda:0')
                n_grid_y_pts = grid_vel_y.shape[1]
                n_grid_x_pts = grid_vel_x.shape[1]

                grid_y_index = torch.arange(n_grid_y_pts, device='cuda:0')
                grid_x_index = torch.arange(n_grid_y_pts, (n_grid_y_pts + n_grid_x_pts), device='cuda:0')

                pred_grid_corr_vel_y = torch.index_select(dom_corr_vel_y, dim=-1, index=grid_y_index)
                pred_grid_corr_vel_x = torch.index_select(dom_corr_vel_x, dim=-1, index=grid_x_index)

                dom_vortex_vel = torch.index_select(dom_points_feat, dim=-1, index=b_vel_index)
                dom_vortex_vel_y, dom_vortex_vel_x = torch.unbind(dom_vortex_vel, dim=-1)

                dom_vortex_vel_grid_y = torch.index_select(dom_vortex_vel_y, dim=-1, index=grid_y_index)
                dom_vortex_vel_grid_x = torch.index_select(dom_vortex_vel_x, dim=-1, index=grid_x_index)

                pred_grid_vel_y = dom_vortex_vel_grid_y + pred_grid_corr_vel_y
                pred_grid_vel_x = dom_vortex_vel_grid_x + pred_grid_corr_vel_x

                loss_grid_y = torch.sum((grid_vel_y - pred_grid_vel_y) ** 2) / grid_vel_y.nelement()
                loss_grid_x = torch.sum((grid_vel_x - pred_grid_vel_x) ** 2) / grid_vel_x.nelement()

                mse_loss = loss_grid_y + loss_grid_x
                loss = mse_loss + div_loss + bc_loss
            else:
                mse_loss = torch.tensor([0.0], dtype=torch.float32, device='cuda:0')
                loss = div_loss + bc_loss

            val_loss = val_loss + loss
            val_div_loss = val_div_loss + div_loss
            val_bc_loss = val_bc_loss + bc_loss
            val_mse_loss = val_mse_loss + mse_loss

    val_loss = val_loss / val_steps_per_epoch
    val_div_loss = val_div_loss / val_steps_per_epoch
    val_bc_loss = val_bc_loss / val_steps_per_epoch
    val_mse_loss = val_mse_loss / val_steps_per_epoch

    print('Epoch; {}, val_loss: {:.4f}, val_div_loss: {:.4f}, '
          'val_bc_loss: {:.4f}, val_mse_loss: {:.4f}'.format(epoch,
                                                             val_loss.item(), val_div_loss.item(),
                                                             val_bc_loss.item(), val_mse_loss.item()))

    val_summary.add_scalar('val_l2_loss', val_loss.item(), (epoch) * steps_per_epoch)

    if val_loss.item() < val_best:
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

    scheduler.step(epoch=epoch)










