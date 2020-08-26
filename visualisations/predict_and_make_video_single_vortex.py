import torch
import numpy as np
from core.networks import SimpleNN
from phi.flow import Domain, Fluid, OPEN
import torch.nn.functional as F
from core.custom_functions import particle_vorticity_to_velocity
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.animation as animation

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[128, 128], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_128x128_8000/train/sim_000100',
                    help='path to the directory with data to make predictions')
parser.add_argument('--load_weights_ex', type=str, default='train_demo_128x128_8000_T20_init_T5', help='name of the experiment to load weights from')
parser.add_argument('--depth', type=int, default=3, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=512, help='number of neurons in hidden layers')
parser.add_argument('--num_time_steps', type=int, default=100, help='number of time steps to make predictions for')

opt = parser.parse_args()

case_dir = opt.case_path
NUM_TIME_STEPS = opt.num_time_steps

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

velocities = [np.load(os.path.join(case_dir, 'velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'))['arr_0']
              for i in range(NUM_TIME_STEPS + 1)]

velocities_gpu = [torch.tensor(velocities[i], dtype=torch.float32, device='cuda:0') for i in range(NUM_TIME_STEPS + 1)]

loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)
points_y = torch.tensor(FLOW.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

v0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')
u0 = torch.zeros((1, 1), dtype=torch.float32, device='cuda:0')

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

inp_feature = torch.cat([loc_gpu.view(-1, 2), tau_gpu.view(-1, 1), sig_gpu.view(-1, 1), v0, u0], dim=-1)
feature_list = [inp_feature]
pred_velocites = [velocities_gpu[0]]
losses= [0.0]

with torch.no_grad():

    for step in range(NUM_TIME_STEPS):

        y, x, tau, sig, v, u = torch.unbind(feature_list[step], dim=-1)
        net_inp = torch.stack([tau, sig, v, u], dim=-1)
        net_out = network(net_inp)
        dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
        v_new = dy * 0.1
        u_new = dx * 0.1
        tau_new = tau + dtau
        sig_new = sig + F.softplus(dsig)
        x_new = x + dx * 0.1
        y_new = y + dy * 0.1
        feature_list.append(torch.stack([y_new, x_new, tau_new, sig_new, v_new, u_new], dim=-1))

        loc_new = torch.stack([y_new, x_new], dim=-1).view(1, 1, 2)

        vel_y = particle_vorticity_to_velocity(loc_new, tau_new.view(1, 1), sig_new.view(1, 1, 1), points_y)
        vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
        vel_x = particle_vorticity_to_velocity(loc_new, tau_new.view(1, 1), sig_new.view(1, 1, 1), points_x)
        vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
        vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
        pred_velocites.append(vel)
        losses.append(F.mse_loss(vel, velocities_gpu[step + 1], reduction='sum'))

# create OpenCV video writer

max_val = velocities[0].max()
min_val = -max_val
# loop over your images
fig, ax = plt.subplots()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ims = []
for i in range(NUM_TIME_STEPS + 1):

    img = np.concatenate([velocities[i][0, :, :, 1], pred_velocites[i].cpu().numpy()[0, :, :, 1]], axis=-1)
    im1 = ax1.imshow(velocities[i][0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val, animated=True)
    im2 = ax2.imshow(pred_velocites[i].cpu().numpy()[0, :, :, 1], cmap='RdYlBu', vmin=min_val, vmax=max_val, animated=True)
    ax1.set_title('Target Velocity- y')
    ax2.set_title('Predicted Velocity- y')

    # ax.set_title('Time step: {}, Loss: {:.2f}'.format(i, losses[i]))
    ims.append([im1, im2])

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False,
                                repeat_delay=1000)
plt.show()
