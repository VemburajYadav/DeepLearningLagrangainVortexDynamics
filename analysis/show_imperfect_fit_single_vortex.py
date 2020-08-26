import torch
import numpy as np
from phi.flow import Domain, Fluid, OPEN
import torch.nn.functional as F
from core.custom_functions import particle_vorticity_to_velocity
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[128, 128], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_128x128_8000/train/sim_000100',
                    help='path to the directory with data to make predictions')

opt = parser.parse_args()

case_dir = opt.case_path

location = np.load(os.path.join(case_dir, 'location_000000.npz'))['arr_0']
strength = np.load(os.path.join(case_dir, 'strength_000000.npz'))['arr_0']
sigma = np.load(os.path.join(case_dir, 'sigma_000000.npz'))['arr_0']

velocity_t1 = np.load(os.path.join(case_dir, 'velocity_000001.npz'))['arr_0']
velocity_t2 = np.load(os.path.join(case_dir, 'velocity_000002.npz'))['arr_0']

velocity_t1_gpu = torch.tensor(velocity_t1, dtype=torch.float32, device='cuda:0')
velocity_t2_gpu = torch.tensor(velocity_t2, dtype=torch.float32, device='cuda:0')

loc_gpu = torch.tensor(location, dtype=torch.float32, device='cuda:0')
tau_gpu = torch.tensor(strength, dtype=torch.float32, device='cuda:0')
sig_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda:0')

dtau_vals = torch.linspace(0, 0.5, 100, dtype=torch.float32, device='cuda:0')

if tau_gpu.view(-1).item() > 0:
    dtau_vals = dtau_vals * (-1.0)

dsig_vals = torch.linspace(0, 5.0, 100, dtype=torch.float32, device='cuda:0')

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)
points_y = torch.tensor(FLOW.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
points_x = torch.tensor(FLOW.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

cat_y = torch.zeros((1, opt.domain[0] + 1, 1), dtype=torch.float32, device='cuda:0')
cat_x = torch.zeros((1, 1, opt.domain[0] + 1), dtype=torch.float32, device='cuda:0')

loss_matrix_t1 = torch.zeros((100, 100), dtype=torch.float32, device='cuda:0')
loss_matrix_t2 = torch.zeros((100, 100), dtype=torch.float32, device='cuda:0')

with torch.no_grad():

    for tau_id in range(100):
        for sig_id in range(100):
            vel_y = particle_vorticity_to_velocity(loc_gpu, tau_gpu + dtau_vals[tau_id], sig_gpu + dsig_vals[sig_id], points_y)
            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_x = particle_vorticity_to_velocity(loc_gpu, tau_gpu + dtau_vals[tau_id], sig_gpu + dsig_vals[sig_id], points_x)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
            vel = torch.stack([torch.cat([vel_yy, cat_y], dim=-1), torch.cat([vel_xx, cat_x], dim=-2)], dim=-1)
            loss_matrix_t1[tau_id, sig_id] = F.mse_loss(vel, velocity_t1_gpu, reduction='sum')
            loss_matrix_t2[tau_id, sig_id] = F.mse_loss(vel, velocity_t2_gpu, reduction='sum')


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(loss_matrix_t1.cpu().numpy(), vmin=0.0)

plt.subplot(1, 2, 2)
plt.imshow(loss_matrix_t2.cpu().numpy(), vmin=0.0)

plt.show()




