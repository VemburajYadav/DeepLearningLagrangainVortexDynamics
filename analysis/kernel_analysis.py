import torch
import numpy as np
from phi.flow import Domain, Fluid, OPEN, math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.animation as animation

parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=list, default=[128, 128], help='resolution of the domain (as list: [256, 256])')
parser.add_argument('--case_path', type=str, default='/home/vemburaj/phi/data/single_vortex_dataset_128x128_8000/train/sim_000204',
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

domain = Domain(resolution=opt.domain, boundaries=OPEN)
FLOW= Fluid(domain=domain)
points_y = FLOW.velocity.data[0].points.data
points_x = FLOW.velocity.data[1].points.data

loc_x =int(location[0, 0, 1])

py = points_x[0, :, loc_x, 0]
px = np.array([loc_x] * len(py), dtype=np.float32)

dist = math.sqrt((px - location[0, 0, 1])**2 + (py - location[0, 0, 0])**2)
sq_distance = dist**2
dist_y = py - location[0, 0, 0]

dist_all = math.sqrt((points_x[0, :, :, 0] - location[0, 0, 0])**2 + (points_x[0, :, :, 1] - location[0, 0, 1])**2)
sq_distance_all = dist_all ** 2
dist_y_all = points_x[0, :, :, 0] - location[0, 0, 0]

vel_values = []
kernel_values = []
legend_list = []
kernel_values_all = []

for time_step in range(0, 6, 1):
    vel = velocities[time_step][0, :-1, loc_x, 1]
    kernel_val = vel * dist / (velocities[time_step].max() * (-dist_y)) * math.sign(strength[0, 0])
    kernel_val_all = velocities[time_step][0, :-1, :, 1] * dist_all / (velocities[time_step].max() * (-dist_y_all)) * math.sign(strength[0, 0])
    vel_values.append(vel)
    kernel_values.append(kernel_val)
    kernel_values_all.append(kernel_val_all)
    legend_list.append('Time Step: {}'.format(time_step))

def offset_gaussians(loc_y, mean, std_r, std_l, x):

    sq_dist = (x - loc_y)**2
    d = math.sqrt(sq_dist)

    f = math.exp(-((d - mean) / std_r)**2) * ((d - mean) > 0.0) + math.exp(-((d - mean) / std_l)**2) * ((d - mean) <=  0.0)

    return f

def kernelNextStep(loc_y, loc_x, tau_, sig_, py_, px_):

    dist_ = math.sqrt((px_ - loc_x) ** 2 + (py_ - loc_y) ** 2)
    f = math.exp(-dist_**2 / sig_**2) * math.exp(-(tau_**2 / sig_**2) * math.exp(-2.0 * dist_**2 / sig_**2))
    dist_y_ = py_ - loc_y
    vel = -tau_ * f * dist_y_ / dist_
    return f, vel


def kernelNextStep2(loc_y, loc_x, tau_, sig_, py_, px_):

    dist_ = math.sqrt((px_ - loc_x) ** 2 + (py_ - loc_y) ** 2)
    f = math.exp(-dist_**2 / sig_**2) * math.exp(-(tau_**2 / sig_**2) * math.exp(-2.0 * dist_**2 / sig_**2)) * dist_ \
        / math.sqrt((dist_**2 + (tau_**2 / sig_**2) * math.exp(-2.0 * dist_**2 / sig_**2)))
    dist_y_ = py_ - loc_y
    vel = -tau_ * f * dist_y_ / dist_
    return f, vel

plt.figure()
p = offset_gaussians(location[0, 0, 0], 3.5, 19.63, 2.5115, py)
p1, vel1 = kernelNextStep(location[0,0,0], location[0,0,1], strength[0,0], sigma[0,0,0], py, px)
p2, vel2 = kernelNextStep2(location[0,0,0], location[0,0,1], strength[0,0], sigma[0,0,0], py, px)

# plt.plot(py, kernel_values[0])
# plt.plot(py, kernel_values[1])
# plt.plot(py, p)
plt.plot(py, math.abs(vel_values[0]))
plt.plot(py, math.abs(vel_values[1]))
plt.plot(py, math.abs(vel1))
plt.plot(py, math.abs(vel2))
# plt.plot(py, math.abs(vel_values[1]) / math.abs(vel1))
plt.legend(['T0', 'T1', 'T1_KERNEL', 'T1_KERNEL_2'])
# plt.legend(['Time step: 0', 'Time step: 1', 'Time step: 1 Fit'])
# plt.show()

c, d, sigg = 5.0, 20.0, 25.0
xg = np.linspace(-64.0, 64.0, 100000)
yg = np.exp(-c * np.exp(-d * xg**2 / sigg**2))
yg_2 = np.exp(-c * np.exp(-d * xg**2 / sigg**2)) * np.abs(xg) / np.sqrt(xg**2 + c * np.exp(-d * xg**2))
gausg = np.exp(-xg**2 / sigg**2)
plt.figure()
plt.plot(xg, gausg)
plt.plot(xg, yg)
plt.plot(xg, yg_2)
# plt.plot(xg, gausg * yg / (yg * gausg).max())
plt.plot(xg, gausg * yg)
plt.plot(xg, gausg * yg_2)
plt.legend(['Guassian', 'Kernel', 'Kernel_Angle', 'Product', 'Product_2'])
plt.show()

# plt.figure()
# for i in range(len(vel_values)):
#     plt.plot(math.abs(vel_values[i]))
# plt.legend(legend_list)
# plt.title('Velocity Values')
# plt.show()
#
# plt.figure()
# for i in range(len(vel_values)):
#     plt.plot(kernel_values[i])
# plt.legend(legend_list)
# plt.title('Kernel Values')
# plt.show()
#

# plt.figure()
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(kernel_values_all[i], vmin=kernel_values_all[0].min(), vmax=kernel_values_all[0].max())
# plt.show()



# vel = velocities[0]
# vals1 = velocities[0][0, :, 30, 1]
# vals2 = velocities[20][0, :, 30, 1]
# vals3 = velocities[40][0, :, 30, 1]
# vals4 = velocities[60][0, :, 30, 1]
# vals5 = velocities[80][0, :, 30, 1]
# vals6 = velocities[100][0, :, 30, 1]
#
#
# plt.figure()
# plt.plot(vals1)
# plt.plot(vals2)
# plt.plot(vals3)
# plt.plot(vals4)
# plt.plot(vals5)
# plt.plot(vals6)
# plt.show()