import torch
import os
import numpy as np
from phi.flow import Domain, Fluid, OPEN, StaggeredGrid

class SingleVortexDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path, num_steps=1, stride=1, resolution=[128, 128]):

        super(SingleVortexDataset, self).__init__()
        self.dir_path = dir_path
        self.resolution = resolution
        self.T = num_steps
        self.stride = stride
        self.sim_paths = [os.path.join(self.dir_path, sim) for sim in os.listdir(self.dir_path)]
        self.filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'
                          for i in range(0, self.stride * (self.T + 1), self.stride)]
        self.domain = Domain(resolution=self.resolution, boundaries=OPEN)
        self.sample_points = self.domain.center_points()


    def __len__(self):
        return len(os.listdir(self.dir_path))

    def __getitem__(self, item):

        case = self.sim_paths[item]

        location = np.load(os.path.join(case, 'location_000000.npz'))['arr_0']
        strength = np.expand_dims(np.load(os.path.join(case, 'strength_000000.npz'))['arr_0'], axis=-1)
        sigma = np.load(os.path.join(case, 'sigma_000000.npz'))['arr_0']

        velocities = [np.squeeze(np.load(os.path.join(case, self.filenames[i]))['arr_0'], axis=0) for i in range(self.T + 1)]

        return {'location': np.squeeze(location, axis=0),
                'strength': np.squeeze(strength, axis=0),
                'sigma': np.squeeze(sigma, axis=0),
                'velocities': velocities}



class MultiVortexDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path, num_steps=1, stride=1, resolution=[128, 128]):

        super(MultiVortexDataset, self).__init__()
        self.dir_path = dir_path
        self.resolution = resolution
        self.T = num_steps
        self.stride = stride
        self.sim_paths = [os.path.join(self.dir_path, sim) for sim in os.listdir(self.dir_path)]
        self.filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'
                          for i in range(0, self.stride * (self.T + 1), self.stride)]
        self.domain = Domain(resolution=self.resolution, boundaries=OPEN)
        self.sample_points = self.domain.center_points()


    def __len__(self):
        return len(os.listdir(self.dir_path))

    def __getitem__(self, item):

        case = self.sim_paths[item]

        location = np.load(os.path.join(case, 'location_000000.npz'))['arr_0']
        strength = np.reshape(np.load(os.path.join(case, 'strength_000000.npz'))['arr_0'], (-1, 1))
        sigma = np.load(os.path.join(case, 'sigma_000000.npz'))['arr_0']

        velocities = [np.squeeze(np.load(os.path.join(case, self.filenames[i]))['arr_0'], axis=0) for i in range(self.T + 1)]

        return {'location': np.squeeze(location, axis=0),
                'strength': strength,
                'sigma': np.squeeze(sigma, axis=0),
                'velocities': velocities}

