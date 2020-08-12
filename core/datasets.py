import torch
import os
import numpy as np
from phi.flow import StaggeredGrid, Domain, AngularVelocity, Fluid, OPEN, IncompressibleFlow, World, Scene
from core.utils import gaussian_falloff
from functools import partial


class SingleVortexDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path):

        super(SingleVortexDataset, self).__init__()
        self.dir_path = dir_path
        self.sim_paths = [os.path.join(self.dir_path, sim) for sim in os.listdir(self.dir_path)]
        self.domain = Domain([256, 256], boundaries=OPEN)
        self.FLOW = Fluid(self.domain)

    def __len__(self):
        return len(os.listdir(self.dir_path))

    def __getitem__(self, item):

        case = self.sim_paths[item]

        location = np.load(os.path.join(case, 'location_000000.npz'))['arr_0']
        strength = np.expand_dims(np.load(os.path.join(case, 'strength_000000.npz'))['arr_0'], axis=-1)
        sigma = np.load(os.path.join(case, 'sigma_000000.npz'))['arr_0']

        velocity0_cg = np.load(os.path.join(case, 'velocity_000000.npz'))['arr_0']
        velocity1_cg = np.load(os.path.join(case, 'velocity_000001.npz'))['arr_0']

        return {'location': np.squeeze(location, axis=0),
                'strength': np.squeeze(strength, axis=0),
                'sigma': np.squeeze(sigma, axis=0),
                'velocity0': np.squeeze(velocity0_cg, axis=0),
                'velocity1': np.squeeze(velocity1_cg, axis=0)}

