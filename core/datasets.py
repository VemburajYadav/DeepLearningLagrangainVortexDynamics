import torch
import os
import numpy as np
from phi.flow import Domain, Fluid, OPEN, StaggeredGrid, CLOSED

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



class ViscousVortexDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path, num_steps=1, stride=1, resolution=[128, 128]):

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
        viscosity = np.load(os.path.join(case, 'viscosity.npz'))['arr_0']

        velocities = [np.squeeze(np.load(os.path.join(case, self.filenames[i]))['arr_0'], axis=0) for i in range(self.T + 1)]

        return {'location': np.squeeze(location, axis=0),
                'strength': strength,
                'sigma': np.squeeze(sigma, axis=0),
                'viscosity': viscosity,
                'velocities': velocities}


class VortexBoundariesDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path, num_steps=1, stride=1, resolution=[128, 128]):

        super(VortexBoundariesDataset, self).__init__()
        self.dir_path = dir_path
        self.resolution = resolution
        self.T = num_steps
        self.stride = stride
        self.sim_paths = [os.path.join(self.dir_path, sim) for sim in os.listdir(self.dir_path)]
        self.filenames = ['velocity_' + '0' * (6 - len(str(i))) + str(i) + '.npz'
                          for i in range(0, self.stride * (self.T + 1), self.stride)]
        self.filenames.append('velocity_div_000000.npz')
        self.domain = Domain(resolution=self.resolution, boundaries=OPEN)
        self.sample_points = self.domain.center_points()


    def __len__(self):
        return len(os.listdir(self.dir_path))

    def __getitem__(self, item):

        case = self.sim_paths[item]

        location = np.load(os.path.join(case, 'location_000000.npz'))['arr_0']
        strength = np.reshape(np.load(os.path.join(case, 'strength_000000.npz'))['arr_0'], (-1, 1))
        sigma = np.load(os.path.join(case, 'sigma_000000.npz'))['arr_0']

        velocities = [np.squeeze(np.load(os.path.join(case, self.filenames[i]))['arr_0'], axis=0)
                      for i in range(len(self.filenames))]

        return {'location': np.squeeze(location, axis=0),
                'strength': strength,
                'sigma': np.squeeze(sigma, axis=0),
                'velocities': velocities}



class DivFreeNetDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path, resolution=(100, 100), n_dom_points=100, n_b_points=10,
                 sample_all=False, use_frac = 1.0, sampling_type='both'):

        super(DivFreeNetDataset, self).__init__()

        self.dir_path = dir_path
        self.resolution = resolution
        self.sim_paths = [os.path.join(self.dir_path, sim) for sim in os.listdir(self.dir_path)]
        N = len(self.sim_paths)
        n = int(N * use_frac)
        self.sim_paths = self.sim_paths[0: n]
        self.dom_filename = 'features_domain.npz'
        self.boundaries_filename = 'features_boundaries.npz'
        self.points_y_filename = 'features_points_y.npz'
        self.points_x_filename = 'features_points_x.npz'
        self.vel_div_free_filename = 'velocity_div_000000.npz'
        self.sampling_type = sampling_type

        if self.sampling_type == 'grid-only':
            self.n_grid_dom_points = n_dom_points
            self.n_grid_dom_points_y = self.n_grid_dom_points // 2
            self.n_grid_dom_points_x = self.n_grid_dom_points - self.n_grid_dom_points_y
        elif self.sampling_type == 'non-grid-only':
            self.n_ngrid_dom_points = n_dom_points
            self.n_ngrid_b_points = n_b_points
        elif self.sampling_type == 'both':
            self.n_grid_dom_points = n_dom_points // 2
            self.n_grid_dom_points_y = self.n_grid_dom_points // 2
            self.n_grid_dom_points_x = self.n_grid_dom_points - self.n_grid_dom_points_y
            self.n_ngrid_dom_points = n_dom_points - self.n_grid_dom_points
            self.n_ngrid_b_points = n_b_points

        # self.n_dom_points = n_dom_points
        # self.n_b_points = n_b_points
        self.sample_all = sample_all


    def __len__(self):
        return len(self.sim_paths)

    def __getitem__(self, item):

        case = self.sim_paths[item]

        if self.sample_all:
            dom_features = np.squeeze(np.load(os.path.join(case, self.dom_filename))['arr_0'])
            b_features = np.squeeze(np.load(os.path.join(case, self.boundaries_filename))['arr_0'])
            data_dict = {'domain_points': dom_features,
                         'b_points': b_features}
        else:
            if self.sampling_type == 'grid-only':
                grid_y_features = np.squeeze(np.load(os.path.join(case, self.points_y_filename))['arr_0'])
                grid_x_features = np.squeeze(np.load(os.path.join(case, self.points_x_filename))['arr_0'])
                div_free_vel = np.squeeze(np.load(os.path.join(case, self.vel_div_free_filename))['arr_0'])

                div_free_vel_y = div_free_vel[:, :-1, 0]
                div_free_vel_x = div_free_vel[:-1, :, 1]

                div_free_vel_y = np.reshape(div_free_vel_y, (-1))
                div_free_vel_x = np.reshape(div_free_vel_x, (-1))

                n_grid_y_pts = grid_y_features.shape[0]
                n_grid_x_pts = grid_x_features.shape[0]

                ids_grid_y = np.arange(n_grid_y_pts)
                np.random.shuffle(ids_grid_y)

                ids_grid_x = np.arange(n_grid_x_pts)
                np.random.shuffle(ids_grid_x)

                sample_grid_y_features = grid_y_features[ids_grid_y[0: self.n_grid_dom_points_y], :]
                sample_grid_x_features = grid_x_features[ids_grid_x[0: self.n_grid_dom_points_x], :]

                sample_grid_vel_y = div_free_vel_y[ids_grid_y[0: self.n_grid_dom_points_y]]
                sample_grid_vel_x = div_free_vel_x[ids_grid_x[0: self.n_grid_dom_points_x]]

                sample_all_dom_features = np.concatenate([sample_grid_y_features, sample_grid_x_features], axis=0)

                data_dict = {'domain_points': sample_all_dom_features,
                             'grid_y_vel': sample_grid_vel_y,
                             'grid_x_vel': sample_grid_vel_x}

                return data_dict


            elif self.sampling_type == 'non-grid-only':
                dom_features = np.squeeze(np.load(os.path.join(case, self.dom_filename))['arr_0'])
                b_features = np.squeeze(np.load(os.path.join(case, self.boundaries_filename))['arr_0'])

                n_dom_pts = dom_features.shape[0]
                n_b_pts = b_features.shape[0]

                ids_dom = np.arange(n_dom_pts)
                np.random.shuffle(ids_dom)

                ids_b = np.arange(n_b_pts)
                np.random.shuffle(ids_b)

                sample_dom_features = dom_features[ids_dom[0: self.n_ngrid_dom_points], :]
                sample_b_features = b_features[ids_b[0: self.n_ngrid_b_points], :]

                data_dict = {'domain_points': sample_dom_features,
                             'b_points': sample_b_features}

                return data_dict

            else:
                grid_y_features = np.squeeze(np.load(os.path.join(case, self.points_y_filename))['arr_0'])
                grid_x_features = np.squeeze(np.load(os.path.join(case, self.points_x_filename))['arr_0'])
                div_free_vel = np.squeeze(np.load(os.path.join(case, self.vel_div_free_filename))['arr_0'])

                div_free_vel_y = div_free_vel[:, :-1, 0]
                div_free_vel_x = div_free_vel[:-1, :, 1]

                div_free_vel_y = np.reshape(div_free_vel_y, (-1))
                div_free_vel_x = np.reshape(div_free_vel_x, (-1))

                n_grid_y_pts = grid_y_features.shape[0]
                n_grid_x_pts = grid_x_features.shape[0]

                ids_grid_y = np.arange(n_grid_y_pts)
                np.random.shuffle(ids_grid_y)

                ids_grid_x = np.arange(n_grid_x_pts)
                np.random.shuffle(ids_grid_x)

                sample_grid_y_features = grid_y_features[ids_grid_y[0: self.n_grid_dom_points_y], :]
                sample_grid_x_features = grid_x_features[ids_grid_x[0: self.n_grid_dom_points_x], :]

                sample_grid_vel_y = div_free_vel_y[ids_grid_y[0: self.n_grid_dom_points_y]]
                sample_grid_vel_x = div_free_vel_x[ids_grid_x[0: self.n_grid_dom_points_x]]

                dom_features = np.squeeze(np.load(os.path.join(case, self.dom_filename))['arr_0'])
                b_features = np.squeeze(np.load(os.path.join(case, self.boundaries_filename))['arr_0'])

                n_dom_pts = dom_features.shape[0]
                n_b_pts = b_features.shape[0]

                ids_dom = np.arange(n_dom_pts)
                np.random.shuffle(ids_dom)

                ids_b = np.arange(n_b_pts)
                np.random.shuffle(ids_b)

                sample_dom_features = dom_features[ids_dom[0: self.n_ngrid_dom_points], :]
                sample_b_features = b_features[ids_b[0: self.n_ngrid_b_points], :]

                sample_all_dom_features = np.concatenate([sample_grid_y_features, sample_grid_x_features,
                                                          sample_dom_features], axis=0)

                data_dict = {'domain_points': sample_all_dom_features,
                             'b_points': sample_b_features,
                             'grid_y_vel': sample_grid_vel_y,
                             'grid_x_vel': sample_grid_vel_x}

                return data_dict





