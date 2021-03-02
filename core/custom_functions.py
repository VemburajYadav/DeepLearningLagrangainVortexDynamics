import torch
import numpy as np



class GaussianFalloffKernelVelocity(torch.nn.Module):

    def __init__(self):
        super(GaussianFalloffKernelVelocity, self).__init__()

    def forward(self, vortex_feature, points):
        y, x, tau, sig = torch.unbind(vortex_feature, dim=-1)

        batch_size = vortex_feature.shape[0]
        nparticles = vortex_feature.shape[1]
        loc = torch.stack([y, x], dim=-1).view(batch_size, -1, 2)
        tau = tau.view(batch_size, -1)
        sigma = sig.view(batch_size, -1, 1)
        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)

        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)

        falloff_value = (1.0 - torch.exp(-sq_distance / src_sigma ** 2)) / (2.0 * np.pi * sq_distance)
        strength = src_strength * falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)

        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=-2)

        return velocity



class GaussianFalloffKernelVorticity(torch.nn.Module):

    def __init__(self):
        super(GaussianFalloffKernelVorticity, self).__init__()

    def forward(self, vortex_feature, points):
        y, x, tau, sig = torch.unbind(vortex_feature, dim=-1)

        batch_size = vortex_feature.shape[0]
        nparticles = vortex_feature.shape[1]
        loc = torch.stack([y, x], dim=-1).view(batch_size, -1, 2)
        tau = tau.view(batch_size, -1)
        sigma = sig.view(batch_size, -1, 1)
        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)

        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)

        falloff_value = (torch.exp(-sq_distance / src_sigma ** 2)) / (np.pi * src_sigma**2)
        strength = src_strength * falloff_value

        vorticity = torch.sum(strength, dim=-2)

        return vorticity


















