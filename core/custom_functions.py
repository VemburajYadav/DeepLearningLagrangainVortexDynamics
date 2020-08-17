import torch


def particle_vorticity_to_velocity(loc, tau, sigma, points, return_only=None):
    points_rank = points.ndim - 2
    src_rank = loc.ndim - 2

    points = torch.unsqueeze(points, dim=-2)
    src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
    src_strength = torch.unsqueeze(tau, dim=-1)
    src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
    src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)

    distances = points - src_location

    sq_distance = torch.sum(distances**2, dim=-1, keepdim=True)
    falloff_value = torch.exp(-sq_distance / src_sigma**2) / torch.sqrt(sq_distance)
    strength = src_strength * falloff_value

    dist_1, dist_2 = torch.unbind(distances, dim=-1)

    src_axes = tuple(range(-2, -2 - src_rank, -1))

    if return_only is None:
        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=src_axes)
        return velocity
    elif return_only == 'y':
        velocity = strength * torch.unsqueeze(dist_2, dim=-1)
        velocity = torch.sum(velocity, dim=src_axes)
        return velocity
    elif return_only == 'x':
        velocity = strength * torch.unsqueeze(-dist_1, dim=-1)
        velocity = torch.sum(velocity, dim=src_axes)
        return velocity
    else:
        raise ValueError('return_only argument must be either "y" or "x" or left as None (default)')
