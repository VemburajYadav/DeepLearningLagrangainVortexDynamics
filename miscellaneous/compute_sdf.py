import torch

def compute_sdf_rectangular_domain(locations_y, locations_x, sy, sx):

    y_sdf = torch.min(locations_y, sy - locations_y)
    x_sdf = torch.min(locations_x, sx - locations_x)

    sdf = torch.min(y_sdf, x_sdf)

    return sdf


