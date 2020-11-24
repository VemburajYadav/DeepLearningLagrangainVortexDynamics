import torch



class GaussianFalloffKernel(torch.nn.Module):

    def __init__(self):
        super(GaussianFalloffKernel, self).__init__()

    def forward(self, vortex_feature, points):
        y, x, tau, sig, v, u = torch.unbind(vortex_feature, dim=-1)

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

        falloff_value = torch.exp(-sq_distance / src_sigma ** 2) / torch.sqrt(sq_distance)
        strength = src_strength * falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)

        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=-2)

        return velocity



class OffsetGaussianFalloffKernel(torch.nn.Module):

    def __init__(self):
        super(OffsetGaussianFalloffKernel, self).__init__()

    def forward(self, vortex_feature, points):
        y, x, tau, sig, v, u, off, sig_l = torch.unbind(vortex_feature, dim=-1)
        # sig_r = sig / (1.0 + fac)
        sig_r = sig - sig_l
        loc = torch.stack([y, x], dim=-1).view(-1, 1, 2)
        tau = tau.view(-1, 1)
        sigma_right = sig_r.view(-1, 1, 1)
        sigma_left = sig_l.view(-1, 1, 1)
        offset = off.view(-1, 1, 1)

        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma_right = torch.unsqueeze(torch.unsqueeze(sigma_right, dim=1), dim=1)
        src_sigma_left = torch.unsqueeze(torch.unsqueeze(sigma_left, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)

        src_offset = torch.unsqueeze(torch.unsqueeze(offset, dim=1), dim=1)
        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)

        offset_dist = torch.sqrt(sq_distance) - src_offset
        right_mask = (offset_dist >= 0.0).to(torch.float32)
        left_mask = (offset_dist < 0.0).to(torch.float32)

        falloff_value = (torch.exp(-offset_dist ** 2 / src_sigma_right ** 2) * right_mask +
                         torch.exp(-offset_dist ** 2 / src_sigma_left ** 2) * left_mask) / torch.sqrt(sq_distance)
        strength = src_strength * falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)

        src_axes = tuple(range(-2, -2 - src_rank, -1))

        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=src_axes)

        return velocity



class GaussExpFalloffKernel(torch.nn.Module):

    def __init__(self, dt=1.0):
        super(GaussExpFalloffKernel, self).__init__()
        self.dt = dt


    def forward(self, vortex_feature, points):
        y, x, tau, sig, c, d = torch.unbind(vortex_feature, dim=-1)

        batch_size = vortex_feature.shape[0]
        nparticles = vortex_feature.shape[1]
        loc = torch.stack([y, x], dim=-1).view(batch_size, -1, 2)
        tau = tau.view(batch_size, -1)
        sigma = sig.view(batch_size, -1, 1)
        c = c.view(batch_size, -1, 1)
        d = d.view(batch_size, -1, 1)

        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)
        src_c = torch.unsqueeze(torch.unsqueeze(c, dim=1), dim=1)
        src_d = torch.unsqueeze(torch.unsqueeze(d, dim=1), dim=1)

        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)

        gaussian_part = torch.exp(-sq_distance / src_sigma**2)
        exp1 = torch.exp(-self.dt**2 * sq_distance / (src_d * src_sigma**2))
        exponential_part = torch.exp(-src_c * exp1) / torch.sqrt(sq_distance + src_c * exp1)
        falloff_value = gaussian_part * exponential_part
        strength = src_strength * falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)

        src_axes = tuple(range(-2, -2 - src_rank, -1))

        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=-2)

        return velocity



class GaussExpFalloffKernelReduced(torch.nn.Module):

    def __init__(self, dt=1.0):
        super(GaussExpFalloffKernelReduced, self).__init__()
        self.dt = dt


    def forward(self, vortex_feature, points):
        y, x, tau, sig, d = torch.unbind(vortex_feature, dim=-1)

        batch_size = vortex_feature.shape[0]
        nparticles = vortex_feature.shape[1]
        loc = torch.stack([y, x], dim=-1).view(batch_size, -1, 2)
        tau = tau.view(batch_size, -1)
        sigma = sig.view(batch_size, -1, 1)
        d = d.view(batch_size, -1, 1)

        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)
        src_d = torch.unsqueeze(torch.unsqueeze(d, dim=1), dim=1)

        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)
        sqrt_sq_distance = torch.sqrt(sq_distance)
        gaussian_part = torch.exp(-sq_distance / src_sigma**2)
        # exp1 = torch.exp(-sq_distance / (src_d * src_sigma**2))
        exponential_part = 1.0 - torch.exp(-sqrt_sq_distance / src_d)
        # exponential_part = torch.exp(-src_c * exp1) / torch.sqrt(sq_distance + src_c * exp1)
        falloff_value = gaussian_part * exponential_part / sqrt_sq_distance
        strength = src_strength * falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)

        src_axes = tuple(range(-2, -2 - src_rank, -1))

        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=-2)

        return velocity



class GaussExpFalloffKernelGrad(torch.nn.Module):

    def __init__(self, dt=1.0):
        super(GaussExpFalloffKernelGrad, self).__init__()
        self.dt = dt


    def forward(self, vortex_feature, points):
        y, x, tau, sig, v, u, c, d = torch.unbind(vortex_feature, dim=-1)

        batch_size = vortex_feature.shape[0]
        nparticles = vortex_feature.shape[1]
        loc = torch.stack([y, x], dim=-1).view(batch_size, -1, 2)
        tau = tau.view(batch_size, -1)
        sigma = sig.view(batch_size, -1, 1)
        c = c.view(batch_size, -1, 1)
        d = d.view(batch_size, -1, 1)

        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)
        src_c = torch.unsqueeze(torch.unsqueeze(c, dim=1), dim=1)
        src_d = torch.unsqueeze(torch.unsqueeze(d, dim=1), dim=1)

        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)
        r = torch.sqrt(sq_distance)
        gaussian_part = torch.exp(-sq_distance / src_sigma**2)
        exp1 = torch.exp(-10.0 * self.dt**2 * sq_distance / (src_d * src_sigma**2))
        denom = torch.sqrt(sq_distance + src_c * exp1)
        exponential_part = torch.exp(-src_c * exp1) / denom
        falloff_value = gaussian_part * exponential_part
        grad_kernel_r = 2.0 * falloff_value * r * (1.0 - src_c * exp1 / (src_d * src_sigma**2)) * (0.5 / denom - denom)
        m = grad_kernel_r * r + falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)
        dist_1 = torch.unsqueeze(dist_1, dim=-1)
        dist_2 = torch.unsqueeze(dist_2, dim=-1)

        grad_u_x = -src_strength * ((dist_1 / dist_2) * m  - (dist_2 / dist_1) * falloff_value)
        grad_u_y = -src_strength * (m + falloff_value)

        grad_v_x = src_strength * (m + falloff_value)
        grad_v_y = src_strength * ((dist_2 / dist_1) * m  - (dist_1 / dist_2) * falloff_value)

        grads = torch.cat([grad_v_y, grad_v_x, grad_u_y, grad_u_x], dim=-1)

        grad_velocities = torch.sum(grads, dim=-2)

        return grad_velocities


class  LorentzianFalloffKernel(torch.nn.Module):

    def __init__(self):
        super(LorentzianFalloffKernel, self).__init__()

    def forward(self, vortex_feature, points):

        y, x, tau, sig, v, u = torch.unbind(vortex_feature, dim=-1)

        loc = torch.stack([y, x], dim=-1).view(-1, 1, 2)
        tau = tau.view(-1, 1)
        sigma = sig.view(-1, 1, 1)
        points_rank = points.ndim - 2
        src_rank = loc.ndim - 2

        points = torch.unsqueeze(points, dim=-2)
        src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
        src_strength = torch.unsqueeze(tau, dim=-1)
        src_sigma = torch.unsqueeze(torch.unsqueeze(sigma, dim=1), dim=1)
        src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)

        distances = points - src_location
        sq_distance = torch.sum(distances ** 2, dim=-1, keepdim=True)

        falloff_value = (src_sigma**2 / (src_sigma**2 + sq_distance)) / torch.sqrt(sq_distance)
        strength = src_strength * falloff_value

        dist_1, dist_2 = torch.unbind(distances, dim=-1)

        src_axes = tuple(range(-2, -2 - src_rank, -1))

        velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
        velocity = torch.sum(velocity, dim=src_axes)

        return velocity


def particle_vorticity_to_velocity_gaussian(loc, tau, sigma, points):
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

    velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
    velocity = torch.sum(velocity, dim=src_axes)

    return velocity


def particle_vorticity_to_velocity_offset_gaussian(loc, tau, sigma_right, offset, sigma_left, points):
    points_rank = points.ndim - 2
    src_rank = loc.ndim - 2

    points = torch.unsqueeze(points, dim=-2)
    src_location = torch.unsqueeze(torch.unsqueeze(loc, dim=1), dim=1)
    src_strength = torch.unsqueeze(tau, dim=-1)
    src_sigma_right = torch.unsqueeze(torch.unsqueeze(sigma_right, dim=1), dim=1)
    src_sigma_left = torch.unsqueeze(torch.unsqueeze(sigma_left, dim=1), dim=1)
    src_strength = torch.unsqueeze(torch.unsqueeze(src_strength, dim=1), dim=1)

    src_offset = torch.unsqueeze(torch.unsqueeze(offset, dim=1), dim=1)
    distances = points - src_location
    sq_distance = torch.sum(distances**2, dim=-1, keepdim=True)

    offset_dist = torch.sqrt(sq_distance) - src_offset
    right_mask = (offset_dist >= 0.0).to(torch.float32)
    left_mask = (offset_dist < 0.0).to(torch.float32)

    falloff_value = (torch.exp(-offset_dist**2 / src_sigma_right**2) * right_mask +
                     torch.exp(-offset_dist**2 / src_sigma_left**2) * left_mask) / torch.sqrt(sq_distance)
    strength = src_strength * falloff_value

    dist_1, dist_2 = torch.unbind(distances, dim=-1)

    src_axes = tuple(range(-2, -2 - src_rank, -1))

    velocity = strength * torch.stack([dist_2, -dist_1], dim=-1)
    velocity = torch.sum(velocity, dim=src_axes)

    return velocity

