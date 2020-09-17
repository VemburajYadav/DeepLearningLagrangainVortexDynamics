import torch
from core.custom_functions import \
    particle_vorticity_to_velocity_gaussian, particle_vorticity_to_velocity_offset_gaussian
from core.custom_functions import GaussianFalloffKernel, OffsetGaussianFalloffKernel
import torch.nn.functional as F
from phi.flow import Domain, OPEN, Fluid

class SimpleNN(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, in_features=4, out_features=4, batch_norm=True):

        super(SimpleNN, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units
        self.in_features = in_features
        self.out_features = out_features

        self.layers = []

        for layer in range(self.depth):
            if layer == 0:
                in_feat = self.in_features
            else:
                in_feat = self.hidden_units

            if batch_norm:
                self.layers.append(torch.nn.Sequential(torch.nn.Linear(in_features=in_feat,
                                                                   out_features=self.hidden_units),
                                                   torch.nn.LeakyReLU(negative_slope=0.1),
                                                   torch.nn.BatchNorm1d(num_features=self.hidden_units)))
            else:
                self.layers.append(torch.nn.Sequential(torch.nn.Linear(in_features=in_feat,
                                                                   out_features=self.hidden_units),
                                                   torch.nn.LeakyReLU(negative_slope=0.1)))

        self.hidden_layers = torch.nn.ModuleList(self.layers)
        self.output_layer = torch.nn.Linear(in_features=self.hidden_units,
                                            out_features=self.out_features)

    def forward(self, x):

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)

        output = self.output_layer(x)

        return output



class VortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 kernel='offset-gaussian', norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.]):

        super(VortexNetwork, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel

        if kernel == 'gaussian':
            self.in_features = 4
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 6
            self.out_features = 6

        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)

    def forward(self, inp):

        if self.kernel == 'gaussian':
            y, x, tau, sig, v, u = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s
            v = v / self.pos_s
            u = u / self.pos_s

            inp_vec = torch.stack([tau, sig, v, u], dim=-1)
            net_out = self.net(inp_vec)

            dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau
            sig_new = sig + dsig
            v_new = y_new - y
            u_new = x_new - x

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m,
                               v_new * self.pos_s, u_new * self.pos_s], dim=-1)

            return out


        elif self.kernel == 'offset-gaussian':
            y, x, tau, sig, v, u, off, sig_l = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s
            sig_l = (sig_l - self.sig_m) / self.sig_s
            off = off / self.pos_s
            v = v / self.pos_s
            u = u / self.pos_s

            inp_vec = torch.stack([tau, sig, v, u, off, sig_l], dim=-1)

            net_out = self.net(inp_vec)

            dy, dx, dtau, dsig, doff, dsig_l = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau
            sig_new = sig + F.softplus(dsig)
            v_new = y_new - y
            u_new = x_new - x
            off_new = off + F.softplus(doff)
            sig_l_new = sig_l + dsig_l

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m,
                               v_new * self.pos_s, u_new * self.pos_s, off_new * self.pos_s,
                               sig_l_new * self.sig_s + self.sig_m], dim=-1)

            return out



class MultiStepVortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0]):

        super(MultiStepVortexNetwork, self).__init__()

        self.single_step_net = VortexNetwork(depth=depth, hidden_units=hidden_units,
                                             batch_norm=batch_norm, kernel=kernel,
                                             norm_mean=norm_mean, norm_stddev=norm_stddev)

        self. multi_step_net = torch.nn.ModuleList([self.single_step_net for step in range(num_steps)])

    def forward(self, inp):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i])
            vortex_features.append(net_out)

        return vortex_features



class MultiStepLoss(torch.nn.Module):

    def __init__(self, kernel='gaussian', resolution=(128, 128), num_steps=1, batch_size=1):

        super(MultiStepLoss, self).__init__()


        self.resolution = resolution
        self.batch_size = batch_size
        domain = Domain(resolution=self.resolution, boundaries=OPEN)
        FLOW_REF = Fluid(domain=domain)
        self.points_y = torch.tensor(FLOW_REF.velocity.data[0].points.data, dtype=torch.float32, device='cuda:0')
        self.points_x = torch.tensor(FLOW_REF.velocity.data[1].points.data, dtype=torch.float32, device='cuda:0')

        self.cat_y = torch.zeros((self.batch_size, self.resolution[0] + 1, 1), dtype=torch.float32, device='cuda:0')
        self.cat_x = torch.zeros((self.batch_size, 1, self.resolution[0] + 1), dtype=torch.float32, device='cuda:0')

        self.kernel = kernel

        if kernel == 'gaussian':
            self.n_features = 4
            self.falloff_kernel = GaussianFalloffKernel()
        elif kernel == 'offset-gaussian':
            self.n_features = 6
            self.falloff_kernel = OffsetGaussianFalloffKernel()

        self.multi_step_falloff_kernel = torch.nn.ModuleList([self.falloff_kernel for step in range(num_steps + 1)])


    def forward(self, vortex_features, target_velocities):

        losses = []

        max_losses = []

        for i, kernel in enumerate(self.multi_step_falloff_kernel):

            vel_y = kernel(vortex_features[i], self.points_y)
            vel_x = kernel(vortex_features[i], self.points_x)

            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)

            pred_vel = torch.stack([torch.cat([vel_yy, self.cat_y], dim=-1), torch.cat([vel_xx, self.cat_x], dim=-2)], dim=-1)

            loss = F.mse_loss(pred_vel, target_velocities[i], reduction='sum') / self.batch_size
            loss_max = torch.max((pred_vel - target_velocities[i])**2)

            losses.append(loss)
            max_losses.append(loss_max)

        return losses, max_losses


class MultiStepLossModule(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, in_features=4,
                 out_features=4, num_steps=2, batch_size=16, resolution=(128, 128), batch_norm=True):

        super(MultiStepLossModule, self).__init__()

        self.resolution = resolution
        self.batch_size = batch_size
        domain = Domain(resolution=self.resolution, boundaries=OPEN)
        FLOW_REF = Fluid(domain=domain)
        self.points_y = FLOW_REF.velocity.data[0].points.data
        self.points_x = FLOW_REF.velocity.data[1].points.data
        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=in_features, out_features=out_features, batch_norm=batch_norm)

        self.net_list = torch.nn.ModuleList([self.net for i in range(num_steps)])


    def forward(self, input_vec, target_velocities):

        y, x, tau, sig, v, u = torch.unbind(input_vec, dim=-1)
        inp_net = torch.stack([tau, sig, v, u], dim=-1)

        losses = []

        points_y = torch.tensor(self.points_y, dtype=torch.float32, device='cuda:0')
        points_x = torch.tensor(self.points_x, dtype=torch.float32, device='cuda:0')

        cat_y = torch.zeros((self.batch_size, self.resolution[0] + 1, 1, 1), dtype=torch.float32, device='cuda:0')
        cat_x = torch.zeros((self.batch_size, 1, self.resolution[0] + 1, 1), dtype=torch.float32, device='cuda:0')

        for i, net in enumerate(self.net_list):
            output_vec = net(inp_net)
            dy, dx, dtau, dsig = torch.unbind(output_vec, dim=-1)
            y = y + dy * 0.1
            x = x + dx * 0.1
            v = dy * 0.1
            u = dx * 0.1
            tau = tau + dtau
            sig = sig + F.softplus(dsig)
            inp_net = torch.stack([tau, sig, v, u], dim=-1)

            loc = torch.unsqueeze(torch.stack([y, x], dim=-1), dim=1)

            vel_y = particle_vorticity_to_velocity_gaussian(loc, tau.view(-1, 1),
                                                   sig.view(-1, 1, 1), points_y)
            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_yy = torch.unsqueeze(vel_yy, dim=-1)

            vel_x = particle_vorticity_to_velocity_gaussian(loc, tau.view(-1, 1),
                                                   sig.view(-1, 1, 1), points_x)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
            vel_xx = torch.unsqueeze(vel_xx, dim=-1)

            pred_vel_y = torch.cat([vel_yy, cat_y], dim=2)
            pred_vel_x = torch.cat([vel_xx, cat_x], dim=1)

            pred_vel = torch.cat([pred_vel_y, pred_vel_x], dim=-1)

            loss_step = F.mse_loss(pred_vel, target_velocities[i + 1], reduction='sum') / self.batch_size
            losses.append(loss_step)

        loss = torch.stack(losses, dim=0) / self.batch_size

        return loss



class MultiStepLossModuleOffsetGaussian(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, in_features=6,
                 out_features=6, num_steps=2, batch_size=16, resolution=(128, 128), batch_norm=True):

        super(MultiStepLossModuleOffsetGaussian, self).__init__()

        self.resolution = resolution
        self.batch_size = batch_size
        domain = Domain(resolution=self.resolution, boundaries=OPEN)
        FLOW_REF = Fluid(domain=domain)
        self.points_y = FLOW_REF.velocity.data[0].points.data
        self.points_x = FLOW_REF.velocity.data[1].points.data
        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=in_features, out_features=out_features, batch_norm=batch_norm)

        self.net_list = torch.nn.ModuleList([self.net for i in range(num_steps)])


    def forward(self, input_vec, target_velocities):

        y, x, tau, sig_r, v, u = torch.unbind(input_vec, dim=-1)

        y = (y - 64.0 / 23.094)
        x = (x - 64.0 / 23.094)

        off = torch.zeros(self.batch_size, dtype=torch.float32, device='cuda:0')
        sig_l = torch.zeros(self.batch_size, dtype=torch.float32, device='cuda:0')
        inp_net = torch.stack([tau / 1.5275, (sig_r - 27.5) / 13.0, v, u, off / 23.094, (sig_l - 27.5) / 13.0], dim=-1)

        losses = []

        points_y = torch.tensor(self.points_y, dtype=torch.float32, device='cuda:0')
        points_x = torch.tensor(self.points_x, dtype=torch.float32, device='cuda:0')

        cat_y = torch.zeros((self.batch_size, self.resolution[0] + 1, 1, 1), dtype=torch.float32, device='cuda:0')
        cat_x = torch.zeros((self.batch_size, 1, self.resolution[0] + 1, 1), dtype=torch.float32, device='cuda:0')

        for i, net in enumerate(self.net_list):
            output_vec = net(inp_net)
            dy, dx, dtau, dsig_r, doff, dsig_l = torch.unbind(output_vec, dim=-1)
            y = y + dy * 0.1
            x = x + dx * 0.1
            v = dy * 0.1
            u = dx * 0.1
            tau = tau + dtau
            sig_r = sig_r + dsig_r
            sig_l = sig_l + dsig_l
            off = off + doff
            inp_net = torch.stack([tau, sig_r, v, u, off, sig_l], dim=-1)

            loc = torch.unsqueeze(torch.stack([y, x], dim=-1), dim=1)

            vel_y = particle_vorticity_to_velocity_offset_gaussian(loc, tau.view(-1, 1),
                                                   sig_r.view(-1, 1, 1), off.view(-1, 1, 1), sig_l.view(-1, 1, 1),  points_y)
            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_yy = torch.unsqueeze(vel_yy, dim=-1)

            vel_x = particle_vorticity_to_velocity_offset_gaussian(loc, tau.view(-1, 1),
                                                   sig_r.view(-1, 1, 1), off.view(-1, 1, 1), sig_l.view(-1, 1, 1), points_x)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
            vel_xx = torch.unsqueeze(vel_xx, dim=-1)

            pred_vel_y = torch.cat([vel_yy, cat_y], dim=2)
            pred_vel_x = torch.cat([vel_xx, cat_x], dim=1)

            pred_vel = torch.cat([pred_vel_y, pred_vel_x], dim=-1)

            loss_step = F.mse_loss(pred_vel, target_velocities[i + 1], reduction='sum') / self.batch_size
            losses.append(loss_step)

        loss = torch.stack(losses, dim=0)

        return loss