import torch
# from core.custom_functions import \
#     particle_vorticity_to_velocity_gaussian, particle_vorticity_to_velocity_offset_gaussian
from core.custom_functions import *
import torch.nn.functional as F
from phi.flow import Domain, OPEN, Fluid
import numpy as np
from core.compute_sdf import *

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
                                                   torch.nn.LayerNorm(self.hidden_units)))
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
            self.in_features = 2
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 4
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.in_features = 4
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.in_features = 3
            self.out_features = 5

        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)

    def forward(self, inp):

        if self.kernel == 'gaussian':
            y, x, tau, sig = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig], dim=-1)
            net_out = self.net(inp_vec)

            dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau
            sig_new = sig + F.softplus(dsig)
            v_new = y_new - y
            u_new = x_new - x

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m], dim=-1)

            return out.view(-1, 1, 4)


        elif self.kernel == 'offset-gaussian':
            y, x, tau, sig, off, sig_l = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s
            sig_l = (sig_l - self.sig_m) / self.sig_s
            off = off / self.pos_s

            inp_vec = torch.stack([tau, sig, off, sig_l], dim=-1)

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
                               off_new * self.pos_s,
                               sig_l_new * self.sig_s + self.sig_m], dim=-1)

            return out.view(-1, 1, 6)

        elif self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp.view(-1, 5), dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig, c], dim=-1)

            net_out = self.net(inp_vec)
            dy, dx, dtau, dsig, dc, dd = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau * 0.1
            sig_new = sig + dsig * 0.1
            v_new = y_new - y
            u_new = x_new - x
            c_new = F.softplus(dc)
            d_new  = F.softplus(dd)

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m, c_new, d_new], dim=-1)

            return out.view(-1, 1, 6)

        elif self.kernel == 'ExpGaussianRed':
            y, x, tau, sig, d = torch.unbind(inp.view(-1, 5), dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig, d], dim=-1)

            net_out = self.net(inp_vec)
            dy, dx, dtau, dsig, dd = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau * 0.1
            sig_new = sig + dsig * 0.1
            v_new = y_new - y
            u_new = x_new - x
            d_new  = d + F.softplus(dd)

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m,
                               d_new], dim=-1)

            return out.view(-1, 1, 5)



class SingleVortexViscousNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 kernel='offset-gaussian', norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.]):

        super(SingleVortexViscousNetwork, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel

        if kernel == 'gaussian':
            self.in_features = 3
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 5
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.in_features = 5
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.in_features = 4
            self.out_features = 5
        if kernel == 'GaussianVorticity':
            self.in_features = 3
            self.out_features = 4

        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)

    def forward(self, inp, nyu):

        if self.kernel == 'gaussian':
            y, x, tau, sig = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig], dim=-1)
            net_out = self.net(inp_vec)

            dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau
            sig_new = sig + F.softplus(dsig)
            v_new = y_new - y
            u_new = x_new - x

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m], dim=-1)

            return out.view(-1, 1, 4)


        elif self.kernel == 'offset-gaussian':
            y, x, tau, sig, off, sig_l = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s
            sig_l = (sig_l - self.sig_m) / self.sig_s
            off = off / self.pos_s

            inp_vec = torch.stack([tau, sig, off, sig_l], dim=-1)

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
                               off_new * self.pos_s,
                               sig_l_new * self.sig_s + self.sig_m], dim=-1)

            return out.view(-1, 1, 6)

        elif self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp.view(-1, 5), dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig, c], dim=-1)

            net_out = self.net(inp_vec)
            dy, dx, dtau, dsig, dc, dd = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau * 0.1
            sig_new = sig + dsig * 0.1
            v_new = y_new - y
            u_new = x_new - x
            c_new = F.softplus(dc)
            d_new  = F.softplus(dd)

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m, c_new, d_new], dim=-1)

            return out.view(-1, 1, 6)


        elif self.kernel == 'GaussianVorticity':
            y, x, tau, sig = torch.unbind(inp.view(-1, 4), dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig, nyu], dim=-1)

            net_out = self.net(inp_vec)
            dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau * 0.1
            sig_new = sig + dsig * 0.1
            v_new = y_new - y
            u_new = x_new - x

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m], dim=-1)

            return out.view(-1, 1, 4)


        elif self.kernel == 'ExpGaussianRed':
            y, x, tau, sig, d = torch.unbind(inp.view(-1, 5), dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s

            inp_vec = torch.stack([tau, sig, d], dim=-1)

            net_out = self.net(inp_vec)
            dy, dx, dtau, dsig, dd = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau * 0.1
            sig_new = sig + dsig * 0.1
            v_new = y_new - y
            u_new = x_new - x
            d_new  = d + F.softplus(dd)

            out = torch.stack([y_new * self.pos_s + self.pos_m, x_new * self.pos_s + self.pos_m,
                               tau_new * self.tau_s + self.tau_m, sig_new * self.sig_s + self.sig_m,
                               d_new], dim=-1)

            return out.view(-1, 1, 5)


class MultiParticleVortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 kernel='ExpGaussian', norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.], order=1):

        super(MultiParticleVortexNetwork, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel
        self.order = order

        if kernel == 'gaussian':
            self.in_features = 4
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 6
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.falloff_kernel = GaussExpFalloffKernel()
            if self.order == 0:
                self.in_features = 4 + 2
            elif self.order == 1:
                self.in_features = 4 + 2 + 4
            elif self.order == 2:
                self.in_features = 4 + 2 + 4 + 6
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.falloff_kernel = GaussExpFalloffKernelReduced()
            if self.order == 0:
                self.in_features = 3 + 2
            elif self.order == 1:
                self.in_features = 3 + 2 + 4
            elif self.order == 2:
                self.in_features = 3 + 2 + 4 + 6
            elif self.order == 3:
                self.in_features = 3 + 2 + 4 + 6 + 8
            self.out_features = 5
        elif kernel == 'GaussianVorticity':
            self.falloff_kernel = GaussianFalloffKernelVelocity()
            if self.order == 0:
                self.in_features = 2 + 2
            elif self.order == 1:
                self.in_features = 2 + 2 + 4
            elif self.order == 2:
                self.in_features = 2 + 2 + 4 + 6
            elif self.order == 3:
                self.in_features = 2 + 2 + 4 + 6 + 8
            self.out_features = 4

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
            sig_new = sig + F.softplus(dsig)
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

        elif self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_c = torch.index_select(c, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone()], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone()], dim=-1)
                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone()], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig, dc, dd = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                c_new = F.softplus(dc) * 0.1
                d_new  = F.softplus(dd) * 0.01

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new,  c_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out

        elif self.kernel == 'GaussianVorticity':
            y, x, tau, sig = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone()], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone()], dim=-1)
                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone()], dim=-1)

                elif self.order == 3:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), py, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone(),
                                           d3v_dy3.detach().clone(), d3v_dx3.detach().clone(),
                                           d3v_dxdy2.detach().clone(), d3v_dx2dy.detach().clone(),
                                           d3u_dy3.detach().clone(), d3u_dx3.detach().clone(),
                                           d3u_dxdy2.detach().clone(), d3u_dx2dy.detach().clone()], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out

        elif self.kernel == 'ExpGaussianRed':
            y, x, tau, sig, d = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone()], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone()], dim=-1)

                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone()], dim=-1)

                elif self.order == 3:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), py, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
                                           du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0,
                                           d2v_dy2.detach().clone() * 10.0, d2v_dx2.detach().clone() * 10.0,
                                           d2v_dxdy.detach().clone() * 10.0,
                                           d2u_dy2.detach().clone() * 10.0, d2u_dx2.detach().clone() * 10.0,
                                           d2u_dydx.detach().clone() * 10.0,
                                           d3v_dy3.detach().clone() * 10.0, d3v_dx3.detach().clone() * 10.0,
                                           d3v_dxdy2.detach().clone() * 10.0, d3v_dx2dy.detach().clone() * 10.0,
                                           d3u_dy3.detach().clone() * 10.0, d3u_dx3.detach().clone() * 10.0,
                                           d3u_dxdy2.detach().clone() * 10.0, d3u_dx2dy.detach().clone() * 10.0], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig, dd = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out



class ViscousVortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 kernel='ExpGaussian', norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.], order=1):

        super(ViscousVortexNetwork, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel
        self.order = order

        if kernel == 'gaussian':
            self.in_features = 5
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 7
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.falloff_kernel = GaussExpFalloffKernel()
            if self.order == 0:
                self.in_features = 5 + 2
            elif self.order == 1:
                self.in_features = 5 + 2 + 4
            elif self.order == 2:
                self.in_features = 5 + 2 + 4 + 6
            elif self.order == 3:
                self.in_features = 5 + 2 + 4 + 6 + 8
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.falloff_kernel = GaussExpFalloffKernelReduced()
            if self.order == 0:
                self.in_features = 4 + 2
            elif self.order == 1:
                self.in_features = 4 + 2 + 4
            elif self.order == 2:
                self.in_features = 4 + 2 + 4 + 6
            elif self.order == 3:
                self.in_features = 4 + 2 + 4 + 6 + 8
            self.out_features = 5
        elif kernel == 'GaussianVorticity':
            self.falloff_kernel = GaussianFalloffKernelVelocity()
            if self.order == 0:
                self.in_features = 3 + 2
            elif self.order == 1:
                self.in_features = 3 + 2 + 4
            elif self.order == 2:
                self.in_features = 3 + 2 + 4 + 6
            elif self.order == 3:
                self.in_features = 3 + 2 + 4 + 6 + 8
            self.out_features = 4

        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)

    def forward(self, inp, nyu):

        if self.kernel == 'gaussian':
            y, x, tau, sig, v, u = torch.unbind(inp, dim=-1)

            y = (y - self.pos_m) / self.pos_s
            x = (x - self.pos_m) / self.pos_s
            tau = (tau - self.tau_m) / self.tau_s
            sig = (sig - self.sig_m) / self.sig_s
            v = v / self.pos_s
            u = u / self.pos_s

            inp_vec = torch.stack([tau, sig, v, u, nyu], dim=-1)
            net_out = self.net(inp_vec)

            dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
            y_new = y + dy * 0.1
            x_new = x + dx * 0.1
            tau_new = tau + dtau
            sig_new = sig + F.softplus(dsig)
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

            inp_vec = torch.stack([tau, sig, v, u, off, sig_l, nyu], dim=-1)

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

        elif self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_c = torch.index_select(c, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(), nyu], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(), nyu], dim=-1)
                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone() * 10.0, d2v_dx2.detach().clone() * 10.0,
                                           d2v_dxdy.detach().clone() * 10.0,
                                           d2u_dy2.detach().clone() * 10.0, d2u_dx2.detach().clone() * 10.0,
                                           d2u_dydx.detach().clone() * 10.0, nyu], dim=-1)

                elif self.order == 3:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), py, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone() * 10.0, d2v_dx2.detach().clone() * 10.0,
                                           d2v_dxdy.detach().clone() * 10.0,
                                           d2u_dy2.detach().clone() * 10.0, d2u_dx2.detach().clone() * 10.0,
                                           d2u_dydx.detach().clone() * 10.0,
                                           d3v_dy3.detach().clone() * 10.0, d3v_dx3.detach().clone() * 10.0,
                                           d3v_dxdy2.detach().clone() * 10.0, d3v_dx2dy.detach().clone() * 10.0,
                                           d3u_dy3.detach().clone() * 10.0, d3u_dx3.detach().clone() * 10.0,
                                           d3u_dxdy2.detach().clone() * 10.0, d3u_dx2dy.detach().clone() * 10.0, nyu], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig, dc, dd = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                c_new = F.softplus(dc) * 0.1
                d_new  = F.softplus(dd) * 0.1

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new,  c_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out

        elif self.kernel == 'GaussianVorticity':
            y, x, tau, sig = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(), nyu], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(), nyu], dim=-1)
                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone(), nyu], dim=-1)

                elif self.order == 3:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), py, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone(),
                                           d3v_dy3.detach().clone(), d3v_dx3.detach().clone(),
                                           d3v_dxdy2.detach().clone(), d3v_dx2dy.detach().clone(),
                                           d3u_dy3.detach().clone(), d3u_dx3.detach().clone(),
                                           d3u_dxdy2.detach().clone(), d3u_dx2dy.detach().clone(), nyu], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out

        elif self.kernel == 'ExpGaussianRed':
            y, x, tau, sig, d = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(), nyu], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(), nyu], dim=-1)

                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone() * 10.0, d2v_dx2.detach().clone() * 10.0,
                                           d2v_dxdy.detach().clone() * 10.0,
                                           d2u_dy2.detach().clone() * 10.0, d2u_dx2.detach().clone() * 10.0,
                                           d2u_dydx.detach().clone() * 10.0, nyu], dim=-1)

                elif self.order == 3:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), py, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone(),
                                           d3v_dy3.detach().clone(), d3v_dx3.detach().clone(),
                                           d3v_dxdy2.detach().clone(), d3v_dx2dy.detach().clone(),
                                           d3u_dy3.detach().clone(), d3u_dx3.detach().clone(),
                                           d3u_dxdy2.detach().clone(), d3u_dx2dy.detach().clone()], nyu, dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig, dd = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out


class InteractionNetwork(torch.nn.Module):

    def __init__(self, depth=1, kernel = 'ExpGaussian',
                 hidden_units=100, batch_norm=True, norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.]):

        super(InteractionNetwork, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel

        if kernel == 'gaussian':
            self.in_features = 4 * 2
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 6 * 2
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.in_features = 4 * 2 + 2
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.in_features = 3 * 2 + 2
            self.out_features = 5
        elif kernel == 'GaussianVorticity':
            self.in_features = 2 * 2 + 2
            self.out_features = 4

        self.single_step_net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)


    def forward(self, inp):

        if self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp, dim=-1)

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)

            out_features = []

            for i in range(nparticles):
                p_features = []

                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y, p_x = torch.unbind(p_loc, dim=-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_c = torch.index_select(c, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)

                for j in range(nparticles):
                    if i != j:
                        other_paxes_tensor = torch.tensor([j], device='cuda:0')
                        other_p_loc = torch.index_select(location, dim=-2, index=other_paxes_tensor).view(-1, 2)
                        other_p_y, other_p_x = torch.unbind(other_p_loc, dim=-1)
                        other_p_tau = torch.index_select(tau, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_sig = torch.index_select(sig, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_c = torch.index_select(c, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_d = torch.index_select(d, dim=-1, index=other_paxes_tensor).view(-1)

                        inp_vec = torch.stack([p_tau, p_sig, p_c, p_d,
                                               other_p_tau, other_p_sig, other_p_c, other_p_d,
                                               p_y - other_p_y, p_x - other_p_x], dim=-1)

                        net_out = self.single_step_net(inp_vec)
                        p_features.append(net_out)

                p_out = torch.sum(torch.stack(p_features, dim=-1), dim=-1)
                print(p_out.shape)
                dy, dx, dtau, dsig, dc, dd = torch.unbind(p_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                c_new = F.softplus(dc)
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new,  c_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return  out

        elif self.kernel == 'ExpGaussianRed':

            y, x, tau, sig, d = torch.unbind(inp, dim=-1)

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)

            out_features = []

            for i in range(nparticles):
                p_features = []

                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y, p_x = torch.unbind(p_loc, dim=-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)

                for j in range(nparticles):
                    if i != j:
                        other_paxes_tensor = torch.tensor([j], device='cuda:0')
                        other_p_loc = torch.index_select(location, dim=-2, index=other_paxes_tensor).view(-1, 2)
                        other_p_y, other_p_x = torch.unbind(other_p_loc, dim=-1)
                        other_p_tau = torch.index_select(tau, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_sig = torch.index_select(sig, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_d = torch.index_select(d, dim=-1, index=other_paxes_tensor).view(-1)

                        inp_vec = torch.stack([p_tau, p_sig, p_d,
                                               other_p_tau, other_p_sig, other_p_d,
                                               p_y - other_p_y, p_x - other_p_x], dim=-1)

                        net_out = self.single_step_net(inp_vec)
                        p_features.append(net_out)

                p_out = torch.sum(torch.stack(p_features, dim=-1), dim=-1)

                dy, dx, dtau, dsig, dd = torch.unbind(p_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return  out

        elif self.kernel == 'GaussianVorticity':

            y, x, tau, sig = torch.unbind(inp, dim=-1)

            nparticles = y.shape[1]
            location = torch.stack([y, x], dim=-1)

            out_features = []

            for i in range(nparticles):
                p_features = []

                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y, p_x = torch.unbind(p_loc, dim=-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)

                for j in range(nparticles):
                    if i != j:
                        other_paxes_tensor = torch.tensor([j], device='cuda:0')
                        other_p_loc = torch.index_select(location, dim=-2, index=other_paxes_tensor).view(-1, 2)
                        other_p_y, other_p_x = torch.unbind(other_p_loc, dim=-1)
                        other_p_tau = torch.index_select(tau, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_sig = torch.index_select(sig, dim=-1, index=other_paxes_tensor).view(-1)

                        inp_vec = torch.stack([p_tau, p_sig,
                                               other_p_tau, other_p_sig,
                                               p_y - other_p_y, p_x - other_p_x], dim=-1)

                        net_out = self.single_step_net(inp_vec)

                        p_features.append(net_out)

                p_out = torch.sum(torch.stack(p_features, dim=-1), dim=-1)

                dy, dx, dtau, dsig = torch.unbind(p_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new], dim=-1))

            out = torch.stack(out_features, dim=-2)
            return  out


class ViscousInteractionNetwork(torch.nn.Module):

    def __init__(self, depth=1, kernel = 'ExpGaussian',
                 hidden_units=100, batch_norm=True, norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.]):

        super(ViscousInteractionNetwork, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel

        if kernel == 'gaussian':
            self.in_features = 4 * 2
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 6 * 2
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.in_features = 4 * 2 + 2 + 1
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.in_features = 3 * 2 + 2 + 1
            self.out_features = 5

        self.single_step_net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)


    def forward(self, inp, viscosity):

        if self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp, dim=-1)

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)

            out_features = []

            for i in range(nparticles):
                p_features = []

                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y, p_x = torch.unbind(p_loc, dim=-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_c = torch.index_select(c, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)

                for j in range(nparticles):
                    if i != j:
                        other_paxes_tensor = torch.tensor([j], device='cuda:0')
                        other_p_loc = torch.index_select(location, dim=-2, index=other_paxes_tensor).view(-1, 2)
                        other_p_y, other_p_x = torch.unbind(other_p_loc, dim=-1)
                        other_p_tau = torch.index_select(tau, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_sig = torch.index_select(sig, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_c = torch.index_select(c, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_d = torch.index_select(d, dim=-1, index=other_paxes_tensor).view(-1)

                        inp_vec = torch.stack([p_tau, p_sig, p_c, p_d,
                                               other_p_tau, other_p_sig, other_p_c, other_p_d,
                                               p_y - other_p_y, p_x - other_p_x, viscosity], dim=-1)

                        net_out = self.single_step_net(inp_vec)
                        p_features.append(net_out)

                p_out = torch.sum(torch.stack(p_features, dim=-1), dim=-1)

                dy, dx, dtau, dsig, dc, dd = torch.unbind(p_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                c_new = F.softplus(dc)
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new,  c_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return  out

        elif self.kernel == 'ExpGaussianRed':

            y, x, tau, sig, d = torch.unbind(inp, dim=-1)

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)

            out_features = []

            for i in range(nparticles):
                p_features = []

                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y, p_x = torch.unbind(p_loc, dim=-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)

                for j in range(nparticles):
                    if i != j:
                        other_paxes_tensor = torch.tensor([j], device='cuda:0')
                        other_p_loc = torch.index_select(location, dim=-2, index=other_paxes_tensor).view(-1, 2)
                        other_p_y, other_p_x = torch.unbind(other_p_loc, dim=-1)
                        other_p_tau = torch.index_select(tau, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_sig = torch.index_select(sig, dim=-1, index=other_paxes_tensor).view(-1)
                        other_p_d = torch.index_select(d, dim=-1, index=other_paxes_tensor).view(-1)

                        inp_vec = torch.stack([p_tau, p_sig, p_d,
                                               other_p_tau, other_p_sig, other_p_d,
                                               p_y - other_p_y, p_x - other_p_x, viscosity], dim=-1)

                        net_out = self.single_step_net(inp_vec)
                        p_features.append(net_out)

                p_out = torch.sum(torch.stack(p_features, dim=-1), dim=-1)

                dy, dx, dtau, dsig, dd = torch.unbind(p_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return  out


class MultiParticleVortexNetworkBC(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 kernel='ExpGaussian', norm_mean=[0., 0., 0.], norm_stddev=[1., 1., 1.], order=1, domain=(100, 100)):

        super(MultiParticleVortexNetworkBC, self).__init__()

        self.mean = torch.tensor(norm_mean, dtype=torch.float32, device='cuda:0')
        self.stddev = torch.tensor(norm_stddev, dtype=torch.float32, device='cuda:0')

        self.sy = torch.tensor(domain[0], dtype=torch.float32, device='cuda:0')
        self.sx = torch.tensor(domain[1], dtype=torch.float32, device='cuda:0')

        self.pos_m, self.tau_m, self.sig_m = torch.unbind(self.mean, dim=-1)
        self.pos_s, self.tau_s, self.sig_s = torch.unbind(self.stddev, dim=-1)

        self.kernel = kernel
        self.order = order

        if kernel == 'gaussian':
            self.in_features = 4 + 3
            self.out_features = 4
        elif kernel == 'offset-gaussian':
            self.in_features = 6 + 3
            self.out_features = 6
        elif kernel == 'ExpGaussian':
            self.falloff_kernel = GaussExpFalloffKernel()
            if self.order == 0:
                self.in_features = 4 + 2 + 3
            elif self.order == 1:
                self.in_features = 4 + 2 + 4 + 3
            elif self.order == 2:
                self.in_features = 4 + 2 + 4 + 6 + 3
            self.out_features = 6
        elif kernel == 'ExpGaussianRed':
            self.falloff_kernel = GaussExpFalloffKernelReduced()
            if self.order == 0:
                self.in_features = 3 + 2 + 3
            elif self.order == 1:
                self.in_features = 3 + 2 + 4 + 3
            elif self.order == 2:
                self.in_features = 3 + 2 + 4 + 6 + 3
            elif self.order == 3:
                self.in_features = 3 + 2 + 4 + 6 + 8 + 3
            self.out_features = 5

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
            sig_new = sig + F.softplus(dsig)
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

        elif self.kernel == 'ExpGaussian':
            y, x, tau, sig, c, d = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_c = torch.index_select(c, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone()], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
                                           du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0], dim=-1)
                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_c, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
                                           du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0,
                                           d2v_dy2.detach().clone() * 100.0, d2v_dx2.detach().clone() * 100.0,
                                           d2v_dxdy.detach().clone() * 100.0,
                                           d2u_dy2.detach().clone() * 100.0, d2u_dx2.detach().clone() * 100.0,
                                           d2u_dydx.detach().clone() * 100.0], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig, dc, dd = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                c_new = F.softplus(dc) * 0.1
                d_new  = F.softplus(dd) * 0.1

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new,  c_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out

        elif self.kernel == 'ExpGaussianRed':
            y, x, tau, sig, d = torch.unbind(inp, dim=-1)

            inp_clone = inp.detach().clone()

            batch_size = y.shape[0]
            nparticles = y.shape[1]

            location = torch.stack([y, x], dim=-1)
            location_clone = location.detach().clone()

            loc_clone_y, loc_clone_x = torch.unbind(location_clone, dim=-1)

            loc_clone_y = loc_clone_y.view(-1)
            loc_clone_x = loc_clone_x.view(-1)

            loc_clone_y = loc_clone_y.clone().detach().requires_grad_(True)
            loc_clone_x = loc_clone_x.clone().detach().requires_grad_(True)

            sdf = compute_sdf_rectangular_domain(loc_clone_y, loc_clone_x, self.sy, self.sx)

            grad_y_sdf = torch.autograd.grad(torch.unbind(sdf, dim=-1), loc_clone_y, retain_graph=True, allow_unused=True)[0]
            grad_x_sdf = torch.autograd.grad(torch.unbind(sdf, dim=-1), loc_clone_x, allow_unused=True)[0]

            grad_y_sdf = grad_y_sdf.view((batch_size, nparticles))
            grad_x_sdf = grad_x_sdf.view((batch_size, nparticles))
            sdf = sdf.view((batch_size, nparticles)) / torch.max(self.sy, self.sx)
            loc_clone_y = loc_clone_y.view((batch_size, nparticles))
            loc_clone_x = loc_clone_x.view((batch_size, nparticles))

            # print(location_clone[0:3, 0:4, :])
            # print(sdf[0:3, 0:4])
            # print(grad_y_sdf[0:3, 0:4])
            # print(grad_x_sdf[0:3, 0:4])

            paxes = np.arange(nparticles)

            out_features = []

            for i in range(nparticles):
                paxes_tensor = torch.tensor([i], device='cuda:0')
                p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
                p_y = torch.index_select(y, dim=-1, index=paxes_tensor).view(-1)
                p_x = torch.index_select(x, dim=-1, index=paxes_tensor).view(-1)
                p_tau = torch.index_select(tau, dim=-1, index=paxes_tensor).view(-1)
                p_sig = torch.index_select(sig, dim=-1, index=paxes_tensor).view(-1)
                p_d = torch.index_select(d, dim=-1, index=paxes_tensor).view(-1)
                p_sdf = torch.index_select(sdf, dim=-1, index=paxes_tensor).view(-1)
                p_grad_y_sdf = torch.index_select(grad_y_sdf, dim=-1, index=paxes_tensor).view(-1)
                p_grad_x_sdf = torch.index_select(grad_x_sdf, dim=-1, index=paxes_tensor).view(-1)

                # print(p_sdf.shape, p_grad_y_sdf.shape, p_grad_x_sdf.shape, p_sig.shape)
                py, px = torch.unbind(p_loc, dim=-1)
                py.requires_grad_(True)
                px.requires_grad_(True)
                p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
                other_p_axes = np.delete(paxes, i)
                other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
                other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
                vel_by_other_ps = self.falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
                vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)

                if self.order == 0:
                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           p_sdf.detach().clone(), p_grad_y_sdf.detach().clone(), p_grad_x_sdf.detach().clone()], dim=-1)
                elif self.order == 1:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, allow_unused=True)[0]
                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
                                           du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0,
                                           p_sdf.detach().clone(), p_grad_y_sdf.detach().clone(), p_grad_x_sdf.detach().clone()], dim=-1)

                elif self.order == 2:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone(), dv_dx.detach().clone(),
                                           du_dy.detach().clone(), du_dx.detach().clone(),
                                           d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                           d2v_dxdy.detach().clone(),
                                           d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                           d2u_dydx.detach().clone(),
                                           p_sdf.detach().clone(), p_grad_y_sdf.detach().clone(), p_grad_x_sdf.detach().clone()], dim=-1)

                elif self.order == 3:
                    dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]

                    d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), py, retain_graph=True, allow_unused=True)[0]
                    d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), px, retain_graph=True, allow_unused=True)[0]
                    d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), py, allow_unused=True)[0]

                    inp_vec = torch.stack([p_tau, p_sig, p_d, vel_y.detach().clone(), vel_x.detach().clone(),
                                           dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
                                           du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0,
                                           d2v_dy2.detach().clone() * 10.0, d2v_dx2.detach().clone() * 10.0,
                                           d2v_dxdy.detach().clone() * 10.0,
                                           d2u_dy2.detach().clone() * 10.0, d2u_dx2.detach().clone() * 10.0,
                                           d2u_dydx.detach().clone() * 10.0,
                                           d3v_dy3.detach().clone() * 10.0, d3v_dx3.detach().clone() * 10.0,
                                           d3v_dxdy2.detach().clone() * 10.0, d3v_dx2dy.detach().clone() * 10.0,
                                           d3u_dy3.detach().clone() * 10.0, d3u_dx3.detach().clone() * 10.0,
                                           d3u_dxdy2.detach().clone() * 10.0, d3u_dx2dy.detach().clone() * 10.0,
                                           p_sdf.detach().clone(), p_grad_y_sdf.detach().clone(), p_grad_x_sdf.detach().clone()], dim=-1)

                net_out = self.net(inp_vec)

                dy, dx, dtau, dsig, dd = torch.unbind(net_out, dim=-1)
                y_new = p_y + dy * 0.1
                x_new = p_x + dx * 0.1
                tau_new = p_tau + dtau * 0.1
                sig_new = p_sig + dsig * 0.1
                v_new = y_new - p_y
                u_new = x_new - p_x
                d_new  = F.softplus(dd)

                out_features.append(torch.stack([y_new, x_new, tau_new, sig_new, d_new], dim=-1))

            out = torch.stack(out_features, dim=-2)

            return out


class BoundaryConditionNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], order=1):

        super(BoundaryConditionNetwork, self).__init__()

        self.order = order

        if self.order == 0:
            self.in_features = 4
        elif self.order == 1:
            self.in_features = 8
        elif self.order == 2:
            self.in_features = 14

        self.out_features = 2

        self.net = SimpleNN(depth=depth, hidden_units=hidden_units,
                       in_features=self.in_features, out_features=self.out_features, batch_norm=batch_norm)

    def forward(self, inp):

        y = self.net(inp)
        return y



class MultiStepBoundaryConditionNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True,
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], order=1):

        super(MultiStepBoundaryConditionNetwork, self).__init__()

        self.single_step_net = BoundaryConditionNetwork(depth=depth, hidden_units=hidden_units,
                                                        batch_norm=batch_norm, order=order)

        module_list = [self.single_step_net]
        module_list = module_list + [self.single_step_net for step in range(num_steps - 1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp):

        corr_fields = []

        for i, net in enumerate(self.multi_step_net):
            net_out = net(inp[i])
            corr_fields.append(net_out)

        return corr_fields


class MultiStepMultiVortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], distinct_nets=True, order=1):

        super(MultiStepMultiVortexNetwork, self).__init__()

        self.single_step_net = MultiParticleVortexNetwork(depth=depth, hidden_units=hidden_units,
                                             batch_norm=batch_norm, kernel=kernel,
                                             norm_mean=norm_mean, norm_stddev=norm_stddev, order=order)

        module_list = [self.single_step_net]
        if num_steps > 1:
            if distinct_nets:
                self.single_step_net2 = MultiParticleVortexNetwork(depth=depth, hidden_units=hidden_units,
                                             batch_norm=batch_norm, kernel=kernel,
                                             norm_mean=norm_mean, norm_stddev=norm_stddev, order=order)

                module_list = module_list + [self.single_step_net2 for step in range(num_steps-1)]
            else:
                module_list = module_list + [self.single_step_net for step in range(num_steps-1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i])
            vortex_features.append(net_out)

        return vortex_features



class MultiStepViscousVortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], distinct_nets=True, order=1):

        super(MultiStepViscousVortexNetwork, self).__init__()

        self.single_step_net = ViscousVortexNetwork(depth=depth, hidden_units=hidden_units,
                                                    batch_norm=batch_norm, kernel=kernel,
                                                    norm_mean=norm_mean, norm_stddev=norm_stddev, order=order)

        module_list = [self.single_step_net]
        if num_steps > 1:
            if distinct_nets:
                self.single_step_net2 = ViscousVortexNetwork(depth=depth, hidden_units=hidden_units,
                                                             batch_norm=batch_norm, kernel=kernel,
                                                             norm_mean=norm_mean, norm_stddev=norm_stddev, order=order)

                module_list = module_list + [self.single_step_net2 for step in range(num_steps-1)]
            else:
                module_list = module_list + [self.single_step_net for step in range(num_steps-1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp, viscosity):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i], viscosity)
            vortex_features.append(net_out)

        return vortex_features


class MultiStepMultiVortexNetworkBC(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], distinct_nets=True, order=1):

        super(MultiStepMultiVortexNetworkBC, self).__init__()

        self.single_step_net = MultiParticleVortexNetworkBC(depth=depth, hidden_units=hidden_units,
                                                            batch_norm=batch_norm, kernel=kernel,
                                                            norm_mean=norm_mean, norm_stddev=norm_stddev, order=order)

        module_list = [self.single_step_net]
        if num_steps > 1:
            if distinct_nets:
                self.single_step_net2 = MultiParticleVortexNetworkBC(depth=depth, hidden_units=hidden_units,
                                                                     batch_norm=batch_norm, kernel=kernel,
                                                                     norm_mean=norm_mean, norm_stddev=norm_stddev, order=order)

                module_list = module_list + [self.single_step_net2 for step in range(num_steps-1)]
            else:
                module_list = module_list + [self.single_step_net for step in range(num_steps-1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i])
            vortex_features.append(net_out)

        return vortex_features



class MultiStepVortexNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, use_2_nets=True, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], distinct_nets=True):

        super(MultiStepVortexNetwork, self).__init__()

        self.single_step_net = SingleVortexViscousNetwork(depth=depth, hidden_units=hidden_units,
                                             batch_norm=batch_norm, kernel=kernel,
                                             norm_mean=norm_mean, norm_stddev=norm_stddev)

        module_list = [self.single_step_net]
        if num_steps > 1:
            if distinct_nets:
                self.single_step_net2 = VortexNetwork(depth=depth, hidden_units=hidden_units,
                                             batch_norm=batch_norm, kernel=kernel,
                                             norm_mean=norm_mean, norm_stddev=norm_stddev)

                module_list = module_list + [self.single_step_net2 for step in range(num_steps-1)]
            else:
                module_list = module_list + [self.single_step_net for step in range(num_steps-1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp, nyu):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i], nyu)
            vortex_features.append(net_out)

        return vortex_features


class MultiStepInteractionNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, use_2_nets=True, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], distinct_nets=True):

        super(MultiStepInteractionNetwork, self).__init__()

        self.single_step_net = InteractionNetwork(depth=depth, hidden_units=hidden_units,
                                                  batch_norm=batch_norm, kernel=kernel,
                                                  norm_mean=norm_mean, norm_stddev=norm_stddev)

        module_list = [self.single_step_net]
        if num_steps > 1:
            if distinct_nets:
                self.single_step_net2 = InteractionNetwork(depth=depth, hidden_units=hidden_units,
                                                           batch_norm=batch_norm, kernel=kernel,
                                                           norm_mean=norm_mean, norm_stddev=norm_stddev)

                module_list = module_list + [self.single_step_net2 for step in range(num_steps-1)]
            else:
                module_list = module_list + [self.single_step_net for step in range(num_steps-1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i])
            vortex_features.append(net_out)

        return vortex_features


class MultiStepViscousInteractionNetwork(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, use_2_nets=True, batch_norm=True, kernel='offset-gaussian',
                 num_steps=1, norm_mean=[0., 0., 0.], norm_stddev=[1.0, 1.0, 1.0], distinct_nets=True):

        super(MultiStepViscousInteractionNetwork, self).__init__()

        self.single_step_net = ViscousInteractionNetwork(depth=depth, hidden_units=hidden_units,
                                                         batch_norm=batch_norm, kernel=kernel,
                                                         norm_mean=norm_mean, norm_stddev=norm_stddev)

        module_list = [self.single_step_net]
        if num_steps > 1:
            if distinct_nets:
                self.single_step_net2 = ViscousInteractionNetwork(depth=depth, hidden_units=hidden_units,
                                                                  batch_norm=batch_norm, kernel=kernel,
                                                                  norm_mean=norm_mean, norm_stddev=norm_stddev)

                module_list = module_list + [self.single_step_net2 for step in range(num_steps-1)]
            else:
                module_list = module_list + [self.single_step_net for step in range(num_steps-1)]

        self. multi_step_net = torch.nn.ModuleList(module_list)

    def forward(self, inp, viscosity):

        vortex_features = [inp]

        for i, net in enumerate(self.multi_step_net):
            net_out = net(vortex_features[i], viscosity)
            vortex_features.append(net_out)

        return vortex_features



class MultiStepLoss(torch.nn.Module):

    def __init__(self, kernel='gaussian', resolution=(128, 128), num_steps=1, batch_size=1, dt=1.0):

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
        elif kernel == 'ExpGaussian':
            self.n_features = 6
            self.falloff_kernel = GaussExpFalloffKernel(dt=dt)
        elif kernel == 'ExpGaussianRed':
            self.n_features = 5
            self.falloff_kernel = GaussExpFalloffKernelReduced()
        elif kernel == 'GaussianVorticity':
            self.n_features = 4
            self.falloff_kernel = GaussianFalloffKernelVelocity()

        self.multi_step_falloff_kernel = torch.nn.ModuleList([self.falloff_kernel for step in range(num_steps + 1)])


    def forward(self, vortex_features, target_velocities):

        losses = []
        l1_losses = []
        max_losses = []

        for i, kernel in enumerate(self.multi_step_falloff_kernel):

            vel_y = kernel(vortex_features[i], self.points_y)
            vel_x = kernel(vortex_features[i], self.points_x)

            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)

            pred_vel = torch.stack([torch.cat([vel_yy, self.cat_y], dim=-1), torch.cat([vel_xx, self.cat_x], dim=-2)], dim=-1)

            loss = F.mse_loss(pred_vel, target_velocities[i], reduction='sum') / self.batch_size
            l1_loss = F.l1_loss(pred_vel, target_velocities[i], reduction='sum') / self.batch_size
            loss_max = torch.max((pred_vel - target_velocities[i])**2)

            losses.append(loss)
            l1_losses.append(l1_loss)
            max_losses.append(loss_max)

        return losses, max_losses


