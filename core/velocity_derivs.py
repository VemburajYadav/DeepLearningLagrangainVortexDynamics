import torch
import numpy as np
from core.custom_functions import *



class VelocityDerivatives(torch.nn.Module):

    def __init__(self, order=2):

        super(VelocityDerivatives, self).__init__()

        self.order = order
        self.falloff_kernel = GaussianFalloffKernelVelocity()


    def forward(self, inp, points):

        y, x, tau, sig = torch.unbind(inp, dim=-1)

        inp_clone = inp.detach().clone()
        nparticles = y.shape[1]

        location = torch.stack([y, x], dim=-1)

        feature_list = []
        batch_size = points.shape[0]

        for i in range(batch_size):
            paxes_tensor = torch.tensor([i], device='cuda:0')
            p_features = torch.index_select(inp, dim=0, index=paxes_tensor)
            b_points = torch.index_select(points, dim=0, index=paxes_tensor).view(-1, 2)
            b_points_y, b_points_x = torch.unbind(b_points, dim=-1)
            b_points_y.requires_grad_(True)
            b_points_x.requires_grad_(True)

            b_points_inp = torch.stack([b_points_y, b_points_x], dim=-1).view(1, -1, 1, 2)
            vel_at_points = self.falloff_kernel(p_features, b_points_inp).view(-1, 2)
            vel_y, vel_x = torch.unbind(vel_at_points, dim=-1)

            if self.order == 0:
                grad_features = torch.stack([vel_y.detach().clone(), vel_x.detach().clone()], dim=-1)
            elif self.order == 1:
                dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), b_points_y, retain_graph=True,allow_unused=True)[0]
                dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), b_points_x, retain_graph=True,allow_unused=True)[0]
                du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), b_points_y, retain_graph=True,allow_unused=True)[0]
                du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), b_points_x, retain_graph=False,allow_unused=True)[0]

                grad_features = torch.stack([vel_y.detach().clone(), vel_x.detach().clone(),
                                             dv_dy.detach().clone(), dv_dx.detach().clone(),
                                             du_dy.detach().clone(), du_dx.detach().clone()], dim=-1)
            elif self.order == 2:
                dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), b_points_y, create_graph=True, retain_graph=True,allow_unused=True)[0]
                dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), b_points_x, create_graph=True, retain_graph=True,allow_unused=True)[0]
                du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), b_points_y, create_graph=True, retain_graph=True,allow_unused=True)[0]
                du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), b_points_x, create_graph=True, retain_graph=True,allow_unused=True)[0]

                d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), b_points_x, retain_graph=True, allow_unused=True)[0]
                d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), b_points_y, retain_graph=True, allow_unused=True)[0]
                d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), b_points_y, retain_graph=True, allow_unused=True)[0]
                d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), b_points_y, retain_graph=True, allow_unused=True)[0]
                d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), b_points_x, retain_graph=True, allow_unused=True)[0]
                d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), b_points_x, allow_unused=True)[0]

                grad_features = torch.stack([vel_y.detach().clone(), vel_x.detach().clone(),
                                             dv_dy.detach().clone(), dv_dx.detach().clone(),
                                             du_dy.detach().clone(), du_dx.detach().clone(),
                                             d2v_dy2.detach().clone(), d2v_dx2.detach().clone(),
                                             d2v_dxdy.detach().clone(),
                                             d2u_dy2.detach().clone(), d2u_dx2.detach().clone(),
                                             d2u_dydx.detach().clone()], dim=-1)

            elif self.order == 3:
                dv_dy = torch.autograd.grad(torch.unbind(vel_y, dim=-1), b_points_y, create_graph=True, retain_graph=True,allow_unused=True)[0]
                dv_dx = torch.autograd.grad(torch.unbind(vel_y, dim=-1), b_points_x, create_graph=True, retain_graph=True,allow_unused=True)[0]
                du_dy = torch.autograd.grad(torch.unbind(vel_x, dim=-1), b_points_y, create_graph=True, retain_graph=True,allow_unused=True)[0]
                du_dx = torch.autograd.grad(torch.unbind(vel_x, dim=-1), b_points_x, create_graph=True, retain_graph=True,allow_unused=True)[0]

                d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), b_points_x, create_graph=True,
                                              retain_graph=True, allow_unused=True)[0]
                d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), b_points_y, create_graph=True,
                                              retain_graph=True, allow_unused=True)[0]
                d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), b_points_y, create_graph=True,
                                               retain_graph=True, allow_unused=True)[0]
                d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), b_points_y, create_graph=True,
                                              retain_graph=True, allow_unused=True)[0]
                d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), b_points_x, create_graph=True,
                                              retain_graph=True, allow_unused=True)[0]
                d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), b_points_x, create_graph=True,
                                               allow_unused=True)[0]

                d3u_dx3 = torch.autograd.grad(torch.unbind(d2u_dx2, dim=-1), b_points_x,
                                              retain_graph=True, allow_unused=True)[0]
                d3u_dy3 = torch.autograd.grad(torch.unbind(d2u_dy2, dim=-1), b_points_y,
                                              retain_graph=True, allow_unused=True)[0]
                d3u_dx2dy = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), b_points_x,
                                                retain_graph=True, allow_unused=True)[0]
                d3u_dxdy2 = torch.autograd.grad(torch.unbind(d2u_dydx, dim=-1), b_points_y,
                                                retain_graph=True, allow_unused=True)[0]
                d3v_dx3 = torch.autograd.grad(torch.unbind(d2v_dx2, dim=-1), b_points_x,
                                              retain_graph=True, allow_unused=True)[0]
                d3v_dy3 = torch.autograd.grad(torch.unbind(d2v_dy2, dim=-1), b_points_y,
                                              retain_graph=True, allow_unused=True)[0]
                d3v_dx2dy = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), b_points_x,
                                                retain_graph=True, allow_unused=True)[0]
                d3v_dxdy2 = torch.autograd.grad(torch.unbind(d2v_dxdy, dim=-1), b_points_y, allow_unused=True)[0]

                grad_features = torch.stack([vel_y.detach().clone(), vel_x.detach().clone(),
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


            feature_list.append(grad_features)

        deriv_features = torch.stack(feature_list, dim=0)

        return deriv_features


# def get_deriv_features(inp, points):
#     y, x, tau, sig, d = torch.unbind(inp, dim=-1)
#
#     inp_clone = inp.detach().clone()
#
#     nparticles = y.shape[1]
#
#     location = torch.stack([y, x], dim=-1)
#     location_clone = location.detach().clone()
#
#     feature_list = []
#     paxes = np.arange(nparticles)
#     for i in range(nparticles):
#         paxes_tensor = torch.tensor([i], device='cuda:0')
#         p_loc = torch.index_select(location, dim=-2, index=paxes_tensor).view(-1, 2)
#         py, px = torch.unbind(p_loc, dim=-1)
#         py.requires_grad_(True)
#         px.requires_grad_(True)
#         p_loc_inp = torch.stack([py, px], dim=-1).view(-1, 1, 1, 2)
#         other_p_axes = np.delete(paxes, i)
#         other_paxes_tensor = torch.tensor(other_p_axes, device='cuda:0')
#         other_p_features = torch.index_select(inp, dim=-2, index=other_paxes_tensor)
#         vel_by_other_ps = falloff_kernel(other_p_features, p_loc_inp).view(-1, 2)
#         vel_y, vel_x = torch.unbind(vel_by_other_ps, dim=-1)
#
#         dv_dy = \
#         torch.autograd.grad(torch.unbind(vel_y, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
#         dv_dx = \
#         torch.autograd.grad(torch.unbind(vel_y, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
#         du_dy = \
#         torch.autograd.grad(torch.unbind(vel_x, dim=-1), py, create_graph=True, retain_graph=True, allow_unused=True)[0]
#         du_dx = \
#         torch.autograd.grad(torch.unbind(vel_x, dim=-1), px, create_graph=True, retain_graph=True, allow_unused=True)[0]
#
#         d2u_dx2 = torch.autograd.grad(torch.unbind(du_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
#         d2u_dy2 = torch.autograd.grad(torch.unbind(du_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
#         d2u_dydx = torch.autograd.grad(torch.unbind(du_dx, dim=-1), py, retain_graph=True, allow_unused=True)[0]
#         d2v_dy2 = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), py, retain_graph=True, allow_unused=True)[0]
#         d2v_dx2 = torch.autograd.grad(torch.unbind(dv_dx, dim=-1), px, retain_graph=True, allow_unused=True)[0]
#         d2v_dxdy = torch.autograd.grad(torch.unbind(dv_dy, dim=-1), px, allow_unused=True)[0]
#
#         feature_list.append(torch.stack([vel_y.detach().clone(), vel_x.detach().clone(),
#                             dv_dy.detach().clone() * 10.0, dv_dx.detach().clone() * 10.0,
#                             du_dy.detach().clone() * 10.0, du_dx.detach().clone() * 10.0,
#                             d2v_dy2.detach().clone() * 100.0, d2v_dx2.detach().clone() * 100.0,
#                             d2v_dxdy.detach().clone() * 100.0,
#                             d2u_dy2.detach().clone() * 100.0, d2u_dx2.detach().clone() * 100.0,
#                             d2u_dydx.detach().clone() * 100.0], dim=-1))
#
#     deriv_features = torch.stack(feature_list, dim=-2)
#
#     return deriv_features