import torch
from core.custom_functions import particle_vorticity_to_velocity
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

            vel_y = particle_vorticity_to_velocity(loc, tau.view(-1, 1),
                                                   sig.view(-1, 1, 1), points_y)

            vel_yy, vel_yx = torch.unbind(vel_y, dim=-1)
            vel_yy = torch.unsqueeze(vel_yy, dim=-1)

            vel_x = particle_vorticity_to_velocity(loc, tau.view(-1, 1),
                                                   sig.view(-1, 1, 1), points_x)
            vel_xy, vel_xx = torch.unbind(vel_x, dim=-1)
            vel_xx = torch.unsqueeze(vel_xx, dim=-1)

            pred_vel_y = torch.cat([vel_yy, cat_y], dim=2)
            pred_vel_x = torch.cat([vel_xx, cat_x], dim=1)

            pred_vel = torch.cat([pred_vel_y, pred_vel_x], dim=-1)

            loss_step = F.mse_loss(pred_vel, target_velocities[i + 1], reduction='sum') / self.batch_size
            losses.append(loss_step)

        loss = torch.stack(losses, dim=0)

        return loss
