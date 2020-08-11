import torch


class SimpleNN(torch.nn.Module):

    def __init__(self, depth=1, hidden_units=100, in_features=100, out_features=1):

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

            self.layers.append(torch.nn.Sequential(torch.nn.Linear(in_features=in_feat,
                                                                   out_features=self.hidden_units),
                                                   torch.nn.LeakyReLU(negative_slope=0.1),
                                                   torch.nn.BatchNorm1d(num_features=self.hidden_units)))

        self.hidden_layers = torch.nn.ModuleList(self.layers)
        self.output_layer = torch.nn.Linear(in_features=self.hidden_units,
                                            out_features=self.out_features)

    def forward(self, x):

        for layer in self.hidden_layers:
            x = layer(x)

        output = self.output_layer(x)

        return output