import torch.nn as nn
import torch

class AdultNetwork(nn.Module):
    def __init__(self, in_features, layer_sizes):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(in_features, layer_sizes[0]),
                    nn.Tanh(),
                    nn.Linear(layer_sizes[0], layer_sizes[1]),
                    nn.Tanh(),
                    nn.Linear(layer_sizes[2], layer_sizes[3]),
                    nn.Tanh(),
                    nn.Linear(layer_sizes[3], layer_sizes[4])
                )
        self.net.apply(self.__init_weights)

    def __init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight.data)
            torch.nn.init.normal_(layer.bias.data)

    def forward(self, x):
        return self.net(x)
