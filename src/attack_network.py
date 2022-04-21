import torch.nn as nn
import torch

'''
Based on attack model from this MIA repo
https://github.com/spring-epfl/mia/blob/master/examples/cifar10.py#L83
'''

class AttackNetwork(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(.2),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
                )

    def forward(self, x):
        return self.net(x)
