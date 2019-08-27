import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        bilinear_dim = (channel // reduction)**2
        self.fc2 = nn.Sequential(
            nn.Linear(bilinear_dim, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y) 
        print(y.shape)
        y = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))  # bilinear operation
        print(y.shape)
        y = self.fc2(y.view(b, -1)).view(b, c, 1, 1)
        print(y.shape)
        return x * y.expand_as(x)
