import torch
import torch.nn as nn


class SpatialReLU(nn.Module):
    def forward(self, x):
        m = torch.sum(x, dim=[2, 3])
        mask = (m > 0.).float().detach().unsqueeze(-1).unsqueeze(-1)
        return x * mask
