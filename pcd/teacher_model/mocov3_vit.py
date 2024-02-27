import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.vit import vit_base_patch16
from .spatial_relu import SpatialReLU


class ModifiedMLP(nn.Module):
    def __init__(self, MLP):
        super(ModifiedMLP, self).__init__()
        self.mlp = MLP

    def forward(self, x):
        # shape of x = [N, 197, D]
        x = x[:, 1:, :]
        N, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        x = x.transpose(1, 2).reshape(N, C, H, W)
        x = self.mlp(x)  # N, C, H, W
        x = F.adaptive_avg_pool2d(x, [7, 7])
        return x


def build_student_mlp(feat_dim):
    return nn.Sequential(
        nn.Conv2d(feat_dim, 4096, 1, 1, 0, bias=False),
        nn.BatchNorm2d(4096),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 256, 1, 1, 0, bias=False),

        nn.Conv2d(256, 4096, 1, 1, 0, bias=False),
        nn.BatchNorm2d(4096),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 256, 1, 1, 0, bias=False),
    )


def load_mocov3_vit(ckpt_path, flatten=False):
    teacher_backbone = vit_base_patch16(flatten=flatten)

    mlp = nn.Sequential(
        nn.Linear(768, 4096, bias=False),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Linear(4096, 4096, bias=False),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Linear(4096, 256, bias=False)
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    backbone_state_dict = OrderedDict()
    mlp_state_dict = OrderedDict()

    for k, v in ckpt["state_dict"].items():
        if k.startswith("module.momentum_encoder."):
            if "head" not in k:
                new_k = k.replace("module.momentum_encoder.", "")
                backbone_state_dict[new_k] = v
            else:
                if "head.7." in k:
                    continue
                new_k = k.replace("module.momentum_encoder.head.", "")
                mlp_state_dict[new_k] = v

    msg = teacher_backbone.load_state_dict(backbone_state_dict, False)
    print(msg)

    msg = mlp.load_state_dict(mlp_state_dict, False)
    print(msg)

    if flatten:
        teacher_model = nn.Sequential(
            teacher_backbone,
            mlp
        )
    else:
        spatial_mlp = [
            nn.Conv2d(768, 4096, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4096),
            SpatialReLU(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4096),
            SpatialReLU(),
            nn.Conv2d(4096, 256, kernel_size=1, stride=1, padding=0, bias=False)
        ]
        spatial_mlp[0].weight.data = mlp[0].weight.data.unsqueeze(-1).unsqueeze(-1)
        spatial_mlp[1].weight.data = mlp[1].weight.data
        spatial_mlp[1].bias.data = mlp[1].bias.data
        spatial_mlp[1].running_mean = mlp[1].running_mean
        spatial_mlp[1].running_var = mlp[1].running_var
        spatial_mlp[1].num_batches_tracked = mlp[1].num_batches_tracked
        spatial_mlp[3].weight.data = mlp[3].weight.data.unsqueeze(-1).unsqueeze(-1)
        spatial_mlp[4].weight.data = mlp[4].weight.data
        spatial_mlp[4].bias.data = mlp[4].bias.data
        spatial_mlp[4].running_mean = mlp[4].running_mean
        spatial_mlp[4].running_var = mlp[4].running_var
        spatial_mlp[4].num_batches_tracked = mlp[4].num_batches_tracked
        spatial_mlp[-1].weight.data = mlp[-1].weight.data

        spatial_mlp = nn.Sequential(*spatial_mlp)
        teacher_model = nn.Sequential(
            teacher_backbone,
            ModifiedMLP(spatial_mlp)
        )
    return teacher_model, 256
