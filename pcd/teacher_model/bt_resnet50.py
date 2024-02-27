from collections import OrderedDict

import torch
import torch.nn as nn

from ..backbone.resnet import ResNet, Bottleneck
from .spatial_relu import SpatialReLU


def build_student_mlp(feat_dim):
    return nn.Sequential(
        nn.Conv2d(feat_dim, 8192, 1, 1, 0, bias=False),
        nn.BatchNorm2d(8192),
        nn.ReLU(inplace=True),
        nn.Conv2d(8192, 8192, 1, 1, 0, bias=False),
        nn.BatchNorm2d(8192),
        nn.ReLU(inplace=True),
        nn.Conv2d(8192, 8192, 1, 1, 0, bias=False),

        nn.Conv2d(8192, 8192, 1, 1, 0, bias=False),
        nn.BatchNorm2d(8192),
        nn.ReLU(inplace=True),
        nn.Conv2d(8192, 8192, 1, 1, 0, bias=False),
        nn.BatchNorm2d(8192),
        nn.ReLU(inplace=True),
        nn.Conv2d(8192, 8192, 1, 1, 0, bias=False),
    )


def load_barlow_twins_resnet50(ckpt_path, flatten=False):
    teacher_backbone = ResNet(Bottleneck, [3, 4, 6, 3], flatten=flatten)
    mlp = nn.Sequential(
        nn.Linear(2048, 8192, bias=False),
        nn.BatchNorm1d(8192),
        nn.ReLU(),
        nn.Linear(8192, 8192, bias=False),
        nn.BatchNorm1d(8192),
        nn.ReLU(),
        nn.Linear(8192, 8192, bias=False),
    )

    pretrain_ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_state_dict = OrderedDict()
    mlp_state_dict = OrderedDict()
    for k, v in pretrain_ckpt["model"].items():
        if k.startswith("module.backbone."):
            backbone_state_dict[k.replace("module.backbone.", "")] = v
        elif k.startswith("module.projector."):
            mlp_state_dict[k.replace("module.projector.", "")] = v
        else:
            continue

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
            nn.Conv2d(2048, 8192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8192),
            SpatialReLU(),
            nn.Conv2d(8192, 8192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8192),
            SpatialReLU(),
            nn.Conv2d(8192, 8192, kernel_size=1, stride=1, padding=0, bias=False),
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
            spatial_mlp
        )
    return teacher_model, 8192
