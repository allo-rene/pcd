from collections import OrderedDict

import torch
import torch.nn as nn

from ..backbone.resnet import ResNet, Bottleneck
from .spatial_relu import SpatialReLU


def build_student_mlp(feat_dim):
    return nn.Sequential(
        nn.Conv2d(feat_dim, 2048, 1, 1, 0),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
        nn.Conv2d(2048, 128, 1, 1, 0),

        nn.Conv2d(128, 2048, 1, 1, 0),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
        nn.Conv2d(2048, 128, 1, 1, 0),
    )


def load_swav_resnet50(ckpt_path, flatten=False):
    teacher_backbone = ResNet(Bottleneck, [3, 4, 6, 3], flatten=flatten)
    mlp = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 128),
    )

    mlp = nn.Sequential(*mlp)

    pretrain_ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_state_dict = OrderedDict()
    mlp_state_dict = OrderedDict()
    for k, v in pretrain_ckpt.items():
        if k.replace("module.", "") in teacher_backbone.state_dict().keys():
            backbone_state_dict[k.replace("module.", "")] = v
        elif "projection_head" in k:
            mlp_state_dict[k.replace("module.projection_head.", "")] = v
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
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2048),
            SpatialReLU(),
            nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True),
        ]
        spatial_mlp[0].weight.data = mlp[0].weight.data.unsqueeze(-1).unsqueeze(-1)
        spatial_mlp[0].bias.data = mlp[0].bias.data
        spatial_mlp[1].weight.data = mlp[1].weight.data
        spatial_mlp[1].bias.data = mlp[1].bias.data
        spatial_mlp[1].running_mean = mlp[1].running_mean
        spatial_mlp[1].running_var = mlp[1].running_var
        spatial_mlp[1].num_batches_tracked = mlp[1].num_batches_tracked
        spatial_mlp[3].weight.data = mlp[3].weight.data.unsqueeze(-1).unsqueeze(-1)
        spatial_mlp[3].bias.data = mlp[3].bias.data

        spatial_mlp = nn.Sequential(*spatial_mlp)
        teacher_model = nn.Sequential(
            teacher_backbone,
            spatial_mlp
        )
    return teacher_model, 128
