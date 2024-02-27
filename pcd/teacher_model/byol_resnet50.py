from collections import OrderedDict

import torch
import torch.nn as nn

from ..backbone.resnet import ResNet, Bottleneck
from .spatial_relu import SpatialReLU


def build_stduent_mlp(feat_dim):
    return nn.Sequential(
        nn.Conv2d(feat_dim, 4096, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(4096),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 256, kernel_size=1, stride=1, padding=0, bias=False),

        nn.Conv2d(256, 4096, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(4096),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 256, kernel_size=1, stride=1, padding=0, bias=False),
    )


def load_byol_resnet50(ckpt_path, flatten=False):
    teacher_backbone = ResNet(Bottleneck, [3, 4, 6, 3], flatten=flatten)
    mlp = [
        nn.Linear(2048, 4096, bias=True),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Linear(4096, 256, bias=False),
    ]

    mlp = nn.Sequential(*mlp)

    pretrain_ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_state_dict = OrderedDict()
    mlp_state_dict = OrderedDict()
    for k, v in pretrain_ckpt["model"].items():
        if "module.target_encoder" in k:
            if "fc" not in k:
                backbone_state_dict[k.replace("module.target_encoder.", "")] = v
            else:
                mlp_state_dict[k.replace("module.target_encoder.fc.", "")] = v
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
            nn.Conv2d(2048, 4096, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4096),
            SpatialReLU(),
            nn.Conv2d(4096, 256, kernel_size=1, stride=1, padding=0, bias=False),
        ]
        spatial_mlp[0].weight.data = mlp[0].weight.data.unsqueeze(-1).unsqueeze(-1)
        spatial_mlp[0].bias.data = mlp[0].bias.data
        spatial_mlp[1].weight.data = mlp[1].weight.data
        spatial_mlp[1].bias.data = mlp[1].bias.data
        spatial_mlp[1].running_mean = mlp[1].running_mean
        spatial_mlp[1].running_var = mlp[1].running_var
        spatial_mlp[1].num_batches_tracked = mlp[1].num_batches_tracked
        spatial_mlp[3].weight.data = mlp[3].weight.data.unsqueeze(-1).unsqueeze(-1)

        spatial_mlp = nn.Sequential(*spatial_mlp)
        teacher_model = nn.Sequential(
            teacher_backbone,
            spatial_mlp
        )
    return teacher_model, 256
