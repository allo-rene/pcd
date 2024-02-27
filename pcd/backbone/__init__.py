import copy
import torch

import resnet
import mobilenetv3


def build_student_backbone(arch, flatten=False):
    """
    :param arch: the architecture of student model
    :param mlp: the following mlp
    :param flatten: if using flatten operation to get vectorized feature
    """
    if arch == "resnet18":
        backbone = resnet.resnet18(flatten=flatten)
    elif arch == "resnet34":
        backbone = resnet.resnet34(flatten=flatten)
    elif arch == "mobilenetv3_large":
        backbone = mobilenetv3.mobilenetv3_large(flatten=flatten)
    else:
        raise ValueError(f"Please add other student model as you like.")

    with torch.no_grad():
        tmp_backbone = copy.deepcopy(backbone)
        feat_dim = tmp_backbone(torch.randn(1, 3, 224, 224)).shape[1]
        del tmp_backbone
    return backbone, feat_dim
