from .bt_resnet50 import load_barlow_twins_resnet50, build_student_mlp as bt_student_mlp
from .byol_resnet50 import load_byol_resnet50, build_stduent_mlp as byol_student_mlp
from .mocov3_resnet50 import load_mocov3_resnet50, build_student_mlp as mocov3_r50_student_mlp
from .mocov3_vit import load_mocov3_vit, build_student_mlp as mocov3_vit_sutdent_mlp
from .swav_resnet50 import load_swav_resnet50, build_student_mlp as swav_student_mlp


def build_teacher_model(arch, ckpt_path, student_feat_dim, flatten):
    if arch == "barlow_twins_r50":
        teacher, output_dim = load_barlow_twins_resnet50(ckpt_path, flatten)
        student_mlp = bt_student_mlp(student_feat_dim)
    elif arch == "byol_r50":
        teacher, output_dim = load_byol_resnet50(ckpt_path, flatten)
        student_mlp = byol_student_mlp(student_feat_dim)
    elif arch == "mocov3_r50":
        teacher, output_dim = load_mocov3_resnet50(ckpt_path, flatten)
        student_mlp = mocov3_r50_student_mlp(student_feat_dim)
    elif arch == "mocov3_vit":
        teacher, output_dim = load_mocov3_vit(ckpt_path, flatten)
        student_mlp = mocov3_vit_sutdent_mlp(student_feat_dim)
    elif arch == "swav_r50":
        teacher, output_dim = load_swav_resnet50(ckpt_path, flatten)
        student_mlp = swav_student_mlp(student_feat_dim)
    else:
        raise ValueError(f"Unsupported teacher architecture: {arch}.")
    return teacher, student_mlp, output_dim