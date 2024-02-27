import bt_resnet50, byol_resnet50, mocov3_resnet50, mocov3_vit, swav_resnet50


def build_teacher_model(arch, ckpt_path, student_feat_dim, flatten):
    if arch == "barlow_twins_r50":
        teacher, output_dim = bt_resnet50.load_barlow_twins_resnet50(ckpt_path, flatten)
        student_mlp = bt_resnet50.build_student_mlp(student_feat_dim)
    elif arch == "byol_r50":
        teacher, output_dim = byol_resnet50.load_byol_resnet50(ckpt_path, flatten)
        student_mlp = byol_resnet50.build_stduent_mlp(student_feat_dim)
    elif arch == "mocov3_r50":
        teacher, output_dim = mocov3_resnet50.load_mocov3_resnet50(ckpt_path, flatten)
        student_mlp = mocov3_resnet50.build_student_mlp(student_feat_dim)
    elif arch == "mocov3_vit":
        teacher, output_dim = mocov3_vit.load_mocov3_vit(ckpt_path, flatten)
        student_mlp = mocov3_vit.build_student_mlp(student_feat_dim)
    elif arch == "swav_r50":
        teacher, output_dim = swav_resnet50.load_swav_resnet50(ckpt_path, flatten)
        student_mlp = swav_resnet50.build_student_mlp(student_feat_dim)
    else:
        raise ValueError(f"Unsupported teacher architecture: {arch}.")
    return teacher, student_mlp, output_dim
