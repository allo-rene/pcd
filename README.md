# Pixel-Wise Contrastive Distillation
This is a PyTorch implementation of the [PCD paper](https://arxiv.org/abs/2211.00218).

<img width="888" alt="image" src="https://github.com/allo-rene/pcd/assets/52401315/f1a00dd3-5d3a-42f7-ad4c-e0273b3180f6">

```
@inproceedings{huang2023pixel,
  title={Pixel-Wise Contrastive Distillation},
  author={Huang, Junqiang and Guo, Zichao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16359--16369},
  year={2023}
}
```

## Preparation

## Distillation
For single node distributed training:
```
python main_pcd.py {dataset_dir} --rank 0 --world-size 1 -md \
  --student-arch resnet18 --teacher-arch mocov3_r50 \
  --teacher-ckpt {checkpoint_path_of_teacher_model} --output-dir {output_dir}
```

For multi nodes distributed training:
```
# For main node
python main_pcd.py {dataset_dir} --rank 0 --world-size {number_of_nodes} -md \
  --student-arch resnet18 --teacher-arch mocov3_r50 \
  --teacher-ckpt {checkpoint_path_of_teacher_model} --output-dir {output_dir}

# For other nodes
python main_pcd.py {dataset_dir} --rank {index_of_current_node} --world-size {number_of_nodes} -md \
  --dist-url {ip_of_main_node} --student-arch resnet18 --teacher-arch mocov3_r50 \
  --teacher-ckpt {checkpoint_path_of_teacher_model} --output-dir {output_dir}
```

## Models
