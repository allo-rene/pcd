#!/usr/bin/env python

import os
import time
import random
import argparse
import builtins
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

import pcd
from pcd import reduce_tensor_mean, AvgMeter, save_checkpoint, \
    adjust_learning_rate, cal_eta_time, setup_logger, synchronize


parser = argparse.ArgumentParser(description="PCD Training on ImageNet")
# data
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("-j", "--workers", default=8, type=int, help="number of data loading workers")
parser.add_argument("-b", "--batch-size", default=1024, type=int, metavar="N", help="mini-batch size (default: 256)")

# model
parser.add_argument("-sa", "--student-arch", default=None, type=str, help="The architecture of student model.")
parser.add_argument("-ta", "--teacher-arch", default=None, type=str, help="The architecture of teacher model.")
parser.add_argument("--teacher-ckpt", default=None, type=str, help="The path storing teacher checkpoint.")
parser.add_argument("-qs", "--queue-size", default=65536, type=int, help="The size the queue storing negative features.")
parser.add_argument("-tau", "--temperature", default=0.2, type=float)

# optimizer
parser.add_argument("--optimizer", type=str, default="LARS", help="choice of optimizer")
parser.add_argument("--start-epoch", default=0, type=int, help="current epoch")
parser.add_argument("--total-epochs", default=100, type=int, help="number of total epochs to run")
parser.add_argument("-lr", "--learning-rate", default=1.0, type=float, help="initial learning rate")
parser.add_argument("-wd", "--weight-decay", default=0.00001, type=float, help="weight decay")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum of optimization")
parser.add_argument("--nesterov", action="store_true", default=False, help="whether to use nesterov")
parser.add_argument("--lr-decay-strategy", default="warmcos", type=str, help="strategy for learning rate decaying.")
parser.add_argument("--warmup-epochs", default=10, type=int, help="numter of warmup epochs")
parser.add_argument("-p", "--print-freq", default=10, type=int, help="print frequency (default: 10)")

# others
parser.add_argument("--resume", default=None, type=str, metavar="PATH", help="path to latest checkpoint (default: None)")
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
parser.add_argument("--output-dir", default=".", type=str, help="output directory")

# distributed
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://localhost:23456", type=str, help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("-md", "--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                    "N processes per node, which has N GPUs. This is the "
                    "fastest way to use PyTorch for either single node or "
                    "multi node data parallel training")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logger(args.output_dir, args.gpu, filename="train_log.txt")
    else:
        logger = None
    synchronize()

    # data
    traindir = os.path.join(args.data, "train")
    train_dataset = datasets.ImageFolder(traindir, pcd.AsymmetricTransform(*pcd.byol_transform()))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    if args.rank == 0:
        logger.info(f"Data is ready.")

    # model
    student, student_feat_dim = pcd.build_student_backbone(args.student_arch, flatten=False)
    teacher, student_mlp, output_dim = pcd.build_teacher_model(args.teacher_arch, args.teacher_ckpt, student_feat_dim, flatten=False)
    student = nn.Sequential(
        student,
        student_mlp
    )
    model = pcd.PCD(student, teacher, output_dim, args.queue_size, args.temperature)

    if args.rank == 0:
        logger.info(model)
        logger.info(f"Model is ready.")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optimizer
    args.base_lr = args.learning_rate * (args.batch_size // 256)  # learning rate scaling rule
    optimizer_kwargs = {
        "lr": args.base_lr if "warm" not in args.lr_decay_strategy else 1e-6,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
    }
    student_model = model.student if not args.multiprocessing_distributed else model.module.student
    optimizer = pcd.build_optimizer(student_model, args.optimizer, **optimizer_kwargs)
    if args.rank == 0:
        logger.info(f"Optimizer is ready.")

    # resume
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.rank == 0:
                logger.info(f"=> loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{args.gpu}"
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if args.rank == 0:
                logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            if args.rank == 0:
                logger.info(f"=> no checkpoint found at '{args.resume}'")

    # training
    args.mtr_dt = None  # dict for storing AvgMeter
    if args.rank == 0:
        logger.info(f"Start training.")
    for epoch in range(args.start_epoch, args.total_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(args, train_loader, model, optimizer, epoch, logger)

        # save
        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="checkpoint_{:04d}.pth.tar".format(epoch),
            )
    if args.rank == 0:
        logger.info(f"Training is done.")


def train(args, train_loader, model, optimizer, epoch, logger):
    lr_decay_kwargs = {
        "lr": args.base_lr,
        "total_epochs": args.total_epochs,
        "warmup_epochs": args.warmup_epochs,
    }

    if args.mtr_dt is None:
        args.mtr_dt = {
            "batch_time_mtr": AvgMeter(),
            "data_time_mtr": AvgMeter(),
            "loss_mtr": AvgMeter(),
            "top1_mtr": AvgMeter(),
            "top5_mtr": AvgMeter(),
        }

    # switch to train mode
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        args.mtr_dt["data_time_mtr"].update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output_dict = model(*images)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output_dict["loss"].backward()
        optimizer.step()

        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, scheduler="warmcos", **lr_decay_kwargs)

        # reduce if necessary
        if args.multiprocessing_distributed:
            for k in output_dict.keys():
                if "loss" in k or "top" in k:
                    output_dict[k] = reduce_tensor_mean(output_dict[k])
                    args.mtr_dt[f"{k}_mtr"].update(output_dict[k], images[0].shape[0])

        # measure elapsed time
        args.mtr_dt["batch_time_mtr"].update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            eta_time = cal_eta_time(args.mtr_dt["batch_time_mtr"],
                                    (args.total_epochs - epoch - 1) * len(train_loader) + len(train_loader) - i - 1)
            print_line = f"epoch: [{epoch + 1}/{args.total_epochs}], iter: [{i+1}/{len(train_loader)}], " \
                         f"eta: {eta_time}, loss: {args.mtr_dt['loss_mtr'].avg:.2f} ({args.mtr_dt['loss_mtr'].val:.2f})"
            if "top1" in args.mtr_dt:
                print_line += f", top1: {args.mtr_dt['top1_mtr'].avg:.2f} ({args.mtr_dt['top1_mtr'].val:.2f})" \
                              f", top5: {args.mtr_dt['top5_mtr'].avg:.2f} ({args.mtr_dt['top5_mtr'].val:.2f})"
            logger.info(print_line)


if __name__ == "__main__":
    main()
