import math
import shutil

import torch
import torch.distributed as dist


@torch.no_grad()
def reduce_tensor_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt.item()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, scheduler, **kwargs):
    """Decay the learning rate based on the schedule"""
    if bool(kwargs):
        base_lr = kwargs.get("lr", 0.1)
        total_epochs = kwargs.get("total_epochs", 100)
    else:
        base_lr = 0.1
        total_epochs = 100

    if scheduler == "warmcos":
        if bool(kwargs):
            warmup_epochs = kwargs.get("warmup_epochs", None)
        else:
            warmup_epochs = None

        if warmup_epochs is None:
            raise ValueError(f"The warmup_epochs should be given.")

        if epoch < warmup_epochs:
            lr = base_lr * epoch / warmup_epochs
        else:
            lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    elif scheduler == "cos":
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch / total_epochs)))
    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cal_eta_time(iter_time_mtr, left_iters):
    left_time = iter_time_mtr.avg * left_iters
    t_m, t_s = divmod(int(left_time), 60)
    t_h, t_m = divmod(t_m, 60)
    t_d, t_h = divmod(t_h, 24)
    eta = f"{t_d}d.{t_h:}h.{t_m}m"
    return eta