import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)
    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            epsilon = group["epsilon"]
            lars = group["lars"]  # whether to use lars to adjust lr

            for p in group["params"]:
                if p.grad is None:
                    continue

                if lars:  # use lars to adjust the lr
                    w_norm = torch.norm(p.data)
                    g_norm = torch.norm(p.grad.data)
                    if w_norm * g_norm > 0:
                        local_lr = eta * w_norm / (g_norm +
                            weight_decay * w_norm + epsilon)
                    else:
                        local_lr = 1.
                else:
                    local_lr = 1.

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1-dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-local_lr*group['lr'])

        return loss


def split_params(model):
    # pick out the batch norm and bias
    params = []
    bn_and_bias = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            if m.weight is not None:  # affine is True
                bn_and_bias.append(m.weight)
            if m.bias is not None:
                bn_and_bias.append(m.bias)
        elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            params.append(m.weight)
            if m.bias is not None:
                bn_and_bias.append(m.bias)
    return params, bn_and_bias


def build_optimizer(student_model, optimizer, **kwargs):
    if bool(kwargs):
        lr = kwargs.get("lr", None)
        momentum = kwargs.get("momentum", None)
        weight_decay = kwargs.get("weight_decay", None)
        nesterov = kwargs.get("nesterov", None)
        exclude_weight_decay = kwargs.get("exclude_weight_decay", None)
    else:
        lr = None
        momentum = None
        weight_decay = None
        nesterov = None
        exclude_weight_decay = None

    if optimizer == "SGD":
        optimizer = optim.SGD(student_model.parameters(), lr, momentum, weight_decay, nesterov)
    elif optimizer == "LARS":
        params, bn_and_bias = split_params(student_model)
        optimizer = LARS([{"params": params, "lars": True, "weight_decay": weight_decay},
                          {"params": bn_and_bias, "lars": False, "weight_decay": exclude_weight_decay}],
                         lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer}.")
    return optimizer
