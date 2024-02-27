import torch.nn as nn
import torch.nn.functional as F

import math

__all__ = [
    "mobilenetv3_large_1_5x",
    "mobilenetv3_large_1_25x",
    "mobilenetv3_large_1_0x",
    "mobilenetv3_large_0_75x",
    "mobilenetv3_large_0_5x",
    "mobilenetv3_large_0_25x",
    "mobilenetv3_small_1_5x",
    "mobilenetv3_small_1_25x",
    "mobilenetv3_small_1_0x",
    "mobilenetv3_small_0_75x",
    "mobilenetv3_small_0_5x",
    "mobilenetv3_small_0_25x",
    "mobilenetv3_large",
    "mobilenetv3_small",
]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=h_sigmoid,
        divisor=8,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SqueezeExcite(hidden_dim, gate_fn=h_sigmoid()) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:  # expand
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SqueezeExcite(hidden_dim, gate_fn=h_sigmoid()) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width=1.0, dropout=0.2, flatten=True):
        super(MobileNetV3, self).__init__()
        self.flatten = flatten
        # setting of inverted residual blocks
        self.dropout = dropout
        self.cfgs = cfgs
        assert mode in ["large", "small"]

        # building first layer
        input_channel = _make_divisible(16 * width, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)

        if self.flatten:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            output_channel = {"large": 1280, "small": 1024}
            output_channel = (
                _make_divisible(output_channel[mode] * width, 8)
                if width > 1.0
                else output_channel[mode]
            )
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                h_swish(),
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes),
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        if self.flatten:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large models
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, mode="large", **kwargs)


def mobilenetv3_large_1_5x(**kwargs):
    return mobilenetv3_large(width=1.5, **kwargs)

def mobilenetv3_large_1_25x(**kwargs):
    return mobilenetv3_large(width=1.25, **kwargs)

def mobilenetv3_large_1_0x(**kwargs):
    return mobilenetv3_large(width=1.0, **kwargs)

def mobilenetv3_large_0_75x(**kwargs):
    return mobilenetv3_large(width=0.75, **kwargs)

def mobilenetv3_large_0_5x(**kwargs):
    return mobilenetv3_large(width=0.5, **kwargs)

def mobilenetv3_large_0_25x(**kwargs):
    return mobilenetv3_large(width=0.25, **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small models
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode="small", **kwargs)


def mobilenetv3_small_1_5x(**kwargs):
    return mobilenetv3_small(width=1.5, **kwargs)

def mobilenetv3_small_1_25x(**kwargs):
    return mobilenetv3_small(width=1.25, **kwargs)

def mobilenetv3_small_1_0x(**kwargs):
    return mobilenetv3_small(width=1.0, **kwargs)

def mobilenetv3_small_0_75x(**kwargs):
    return mobilenetv3_small(width=0.75, **kwargs)

def mobilenetv3_small_0_5x(**kwargs):
    return mobilenetv3_small(width=0.5, **kwargs)

def mobilenetv3_small_0_25x(**kwargs):
    return mobilenetv3_small(width=0.25, **kwargs)
