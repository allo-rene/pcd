import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "shufflenet_0_5x",
    "shufflenet_1_0x",
    "shufflenet_1_5x",
    "shufflenet_2_0x",
    "ShuffleNetV2",
]


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                mid_channels,
                mid_channels,
                ksize,
                stride,
                pad,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    # def channel_shuffle(self, x):
    #     batchsize, num_channels, height, width = x.data.size()
    #     assert num_channels % 4 == 0
    #     x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    #     x = x.permute(1, 0, 2)
    #     x = x.reshape(2, -1, num_channels // 2, height, width)
    #     return x[0], x[1]

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.view(batchsize, 2, num_channels // 2, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x[:, :num_channels//2, :, :], x[:, num_channels//2:, :, :]


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, width=1.0, dropout=0.2, flatten=True):
        super(ShuffleNetV2, self).__init__()
        self.flatten = flatten
        self.dropout = dropout

        self.stage_repeats = [4, 8, 4]
        self.width = width
        if width == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=2,
                        )
                    )
                else:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel // 2,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=1,
                        )
                    )

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )

        if self.flatten:
            self.globalpool = nn.AvgPool2d(7)

            self.classifier = nn.Sequential(
                nn.Linear(self.stage_out_channels[-1], n_class, bias=False)
            )
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        if self.flatten:
            x = self.globalpool(x)
            x = x.contiguous().view(-1, self.stage_out_channels[-1])
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def shufflenet_0_5x(**kwargs):
    return ShuffleNetV2(width=0.5, **kwargs)


def shufflenet_1_0x(**kwargs):
    return ShuffleNetV2(width=1.0, **kwargs)


def shufflenet_1_5x(**kwargs):
    return ShuffleNetV2(width=1.5, **kwargs)


def shufflenet_2_0x(**kwargs):
    return ShuffleNetV2(width=2.0, **kwargs)
