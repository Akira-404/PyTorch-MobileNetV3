import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from typing import Callable, List, Optional


# 调整通道为最靠近8的整数倍，利于计算机计算


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor

    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[...,
                                               nn.Module]] = None,
                 activation_layer: Optional[Callable[...,
                                                     nn.Module]] = None):
        padding = (kernel_size - 1) / 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super(
            ConvBNActivation,
            self).__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False),
            norm_layer(out_planes),
            activation_layer(
                inplace=True))


# SE模块:squeeze+excitaion


class SqueezeExcitaion(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitaion, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc1 = nn.Conv2d(squeeze_c, input_c, 1)

    def forwar(self, x: Tensor) -> Tensor:
        # squeeze:通过全局池化压缩为同通道，长宽为1x1的数据
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))

        # excitaion
        # 压缩为1/4的通道
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)

        # 扩展为4倍的通道
        scale = self.fc1(scale)
        scale = F.hardsigmoid(scale, inplace=True)

        # 重新设置权重
        return scale * x


'''
倒残差结构配置param：
input_c:        输入数据的通道
kernel:         kernel size
expanded_c:     第一个卷积层所用的卷积核的个数
out_c:          最后输出的通道数
use_se:         是否使用se模块
activation:     激活函数
stride:         DW步距
width_multi:    倍率因子
'''


class InvertedResidualConfig:
    def __init__(
            self,
            input_c: int,
            kernel: int,
            expanded_c: int,
            out_c: int,
            use_se: bool,
            activation: str,
            stride: int,
            width_multi: float):
        self.input_c = self._adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self._adjust_channels(expanded_c, width_multi)
        self.out_c = self._adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


# 倒残差结构


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        # 检测步距是否正确
        if cnf.stride not in [1, 2]:
            raise ValueError('illegal stride value.')

        # 是否使用残差结构shortcut->bool
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []

        activation_layer = nn.Hardsigmoid if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(
                ConvBNActivation(
                    cnf.input_c,
                    cnf.expanded_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer))

        # DepthWise
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.expanded_c,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_c,
                norm_layer=norm_layer,
                activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitaion(cnf.expanded_c))
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_stride = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        ret = self.block(x)
        if self.use_res_connect:
            ret += x
        return ret


class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError(
                'the inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, List)) and all([isinstance(s, InvertedResidualConfig) for s in
                                                                        inverted_residual_setting]):

            raise ValueError(
                'the inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(
            ConvBNActivation(
                3,
                firstconv_output_c,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish))

        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(
            ConvBNActivation(
                lastconv_input_c,
                lastconv_output_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish))

        self.featrues = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(
                lastconv_output_c, last_channel), nn.Hardswish(
                inplace=True), nn.Dropout(
                p=0.2, inplace=True), nn.Linear(
                last_channel, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.featrues(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# https://download.pytorch.org/models/mobilenet_v3_larg-8738ca79.pth

def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels(), width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
