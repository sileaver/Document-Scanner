import glob
import os
import warnings
import numpy as np
import cv2
import sys
import math
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import matplotlib.pyplot as plt
from transform import *

warnings.filterwarnings('ignore')


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = Activation("relu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class Activation(nn.Layer):
    """
    The wrapper of activations.

    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.

    Returns:
        A callable object of Activation.

    Raises:
        KeyError: When parameter `act` is not in the optional range.

    Examples:

        from paddleseg.models.common.activation import Activation

        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>

        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>

        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """

    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.layer.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("nn.layer.activation.{}()".format(
                    act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x


class Add(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return paddle.add(x, y, name)


class HRNet(nn.Layer):
    """
    The HRNet implementation based on PaddlePaddle.

    The original article refers to
    Jingdong Wang, et, al. "HRNet：Deep High-Resolution Representation Learning for Visual Recognition"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:
        pretrained (str, optional): The path of pretrained model.
        stage1_num_modules (int, optional): Number of modules for stage1. Default 1.
        stage1_num_blocks (list, optional): Number of blocks per module for stage1. Default (4).
        stage1_num_channels (list, optional): Number of channels per branch for stage1. Default (64).
        stage2_num_modules (int, optional): Number of modules for stage2. Default 1.
        stage2_num_blocks (list, optional): Number of blocks per module for stage2. Default (4, 4).
        stage2_num_channels (list, optional): Number of channels per branch for stage2. Default (18, 36).
        stage3_num_modules (int, optional): Number of modules for stage3. Default 4.
        stage3_num_blocks (list, optional): Number of blocks per module for stage3. Default (4, 4, 4).
        stage3_num_channels (list, optional): Number of channels per branch for stage3. Default [18, 36, 72).
        stage4_num_modules (int, optional): Number of modules for stage4. Default 3.
        stage4_num_blocks (list, optional): Number of blocks per module for stage4. Default (4, 4, 4, 4).
        stage4_num_channels (list, optional): Number of channels per branch for stage4. Default (18, 36, 72. 144).
        has_se (bool, optional): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(self,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4,),
                 stage1_num_channels=(64,),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=4,
                 stage3_num_blocks=(4, 4, 4),
                 stage3_num_channels=(18, 36, 72),
                 stage4_num_modules=3,
                 stage4_num_blocks=(4, 4, 4, 4),
                 stage4_num_channels=(18, 36, 72, 144),
                 has_se=False,
                 align_corners=False,
                 padding_same=True):
        super(HRNet, self).__init__()
        self.pretrained = pretrained
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.has_se = has_se
        self.align_corners = align_corners
        self.feat_channels = [sum(stage4_num_channels)]

        self.conv_layer1_1 = ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.conv_layer1_2 = ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=self.stage1_num_blocks[0],
            num_filters=self.stage1_num_channels[0],
            has_se=has_se,
            name="layer2",
            padding_same=padding_same)

        self.tr1 = TransitionLayer(
            in_channels=[self.stage1_num_channels[0] * 4],
            out_channels=self.stage2_num_channels,
            name="tr1",
            padding_same=padding_same)

        self.st2 = Stage(
            num_channels=self.stage2_num_channels,
            num_modules=self.stage2_num_modules,
            num_blocks=self.stage2_num_blocks,
            num_filters=self.stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners,
            padding_same=padding_same)

        self.tr2 = TransitionLayer(
            in_channels=self.stage2_num_channels,
            out_channels=self.stage3_num_channels,
            name="tr2",
            padding_same=padding_same)
        self.st3 = Stage(
            num_channels=self.stage3_num_channels,
            num_modules=self.stage3_num_modules,
            num_blocks=self.stage3_num_blocks,
            num_filters=self.stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners,
            padding_same=padding_same)

        self.tr3 = TransitionLayer(
            in_channels=self.stage3_num_channels,
            out_channels=self.stage4_num_channels,
            name="tr3",
            padding_same=padding_same)
        self.st4 = Stage(
            num_channels=self.stage4_num_channels,
            num_modules=self.stage4_num_modules,
            num_blocks=self.stage4_num_blocks,
            num_filters=self.stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners,
            padding_same=padding_same)

    def forward(self, x):
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1([la1])
        st2 = self.st2(tr1)

        tr2 = self.tr2(st2)
        st3 = self.st3(tr2)

        tr3 = self.tr3(st3)
        st4 = self.st4(tr3)

        size = paddle.shape(st4[0])[2:]
        x1 = F.interpolate(
            st4[1], size, mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(
            st4[2], size, mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(
            st4[3], size, mode='bilinear', align_corners=self.align_corners)
        x = paddle.concat([st4[0], x1, x2, x3], axis=1)

        return [x]


class Layer1(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None,
                 padding_same=True):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(num_blocks):
            bottleneck_block = self.add_sublayer(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1),
                    padding_same=padding_same))
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, x):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, name=None, padding_same=True):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)
        num_out = len(out_channels)
        self.conv_bn_func_list = []
        for i in range(num_out):
            residual = None
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.add_sublayer(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBNReLU(
                            in_channels=in_channels[i],
                            out_channels=out_channels[i],
                            kernel_size=3,
                            padding=1 if not padding_same else 'same',
                            bias_attr=False))
            else:
                residual = self.add_sublayer(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBNReLU(
                        in_channels=in_channels[-1],
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding=1 if not padding_same else 'same',
                        bias_attr=False))
            self.conv_bn_func_list.append(residual)

    def forward(self, x):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(x[idx])
            else:
                if idx < len(x):
                    outs.append(conv_bn_func(x[idx]))
                else:
                    outs.append(conv_bn_func(x[-1]))
        return outs


class Branches(nn.Layer):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 has_se=False,
                 name=None,
                 padding_same=True):
        super(Branches, self).__init__()

        self.basic_block_list = []

        for i in range(len(out_channels)):
            self.basic_block_list.append([])
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func = self.add_sublayer(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlock(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                             str(j + 1),
                        padding_same=padding_same))
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, x):
        outs = []
        for idx, input in enumerate(x):
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None,
                 padding_same=True):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            bias_attr=False)

        self.conv2 = ConvBNReLU(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.conv3 = ConvBN(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            bias_attr=False)

        if self.downsample:
            self.conv_down = ConvBN(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                bias_attr=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

        self.add = Add()
        self.relu = Activation("relu")

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = self.add(conv3, residual)
        y = self.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None,
                 padding_same=True):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding=1 if not padding_same else 'same',
            bias_attr=False)
        self.conv2 = ConvBN(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        if self.downsample:
            self.conv_down = ConvBNReLU(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                bias_attr=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

        self.add = Add()
        self.relu = Activation("relu")

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv2 = self.se(conv2)

        y = self.add(conv2, residual)
        y = self.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = paddle.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = []
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,
                        padding_same=padding_same))
            else:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,
                        padding_same=padding_same))

            self.stage_func_list.append(stage_func)

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name,
            padding_same=padding_same)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners,
            padding_same=padding_same)

    def forward(self, x):
        out = self.branches_func(x)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = []
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    residual_func = self.add_sublayer(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBN(
                            in_channels=in_channels[j],
                            out_channels=out_channels[i],
                            kernel_size=1,
                            bias_attr=False))
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBN(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1 if not padding_same else 'same',
                                    bias_attr=False))
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBNReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1 if not padding_same else 'same',
                                    bias_attr=False))
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

    def forward(self, x):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = x[i]
            residual_shape = paddle.shape(residual)[-2:]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = residual + y

            residual = F.relu(residual)
            outs.append(residual)

        return outs


def HRNet_W48(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[48, 96],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[48, 96, 192],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[48, 96, 192, 384],
        **kwargs)
    return model


class OCRNet(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]

        self.head = OCRHead(
            num_classes=num_classes,
            in_channels=in_channels,
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels)

        self.align_corners = align_corners
        self.pretrained = pretrained

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        if not self.training:
            logit_list = [logit_list[0]]

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        return logit_list


class SpatialGatherBlock(nn.Layer):
    """Aggregation layer to compute the pixel-region representation."""

    def __init__(self, pixels_channels, regions_channels):
        super().__init__()
        self.pixels_channels = pixels_channels
        self.regions_channels = regions_channels

    def forward(self, pixels, regions):
        # pixels: from (n, c, h, w) to (n, h*w, c)
        pixels = paddle.reshape(pixels, (0, self.pixels_channels, -1))
        pixels = paddle.transpose(pixels, (0, 2, 1))

        # regions: from (n, k, h, w) to (n, k, h*w)
        regions = paddle.reshape(regions, (0, self.regions_channels, -1))
        regions = F.softmax(regions, axis=2)

        # feats: from (n, k, c) to (n, c, k, 1)
        feats = paddle.bmm(regions, pixels)
        feats = paddle.transpose(feats, (0, 2, 1))
        feats = paddle.unsqueeze(feats, axis=-1)

        return feats


class SpatialOCRModule(nn.Layer):
    """Aggregate the global object representation to update the representation for each pixel."""

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 dropout_rate=0.1):
        super().__init__()

        self.attention_block = ObjectAttentionBlock(in_channels, key_channels)
        self.conv1x1 = nn.Sequential(
            ConvBNReLU(2 * in_channels, out_channels, 1),
            nn.Dropout2D(dropout_rate))

    def forward(self, pixels, regions):
        context = self.attention_block(pixels, regions)
        feats = paddle.concat([context, pixels], axis=1)
        feats = self.conv1x1(feats)

        return feats


class ObjectAttentionBlock(nn.Layer):
    """A self-attention module."""

    def __init__(self, in_channels, key_channels):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels

        self.f_pixel = nn.Sequential(
            ConvBNReLU(in_channels, key_channels, 1),
            ConvBNReLU(key_channels, key_channels, 1))

        self.f_object = nn.Sequential(
            ConvBNReLU(in_channels, key_channels, 1),
            ConvBNReLU(key_channels, key_channels, 1))

        self.f_down = ConvBNReLU(in_channels, key_channels, 1)

        self.f_up = ConvBNReLU(key_channels, in_channels, 1)

    def forward(self, x, proxy):
        x_shape = paddle.shape(x)
        # query : from (n, c1, h1, w1) to (n, h1*w1, key_channels)
        query = self.f_pixel(x)
        query = paddle.reshape(query, (0, self.key_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        # key : from (n, c2, h2, w2) to (n, key_channels, h2*w2)
        key = self.f_object(proxy)
        key = paddle.reshape(key, (0, self.key_channels, -1))

        # value : from (n, c2, h2, w2) to (n, h2*w2, key_channels)
        value = self.f_down(proxy)
        value = paddle.reshape(value, (0, self.key_channels, -1))
        value = paddle.transpose(value, (0, 2, 1))

        # sim_map (n, h1*w1, h2*w2)
        sim_map = paddle.bmm(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        # context from (n, h1*w1, key_channels) to (n , out_channels, h1, w1)
        context = paddle.bmm(sim_map, value)
        context = paddle.transpose(context, (0, 2, 1))
        context = paddle.reshape(context,
                                 (0, self.key_channels, x_shape[2], x_shape[3]))
        context = self.f_up(context)

        return context


class OCRHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 ocr_mid_channels=512,
                 ocr_key_channels=256):
        super().__init__()

        self.num_classes = num_classes
        self.spatial_gather = SpatialGatherBlock(ocr_mid_channels, num_classes)
        self.spatial_ocr = SpatialOCRModule(ocr_mid_channels, ocr_key_channels,
                                            ocr_mid_channels)

        self.indices = [-2, -1] if len(in_channels) > 1 else [-1, -1]

        self.conv3x3_ocr = ConvBNReLU(
            in_channels[self.indices[1]], ocr_mid_channels, 3, padding=1)
        self.cls_head = nn.Conv2D(ocr_mid_channels, self.num_classes, 1)
        self.aux_head = nn.Sequential(
            ConvBNReLU(in_channels[self.indices[0]],
                       in_channels[self.indices[0]], 1),
            nn.Conv2D(in_channels[self.indices[0]], self.num_classes, 1))

    def forward(self, feat_list):
        feat_shallow, feat_deep = feat_list[self.indices[0]], feat_list[
            self.indices[1]]

        soft_regions = self.aux_head(feat_shallow)
        pixels = self.conv3x3_ocr(feat_deep)

        object_regions = self.spatial_gather(pixels, soft_regions)
        ocr = self.spatial_ocr(pixels, object_regions)

        logit = self.cls_head(ocr)
        return [logit, soft_regions]


def process(src_image_dir, save_dir):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    backbone = HRNet_W48()
    model = OCRNet(num_classes=2, backbone=backbone, backbone_indices=[0])
    param_dict = paddle.load('./model.pdparams')
    model.load_dict(param_dict)
    model.eval()
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    with paddle.no_grad():
        for image_path in image_paths:
            # do something
            im = cv2.imread(image_path)
            raw_h, raw_w = im.shape[:2]
            w, h = 512, 512
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            im = im.astype(np.float32, copy=False) / 255.0
            im -= mean
            im /= std
            im = im.transpose((2, 0, 1))
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)
            logits = model(im)
            pred = paddle.argmax(logits[0], axis=1, keepdim=True, dtype='int32')
            pred = paddle.squeeze(pred).numpy() * 255
            out_image = paddle.vision.transforms.resize(pred, (raw_h, raw_w), interpolation='bilinear')

            # 保存结果图片
            save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
            cv2.imwrite(save_path, out_image)


def get_main_point(image_gray):
    image_gray = np.uint8(image_gray)
    # canny边缘检测
    img_edge = cv2.Canny(image_gray, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    img_edge = cv2.dilate(img_edge, kernel, iterations=3)
    # 寻找边界轮廓
    contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    cv2.drawContours(image_gray, contours, -1, (0, 0, 0), 1)
    biggest_contour = contours[0]
    biggest_len = cv2.arcLength(biggest_contour, True)
    for contour in contours:
        # 获取contour长度
        lenght = cv2.arcLength(contour, True)
        if lenght > biggest_len:
            biggest_contour = contour
            biggest_len = lenght
    # 拟合多边形
    min_epsilon = 0
    max_epsilon = 500
    approx = cv2.approxPolyDP(biggest_contour, (max_epsilon + min_epsilon) / 2, True)
    while len(approx) != 4:
        epsilon = (max_epsilon + min_epsilon) / 2
        if len(approx) > 4:
            min_epsilon = epsilon
        else:
            max_epsilon = epsilon
        approx = cv2.approxPolyDP(biggest_contour, (max_epsilon + min_epsilon) / 2, True)
    return approx


def show(img):
    plt.imshow(img)
    plt.show()


def sort_pts(pts, w, h):
    dis1 = pts[:, :, 1] + pts[:, :, 0]
    dis2 = np.square(pts[:, :, 1] - h) + np.square(pts[:, :, 0])
    dis3 = np.square(pts[:, :, 1]) + np.square(pts[:, :, 0] - w)
    dis4 = np.square(pts[:, :, 1] - h) + np.square(pts[:, :, 0] - w)
    sort_result = np.array([pts[np.argmin(dis1)], pts[np.argmin(dis2)], pts[np.argmin(dis3)], pts[np.argmin(dis4)]])
    return sort_result


def predict_one_image(image_path):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    backbone = HRNet_W48()
    model = OCRNet(num_classes=2, backbone=backbone, backbone_indices=[0])
    param_dict = paddle.load('./model.pdparams')
    model.load_dict(param_dict)
    model.eval()
    with paddle.no_grad():
        # do something
        im = cv2.imread(image_path)
        # 保持图形的比例放大
        im = cv2.resize(im, (1500, int(im.shape[0] * 1500 / im.shape[1])), interpolation=cv2.INTER_LINEAR)
        im_copy = np.copy(im)
        raw_h, raw_w = im.shape[:2]
        w, h = 512, 512
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)
        logits = model(im)
        pred = paddle.argmax(logits[0], axis=1, keepdim=True, dtype='int32')
        pred = paddle.squeeze(pred).numpy() * 255
        pred = pred.astype(np.uint8)
        out_image = paddle.vision.transforms.resize(pred, (raw_h, raw_w), interpolation='bilinear')
        out_image = cv2.copyMakeBorder(out_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        im = im_copy
        im = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        main_point = get_main_point(out_image)
        # 仿射变换
        h, w = im.shape[0], im.shape[1]
        # 对坐标排序
        src_pts = sort_pts(np.float32(main_point), w, h)
        scan_h = max(np.sqrt(np.square(src_pts[0][0][0] - src_pts[1][0][0]) + np.square(src_pts[0][0][1] - src_pts[1][0][1])), np.sqrt(np.square(src_pts[2][0][0] - src_pts[3][0][0]) + np.square(src_pts[2][0][1] - src_pts[3][0][1])))
        scan_w = max(np.sqrt(np.square(src_pts[0][0][0] - src_pts[2][0][0]) + np.square(src_pts[0][0][1] - src_pts[2][0][1])), np.sqrt(np.square(src_pts[1][0][0] - src_pts[3][0][0]) + np.square(src_pts[1][0][1] - src_pts[3][0][1])))
        dst_pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        out_image = cv2.warpPerspective(im, M, (w, h))
        out_image = cv2.resize(out_image, (int(scan_w), int(scan_h)), interpolation=cv2.INTER_LINEAR)
        ## 文档后处理增强
        # 灰度化
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
        sharpen = cv2.GaussianBlur(out_image, (0, 0), 3)
        sharpen = cv2.addWeighted(out_image, 1.5, sharpen, -0.5, 0)
        out_image = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        return out_image


if __name__ == "__main__":

    src_image_dir = './images'
    save_dir = './results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result = predict_one_image(
        'images/IMG_0817.JPG')
    cv2.imwrite('./output.jpg', result)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    show(result)
    # process(src_image_dir, save_dir)
    # process(r'C:\Users\david\Desktop\train_datasets_document_detection_0411\images', './')
