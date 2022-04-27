'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from .configs import InfoPro, InfoPro_balanced_memory
from model.generative.aux_net_configs import InfoPro
from model.generative.auxiliary_nets import Decoder, AuxClassifier

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, args=None, image_size=32, model_input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.args = args
        self.image_size = image_size

        self.conv1 = nn.Conv2d(model_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layers_name_map = {
            "classifier": "linear"
        }

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer1":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 1. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.layer2(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer2":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 2. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.layer3(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer3":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 3. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.layer4(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer4":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 4. feat shape: {feat.shape}, out.shape: {out.shape}")
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "last":
            # feat = out
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat before last layer. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.linear(out.view(out.size(0), -1))

        if self.args.model_out_feature:
            return out, feat
        else:
            return out


class InfoProResNet(nn.Module):
    # def __init__(self, block, num_blocks, num_classes=10):
    # def __init__(self, block, num_blocks, arch, infopro_module_num, batch_size, image_size=32,
    #             balanced_memory=False, dataset='cifar10', num_classes=10,
    #             infopro_wide_list=(16, 16, 32, 64), dropout_rate=0,
    #             infopro_aux_cls_config='1c2f', local_loss_mode='contrast',
    #             infopro_aux_widen=1, infopro_aux_cls_feature_dim=128):
    def __init__(self, block, num_blocks,
                num_classes=10, args=None, image_size=32):
        super(InfoProResNet, self).__init__()
        self.in_planes = 64

        self.arch = args.model
        assert self.arch in ['resnet18_v2'], "This repo supports resnet18 currently. " \
                    "For other networks, please set network configs in .configs."

        self.args = args 

        self.dataset = args.dataset
        self.infopro_module_num = args.infopro_module_num
        self.infopro_wide_list = args.infopro_wide_list
        # dropout_rate=0,
        self.infopro_aux_decoder_config = args.infopro_aux_decoder_config
        self.infopro_aux_cls_config = args.infopro_aux_cls_config
        # self.local_loss_mode = args.local_loss_mode
        self.infopro_aux_widen = args.infopro_aux_widen
        self.infopro_aux_cls_feature_dim = args.infopro_aux_cls_feature_dim


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.layers_name_map = {
            "classifier": "linear"
        }

        try:
            self.infopro_config = InfoPro[self.arch][self.infopro_module_num]
        except:
            raise NotImplementedError

        self.decoder = Decoder(self.infopro_wide_list[-1], image_size, z_dim=64, h_dim=512,
                    net_config=self.infopro_aux_decoder_config, widen=self.infopro_aux_widen)

        # for item in self.infopro_config:
        #     module_index, layer_index = item

        #     exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
        #         '= Decoder(self.infopro_wide_list[module_index], image_size, widen=self.infopro_aux_widen)')

            # exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
            #      '= AuxClassifier(self.infopro_wide_list[module_index], net_config=self.infopro_aux_cls_config, '
            #      'loss_mode=local_loss_mode, class_num=class_num, '
            #      'widen=self.infopro_aux_widen, feature_dim=self.infopro_aux_cls_feature_dim)')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        if self.dataset == 'cifar10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
            self.mask_train_mean = torch.Tensor([CIFAR_MEAN]).view(1, 3, 1, 1).expand(
                args.batch_size, 3, image_size, image_size
            )
            self.mask_train_std = torch.Tensor(CIFAR_STD).view(1, 3, 1, 1).expand(
                args.batch_size, 3, image_size, image_size
            )
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                args.batch_size, 3, image_size, image_size
            )
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                args.batch_size, 3, image_size, image_size
            )

    def to_device(self, device):
        self.mask_train_mean = self.mask_train_mean.to(device)
        self.mask_train_std = self.mask_train_std.to(device)

    def _image_restore(self, normalized_image):
        # logging.debug(f"normalized_image.device: {normalized_image.device}, self.mask_train_std.device: {self.mask_train_std.device}")
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
            + self.mask_train_mean[:normalized_image.size(0)]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, img, target=None):

        if self.training:
            loss_inner = 0.0
            out = F.relu(self.bn1(self.conv1(img)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            # loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(img))
            if self.args.infopro_IXH:
                loss_ixx, decode_img = eval('self.decoder')(out, self._image_restore(img))
            loss_inner += loss_ixx

            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out, loss_inner, decode_img

        else:
            out = F.relu(self.bn1(self.conv1(img)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


def ResNet10(args, num_classes=10, **kwargs):
    if args.infopro:
        return InfoProResNet(BasicBlock, [1,1,1,1], num_classes=num_classes, args=args, **kwargs)
    else:
        return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes, args=args, **kwargs)

def ResNet18(args, num_classes=10, **kwargs):
    if args.infopro:
        return InfoProResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, args=args, **kwargs)
    else:
        return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, args=args, **kwargs)

def ResNet34(args, num_classes=10, **kwargs):
    if args.infopro:
        return InfoProResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, args=args, **kwargs)
    else:
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, args=args, **kwargs)

def ResNet50(args, num_classes=10, **kwargs):
    if args.infopro:
        return InfoProResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, args=args, **kwargs)
    else:
        return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, args=args, **kwargs)

def ResNet101(args, num_classes=10, **kwargs):
    if args.infopro:
        return InfoProResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, args=args, **kwargs)
    else:
        return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, args=args, **kwargs)

def ResNet152(args, num_classes=10, **kwargs):
    if args.infopro:
        return InfoProResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, args=args, **kwargs)
    else:
        return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, args=args, **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()















