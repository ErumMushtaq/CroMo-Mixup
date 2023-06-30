''' My own implementation of ResNet, which can produce the standard ImageNet architecture as well as 
two different CIFAR-10 architectures (compare resnet18, resnetc18, and resnetc20 for illustration).'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class PaddedIdentity(nn.Module):

    def __init__(self, padfunc):
        super().__init__()
        self.padfunc = padfunc

    def forward(self, x):
        return self.padfunc(x)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BasicBlock(nn.Module):

    def __init__(self, input_channels, output_channels, first_stride, projection=True, normalization = 'batch', weight_standard = False):

        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_stride = first_stride
        self.projection = projection

        self.act = OrderedDict()
        self.count = 0

        if weight_standard == True:
            self.conv1 = Conv2d(input_channels, output_channels, 
                                    kernel_size=3, stride=first_stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(input_channels, output_channels, 
                                   kernel_size=3, stride=first_stride, padding=1, bias=False)

        if normalization == 'batch':
            self.bn1 = nn.BatchNorm2d(output_channels)
        elif normalization == 'group':
            self.bn1 = nn.GroupNorm(32,output_channels)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        if weight_standard == True:
            self.conv2 = Conv2d(output_channels, output_channels, 
                                    kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(output_channels, output_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False)

        if normalization == 'batch':
            self.bn2 = nn.BatchNorm2d(output_channels)
        elif normalization == 'group':
            self.bn2 = nn.GroupNorm(32,output_channels)
        else:
            self.bn2 = nn.Identity()
        self.shortcut = nn.Sequential()        
        if self.first_stride != 1 or self.input_channels != self.output_channels:
            # Option A: downsample spatially, pad with zeros depth-wise
            if self.projection == False:
                self.shortcut = PaddedIdentity(lambda x: F.pad(x[:, :, ::2, ::2], 
                                                           (0, 0, 0, 0, self.output_channels//4, 
                                                            self.output_channels//4)))
            # Option B: project onto new dimension
            else:
                if weight_standard == True:
                    conv = Conv2d(self.input_channels, self.output_channels, 
                                 kernel_size=1, stride=self.first_stride, bias=False)
                else:
                    conv = nn.Conv2d(self.input_channels, self.output_channels, 
                                 kernel_size=1, stride=self.first_stride, bias=False)
                                 
                if normalization == 'batch':
                    bn = nn.BatchNorm2d(output_channels)
                elif normalization == 'group':
                    bn = nn.GroupNorm(32,output_channels)
                else:
                    bn = nn.Identity()

                self.shortcut = nn.Sequential(conv, bn)

    def forward(self, x):

     
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)



        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x) 
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channels, output_channels_list, 
                 layer_depths, num_classes, c1_kernel, c1_stride, c1_pad, maxpool=True, normalization = 'batch', weight_standard = False):

        super().__init__()
        self.input_channels = input_channels
        self.output_channels_list = output_channels_list
        self.layer_depths = layer_depths
        self.channels = output_channels_list[0]
        self.maxpool = maxpool

        self.act = OrderedDict()

        if weight_standard == True:
            self.conv1 = Conv2d(input_channels, output_channels_list[0], 
                                kernel_size=c1_kernel, stride=c1_stride, padding=c1_pad, bias=False)
        else:
            self.conv1 = nn.Conv2d(input_channels, output_channels_list[0], 
                                kernel_size=c1_kernel, stride=c1_stride, padding=c1_pad, bias=False)

        if normalization == 'batch':
            self.bn = nn.BatchNorm2d(output_channels_list[0])
        elif normalization == 'group':
            self.bn = nn.GroupNorm(32,output_channels_list[0])
        else:
            self.bn = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        if self.maxpool is True:
            self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = self._make_layer(BasicBlock, layer_depths[0], output_channels_list[0], first_stride=1, normalization = normalization, weight_standard = weight_standard)
        self.res2 = self._make_layer(BasicBlock, layer_depths[1], output_channels_list[1], first_stride=2, normalization = normalization, weight_standard = weight_standard)
        self.res3 = self._make_layer(BasicBlock, layer_depths[2], output_channels_list[2], first_stride=2, normalization = normalization, weight_standard = weight_standard)
        self.res4 = self._make_layer(BasicBlock, layer_depths[3], output_channels_list[3], first_stride=2, normalization = normalization, weight_standard = weight_standard)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(output_channels_list[3], num_classes)
                
        self._weights_init()

    def forward(self, x):

        self.act['conv_in'] = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.maxpool is True:
            out = self.mp(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)     
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels_list[3])
        out = self.fc(out)

        return out

    def _make_layer(self, block_class, layer_depth, output_channels, first_stride, normalization = 'batch', weight_standard = False):

        if layer_depth is None:
            return nn.Sequential()

        strides = [first_stride] + [1] * (layer_depth - 1)
        layers = []
        for s in strides:
            layers.append(block_class(self.channels, output_channels, first_stride = s, normalization = normalization, weight_standard = weight_standard))
            self.channels = output_channels

        return nn.Sequential(*layers)

    def _weights_init(self):

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0) # See https://arxiv.org/abs/1706.02677


def resnet18():
    return ResNet(input_channels=3, output_channels_list=[64, 128, 256, 512], layer_depths=[2,2,2,2], num_classes=1000, c1_kernel=7, c1_stride=2, c1_pad=3, maxpool=True)

def resnetc18(nclass, normalization = 'batch', weight_standard = False):
    return ResNet(input_channels=3, output_channels_list=[64, 128, 256, 512], layer_depths=[2,2,2,2], num_classes=nclass, c1_kernel=3, c1_stride=1, c1_pad=1, maxpool=False,normalization = normalization, weight_standard = weight_standard)

def resnetc20():
    return ResNet(input_channels=3, output_channels_list=[16, 32, 64, 64], layer_depths=[3,3,3,None], num_classes=10, c1_kernel=3, c1_stride=1, c1_pad=1, maxpool=False)