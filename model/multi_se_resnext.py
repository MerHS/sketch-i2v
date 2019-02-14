"""
PyTorch SE-ResNet implementation 
Block Definition from moskomule/senet.pytorch
https://github.com/moskomule/senet.pytorch/blob/master/senet/baseline.py
"""
import math, torch
import torch.nn as nn
from torchvision.models import ResNet

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, reduction=16):
        super(SEBottleneckX, self).__init__()

        D = int(math.floor(planes * (base_width/64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, bias=False, groups=C)
        self.bn2 = nn.BatchNorm2d(D*C)
        
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MiniUNet(nn.Module):
    def __init__(self, inplanes, classes):
        super(MiniUNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.down1 = self.conv(32, 64, stride=2)
        self.down2 = self.conv(64, 128, stride=2)
        self.down3 = self.conv(128, 128)

        self.up1 = self.conv(128, 64)
        self.up2 = self.conv(64, 32)
        self.up_conv1 = self.up_conv(128, 64)
        self.up_conv2 = self.up_conv(64, 32)

        self.segmentize = nn.Sequential(
            nn.Conv2d(32, classes, 3, padding=1),
            nn.Sigmoid()
        )

    def conv(self, in_channel, out_channel, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = self.up_conv1(x3)
        x = torch.cat([x1, x], 1)
        x = self.up1(x)

        x = self.up_conv2(x)
        x = torch.cat([x0, x], 1)
        x = self.up2(x)

        x = self.segmentize(x)
        return x

class MultiSEResNeXt(nn.Module):
    def __init__(self, block, layers, class_list, input_channels=3, cardinality=32, base_width=4):
        super(MultiSEResNeXt, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.inplanes = 64
        self.class_list = class_list

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer_mask = MiniUNet(128 * block.expansion, len(class_list))

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_list = nn.ModuleList([])
        for class_name, class_part_list in class_list:
            self.classifier_list.append(
                nn.Linear(512 * block.expansion, len(class_part_list))
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, block_count, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, block_count):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x_mid = self.layer2(x)

        mask = self.layer_mask(x_mid)
        out = []
        for i, classifier in enumerate(self.classifier_list):
            class_mask = mask[:, i, :, :].unsqueeze(1)
            x = x_mid * class_mask

            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = classifier(x)
            out.append(x)
            
        out = torch.cat(out, 1)
        return out

    def get_mask(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        mask = self.layer_mask(x)
        return mask


def multi_serx50(class_list, input_channels):
    model = MultiSEResNeXt(SEBottleneckX, [3, 4, 6, 3], class_list=class_list, input_channels=input_channels)
    return model