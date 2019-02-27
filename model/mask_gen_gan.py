import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_resnet import BottleneckX, SEResNeXt

def make_resnext_layer(block, inplanes, planes, block_count, stride=1):
    outplanes = planes // 4
    downsample = None
    if stride != 1 or inplanes != outplanes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, outplanes, 16, stride, downsample))
    for i in range(1, block_count):
        layers.append(block(planes, outplanes, 16))

    return nn.Sequential(*layers)

class MaskGenerator(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, input_size=256, layers=[3, 3, 3, 3]):
        super(MaskGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv1 = self._make_encoder_block_first(1, 32)
        self.conv2 = self._make_encoder_block(32, 64)
        self.conv3 = self._make_encoder_block(64, 128)
        self.conv4 = self._make_encoder_block(128, 256)
        self.conv5 = self._make_encoder_block(256, 512)
        
        self.deconv1 = self._make_block(512, 256, layers[0])
        self.deconv2 = self._make_block(256 + 256, 128, layers[1])
        self.deconv3= self._make_block(128 + 128, 64, layers[2])
        self.deconv4 = self._make_block(64 + 64, 32, layers[3])
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, self.output_dim, 3, 1, 1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_block(self, inplanes, planes, block_num):
        return nn.Sequential(
            make_resnext_layer(BottleneckX, inplanes, planes*4, block_num),
            nn.PixelShuffle(2),
        )
    
    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        # ==============================
        out4_prime = self.deconv1(out5)

        concat_tensor = torch.cat([out4_prime, out4], 1)
        out3_prime = self.deconv2(concat_tensor)

        concat_tensor = torch.cat([out3_prime, out3], 1)
        out2_prime = self.deconv3(concat_tensor)

        concat_tensor = torch.cat([out2_prime, out2], 1)
        out1_prime = self.deconv4(concat_tensor)

        concat_tensor = torch.cat([out1_prime, out1], 1)
        full_output = self.deconv5(concat_tensor)

        return full_output


class MaskDiscriminator(SEResNeXt):
    """SE-ResNeXt 50 based discriminator"""
    def __init__(self, block, layers, input_channels=3, cardinality=32, num_classes=1000):
        super(MaskDiscriminator, self).__init__(block, layers, input_channels, cardinality, num_classes)
        self.adv_fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        class_x = self.fc(x)
        class_x = self.classifier(class_x)
        adv_x = self.adv_fc(x)
        
        return class_x, adv_x


def get_mask_disc(**kwargs):
    model = MaskDiscriminator(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model