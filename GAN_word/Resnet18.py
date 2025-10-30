import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, nb_feat=384, in_channels=3):
        super(ResNet18, self).__init__()
        self.inplanes = nb_feat // 4  # = 96 if nb_feat=384

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=1)

        self.layer1 = self._make_layer(BasicBlock, nb_feat // 4, 2, stride=(2, 2))  # 96
        self.layer2 = self._make_layer(BasicBlock, nb_feat // 2, 2, stride=2)       # 192
        self.layer3 = self._make_layer(BasicBlock, nb_feat, 2, stride=2)            # 384

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        results = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        results.append(x)  # 0

        x = self.layer1(x)
        results.append(x)  # 1

        x = self.layer2(x)
        results.append(x)  # 2

        x = self.layer3(x)
        results.append(x)  # 3

        x = self.maxpool2(x)
        results.append(x)  # 4 → used as feat_embed in mix()

        return results  # each is [B, C, H, W]
