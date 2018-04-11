from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


def make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes[0] != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes[0], planes * block.expansion, 1, stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        )
    layers = []
    layers.append(block(inplanes[0], planes, stride, downsample))
    inplanes[0] = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes[0], planes))
    return nn.Sequential(*layers)


def resnet_feat(block, layers):
    feat = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)]
    inplanes = [64]
    planes = 64
    for i in range(4):
        feat += make_layer(block, inplanes, planes, layers[i], 2 if i in [1, 2] else 1)
        planes *= 2
    return feat


def resnet18_feat():
    return resnet_feat(BasicBlock, [2, 2, 2, 2])


def resnet34_feat():
    return resnet_feat(BasicBlock, [3, 4, 6, 3])


def resnet50_feat():
    return resnet_feat(Bottleneck, [3, 4, 6, 3])


def resnet101_feat():
    return resnet_feat(Bottleneck, [3, 4, 23, 3])


def resnet152_feat():
    return resnet_feat(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    import torch

    feat = resnet101_feat()
    net = nn.Sequential(*feat)
    print(net)
    # img = torch.randn((1, 3, 300, 300))
    # img = net(img)
    # print(img.size())
