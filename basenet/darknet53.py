from torch import nn


# module1: conv+bn+leaky_relu
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=True):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplane, plane):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(inplane, plane, 1)
        self.conv2 = ConvLayer(plane, inplane, 3)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out


def darknet_feat(layers=[1, 2, 8, 8, 4], block=BasicBlock):
    feat = [ConvLayer(3, 32, 3), ConvLayer(32, 64, 3, 2)]
    inplane, plane, total = 64, 32, len(layers)
    for i, num in enumerate(layers):
        for j in range(num):
            feat.append(block(inplane, plane))
        feat.append(ConvLayer(inplane, 2 * inplane, 3, 2)) if i < total - 1 else None
        inplane, plane = 2 * inplane, 2 * plane
    return feat


if __name__ == '__main__':
    import torch
    feat = darknet_feat()
    feat = nn.Sequential(*feat)
    img = torch.ones((1, 3, 416, 416))
    out = feat(img)
    print(out)