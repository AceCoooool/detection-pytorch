from torch import nn

cfg = {
    'voc': {'tiny': False, 'voc': True, 'num': 18, 'flag': [1, 1, 0] * 3 + [1, 0, 1] * 2 + [0, 1, 0],
            'size_flag': [3, 6, 9, 11, 14, 16], 'pool': [0, 1, 4, 7, 13]},
    'coco': {'tiny': False, 'voc': False, 'num': 18, 'flag': [1, 1, 0] * 3 + [1, 0, 1] * 2 + [0, 1, 0],
             'size_flag': [3, 6, 9, 11, 14, 16], 'pool': [0, 1, 4, 7, 13]},
    'tiny-voc': {'tiny': True, 'voc': True, 'num': 7, 'flag': [1] * 7,
                 'size_flag': [], 'pool': [0, 1, 2, 3, 4]},
    'tiny-coco': {'tiny': True, 'voc': False, 'num': 7, 'flag': [1] * 7,
                  'size_flag': [], 'pool': [0, 1, 2, 3, 4]},
}


# module1: conv+bn+leaky_relu
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# reorg layer
class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.size()
        s = self.stride
        x = x.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x = x.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        return x.view(B, s * s * C, H // s, W // s)


# Note: use cfg for expand to tiny darknet
def darknet(cfg=cfg['voc']):
    in_c, out_c = 3, 16 if cfg['tiny'] else 32
    flag, pool, size_flag = cfg['flag'], cfg['pool'], cfg['size_flag']
    layers = []
    for i in range(cfg['num']):
        ksize = 1 if i in size_flag else 3
        if i < 13:
            layers.append(ConvLayer(in_c, out_c, ksize, same_padding=True))
            layers.append(nn.MaxPool2d(2)) if i in pool else None
            layers += [nn.ReflectionPad2d([0, 1, 0, 1]), nn.MaxPool2d(2, 1)] if i == 5 and cfg['tiny'] else []
        else:
            layers.append(nn.MaxPool2d(2)) if i in pool else None
            layers.append(ConvLayer(in_c, out_c, ksize, same_padding=True))
        in_c, out_c = out_c, out_c * 2 if flag[i] else out_c // 2
    return layers


if __name__ == '__main__':
    import torch

    net = nn.Sequential(*darknet(cfg['voc']))
    img = torch.randn((1, 3, 416, 416))
    print(net(img).size())
