from torch import nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'C', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'C', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


# vgg feature part: remove last max-pooling
def vgg_feat(cfg=cfg['D'], i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


def convertweight(vgg_trained, root):
    torch.save(vgg_trained.features.state_dict(), root)


if __name__ == '__main__':
    import torch
    net = vgg_feat(cfg['D'], 3)
    net = nn.Sequential(*net)
    img = torch.randn((1, 3, 300, 300))
    res = net(img)
    print(res.size())
