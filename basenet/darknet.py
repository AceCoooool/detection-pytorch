from torch import nn


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


def darknet(cfg):
    in_c, out_c = 3, 16 if cfg.tiny else 32
    flag, pool, size_flag = cfg.flag, cfg.pool, cfg.size_flag
    layers = []
    for i in range(cfg.num):
        ksize = 1 if i in size_flag else 3
        if i < 13:
            layers.append(ConvLayer(in_c, out_c, ksize, same_padding=True))
            layers.append(nn.MaxPool2d(2)) if i in pool else None
            layers += [nn.ReflectionPad2d([0, 1, 0, 1]), nn.MaxPool2d(2, 1)] if i == 5 and cfg.tiny else []
        else:
            layers.append(nn.MaxPool2d(2)) if i in pool else None
            layers.append(ConvLayer(in_c, out_c, ksize, same_padding=True))
        in_c, out_c = out_c, out_c * 2 if flag[i] else out_c // 2
    return layers


if __name__ == '__main__':
    from yolo.config import yolo_voc as cfg
    import torch
    import cv2
    from torchvision import transforms
    net = nn.ModuleList(darknet(cfg))
    net.load_state_dict(torch.load('../weights/yolo/darknet.pth'))

    img = cv2.imread('../results/dog.jpg')
    img = cv2.resize(img, (416, 416))
    t = transforms.ToTensor()
    img = t(img).unsqueeze(0)
    for i in range(len(net)):
        img = net[i](img)

