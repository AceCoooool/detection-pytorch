import torch
from torch import nn
from torch.nn import functional as F

from ssd import config as cfg
from basenet.vgg import vgg_feat
from basenet.resnet import resnet101_feat
from ssd.utils_ssd.priorbox import PriorBox
from ssd.utils_ssd.L2Norm import L2Norm
from ssd.utils_ssd.detect import Detect


# extend vgg: 5 "additional" feature parts
def add_extras(cfg, i, vgg=True):
    fc7 = [nn.MaxPool2d(3, 1, 1), nn.Conv2d(512, 1024, 3, 1, 6, 6), nn.ReLU(inplace=True),
           nn.Conv2d(1024, 1024, 1), nn.ReLU(inplace=True)] if vgg else []
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return fc7, layers


# feature map to loc+conf
def multibox(cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for channel, n in cfg:
        loc_layers += [nn.Conv2d(channel, n * 4, 3, 1, 1)]
        conf_layers += [nn.Conv2d(channel, n * num_classes, 3, 1, 1)]
    return loc_layers, conf_layers


# single shot multibox detector
class SSD(nn.Module):
    def __init__(self, phase, base, extras, loc, conf, num_classes, l=[23, 512]):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priors = PriorBox(cfg.v2)()
        self.size = 300
        self.l = l[0]

        self.bone = nn.ModuleList(base)
        self.l2norm = L2Norm(l[1], 20)
        self.extras = nn.ModuleList(extras)
        self.loc, self.conf = nn.ModuleList(loc), nn.ModuleList(conf)

        if phase == 'test':
            self.detect = Detect(num_classes, cfg.top_k, cfg.conf_thresh, cfg.nms_thresh)

    def forward(self, x):
        source, loc, conf = list(), list(), list()
        for k in range(self.l):
            x = self.bone[k](x)
        source.append(self.l2norm(x))
        for k in range(self.l, len(self.bone)):
            x = self.bone[k](x)
        source.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                source.append(x)
        # apply multibox head to source layers
        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if not self.priors.is_cuda and loc.is_cuda:
            self.priors = self.priors.cuda()
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                F.softmax(conf.view(conf.size(0), -1, self.num_classes), dim=2),
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output


def build_ssd(phase, size=300, num_classes=21, bone='vgg'):
    if phase != 'test' and phase != 'train':
        assert "Error: Phase not recognized"
    if size != 300:
        assert "Error: Only SSD300 us supported"
    if bone == 'vgg':
        base_ = vgg_feat()
        fc7_, extras_ = add_extras(cfg.extras_vgg['300'], 1024)
        loc_, conf_ = multibox(cfg.mbox_vgg['300'], num_classes)
        l = cfg.l_vgg
    elif bone == 'res101':
        base_ = resnet101_feat()
        fc7_, extras_ = add_extras(cfg.extras_res['300'], 2048, False)
        loc_, conf_ = multibox(cfg.mbox_res['300'], num_classes)
        l = cfg.l_res
    else:
        raise IOError("only vgg or res101")
    return SSD(phase, base_ + fc7_, extras_, loc_, conf_, num_classes, l)


if __name__ == '__main__':
    net = build_ssd('train', bone='vgg')
    bone = net.bone
    print(bone)
    img = torch.randn((1, 3, 300, 300))
    out = net(img)
    print(out[1])
