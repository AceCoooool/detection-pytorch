import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from basenet.darknet import darknet, ConvLayer, ReorgLayer
from yolo.utils_yolo.priorbox import PriorBox
from yolo.utils_yolo.detect import Detect
from yolo.config import yolo_voc


class Yolo(nn.Module):
    def __init__(self, phase, cfg, eval=False):
        super(Yolo, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.darknet = nn.ModuleList(darknet(cfg))
        self.anchors = np.array(cfg.anchors).reshape(-1, 2)
        self.class_num, self.anchor_num = cfg.class_num, self.anchors.shape[0]
        self.feat_size = 13
        self.priors = PriorBox(self.anchors)()
        if cfg.tiny:
            out = 1024 if cfg.voc else 512
            self.conv = nn.Sequential(
                ConvLayer(1024, out, 3, same_padding=True),
                nn.Conv2d(out, cfg.anchor_num * (cfg.class_num + 5), 1)
            )
        else:
            self.conv2 = nn.Sequential(
                ConvLayer(1024, 1024, 3, same_padding=True),
                ConvLayer(1024, 1024, 3, same_padding=True))
            self.conv1 = nn.Sequential(
                ConvLayer(512, 64, 1, same_padding=True),
                ReorgLayer(2))
            self.conv = nn.Sequential(
                ConvLayer(1280, 1024, 3, same_padding=True),
                nn.Conv2d(1024, cfg.anchor_num * (cfg.class_num + 5), 1))
        if phase == 'test':
            self.detect = Detect(cfg, eval)

    def forward(self, x):
        if self.cfg.tiny:
            for i in range(len(self.darknet)):
                x = self.darknet[i](x)
            x = self.conv(x)
        else:
            for i in range(17):
                x = self.darknet[i](x)
            x1 = self.conv1(x)
            for i in range(17, len(self.darknet)):
                x = self.darknet[i](x)
            x2 = self.conv2(x)
            x = self.conv(torch.cat([x1, x2], 1))
        if not self.priors.is_cuda and x.is_cuda:
            self.priors = self.priors.cuda()
        b, c, h, w = x.size()
        if self.feat_size != h:
            self.priors = PriorBox(self.anchors, h)()
            self.feat_size = h
            self.priors = self.priors.cuda() if x.is_cuda else self.priors
        if self.priors.size(0) != b:
            self.priors = self.priors.repeat((b, 1, 1, 1))
        feat = x.permute(0, 2, 3, 1).contiguous().view(b, -1, self.anchor_num, self.class_num + 5)
        box_xy, box_wh = F.sigmoid(feat[..., :2]), feat[..., 2:4].exp()
        box_xy += self.priors[..., 0:2]
        box_wh *= self.priors[..., 2:]
        box_conf, box_prob = F.sigmoid(feat[..., 4:5]), feat[..., 5:]
        box_pred = torch.cat([box_xy, box_wh], 3) / h
        if self.phase == 'test':
            output = self.detect(box_pred, box_conf, F.softmax(box_prob, dim=3))
        else:
            output = (feat, box_pred, box_conf, box_prob)
        return output


def build_yolo(phase, cfg=yolo_voc, eval=False):
    if phase != 'test' and phase != 'train':
        assert "Error: Phase not recognized"
    return Yolo(phase, cfg, eval)


if __name__ == '__main__':
    from yolo.utils_yolo.multiloss import YoloLoss
    import yolo.config.yolo_voc as cfg

    net = build_yolo('train')
    net = net.cuda()
    # net.load_state_dict(torch.load('../weights/yolo.pth'))
    multiloss = YoloLoss(cfg)
    img = torch.randn((1, 3, 416, 416))
    img = img.cuda()
    targets = [torch.Tensor([[0.1, 0.1, 0.5, 0.3, 6], [0.2, 0.4, 0.4, 0.6, 5]]).cuda()]
    out = net(img)
    loss = multiloss(out, targets)
    print(loss)
