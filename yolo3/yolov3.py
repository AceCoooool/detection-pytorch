import torch
from torch import nn
from basenet.darknet53 import darknet_feat, ConvLayer
from yolo3.config import yolo3_coco as cfg

yolo_extras = [7, 7, 7]
yolo_channels = [[1024, 512], [768, 256], [384, 128]]
yolo_combine = [14, 23]


# three extra layers
def extras_layers(num_classes, _extras=yolo_extras, _channels=yolo_channels):
    layers = [[] for _ in range(len(_extras))]
    convs = [[] for _ in range(len(_extras) - 1)]
    for i, num in enumerate(_extras):
        inplane, plane, flag = _channels[i][0], _channels[i][1], False
        if i < len(_extras) - 1:
            convs[i] += [ConvLayer(plane, plane // 2, 1), nn.Upsample(scale_factor=2)]
        for k in range(num):
            if k == num - 1:
                plane = 3 * (num_classes + 5)
                layers[i].append(nn.Conv2d(inplane, plane, 1, 1))
            else:
                layers[i].append(ConvLayer(inplane, plane, [1, 3][flag]))
                inplane, plane = plane, int(plane * [2, 1 / 2][flag])
                plane = 3 * (num_classes + 5) if k == num - 2 else plane
            flag = not flag
    extras = list()
    for i in range(len(layers)):
        extras.append(layers[i])
        extras.append(convs[i]) if i < len(convs) else None
    return extras


# TODO: change to voc
class Yolo3(nn.Module):
    def __init__(self, phase, cfg, out=yolo_combine):
        super(Yolo3, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.bone = nn.ModuleList(darknet_feat())
        self.extras = extras_layers(cfg.num_classes)
        self.extras = nn.ModuleList([nn.ModuleList(ex) for ex in self.extras])
        self.convnum = range(1, len(self.extras), 2)
        self.out = out

    def forward(self, x):
        b_out, out = list(), list()
        for i in range(len(self.bone)):
            x = self.bone[i](x)
            b_out.append(x) if i in self.out else None
        n_out = len(self.out)
        for i in range(len(self.extras)):
            if i in self.convnum:
                for j in range(len(self.extras[i])):
                    temp = self.extras[i][j](temp)
                x = torch.cat((temp, b_out[n_out - 1]), 1)
                n_out = n_out - 1
            else:
                for j in range(len(self.extras[i])):
                    x = self.extras[i][j](x)
                    if j == 4: temp = x
                out.append(x)
        if self.phase == 'train':
            pass  # TODO: unfinish
        else:
            return out


def build_yolo3(phase, cfg=cfg):
    if phase != 'test' and phase != 'train':
        assert "Error: Phase not recognized"
    return Yolo3(phase, cfg)


if __name__ == '__main__':
    from utils.extras import weight_convert

    yolo = build_yolo3('test')
    yolo.eval()  # Note: BN influence it
    # print(yolo)
    # print(yolo.extras)
    yolo.load_state_dict(torch.load('../weights/yolo3/yolo3.pth'))
    # weights = torch.load('../weights/yolo3/yolov3-coco.pth')
    # print(yolo.state_dict().values())
    # print(weights.keys())
    # weight_convert(yolo.state_dict(), weights, '../weights/yolo3/yolo3.pth')
    # yolo.load_state_dict(torch.load('../weights/yolo3/yolo3.pth'))
    img = torch.ones((1, 3, 416, 416))
    out = yolo(img)
    print(out[2])
