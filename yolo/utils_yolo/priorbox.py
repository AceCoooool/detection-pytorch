import torch
import numpy as np


# prior box in each position
class PriorBox(object):
    def __init__(self, anchors, feat_size=13):
        super(PriorBox, self).__init__()
        self.feat_size = feat_size
        self.anchors = anchors
        self.anchor_num = anchors.shape[0]

    def __call__(self):
        x, y = np.meshgrid(np.arange(self.feat_size), np.arange(self.feat_size))
        x, y = x.repeat(self.anchor_num), y.repeat(self.anchor_num)
        xy = np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]
        wh = np.tile(self.anchors, (self.feat_size * self.feat_size, 1))
        output = torch.from_numpy(np.c_[xy, wh].astype(np.float32)).view(1, -1, self.anchor_num, 4)
        return output


if __name__ == '__main__':
    from yolo.config import yolo_voc as cfg
    anchors = np.array(cfg.anchors).reshape(-1, 2)
    prior = PriorBox(anchors, 14)
    box = prior()
    print(box)
