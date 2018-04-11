import math
import torch
from torch import nn
import torch.nn.functional as F
from yolo.utils_yolo.utils_train import build_mask


class YoloLoss(nn.Module):
    def __init__(self, cfg):
        super(YoloLoss, self).__init__()
        self.anchors = torch.Tensor(cfg.anchors).reshape(-1, 2)
        self.num_anchors = self.anchors.size(0)
        self.nclass = cfg.class_num
        self.no_object_scale = cfg.no_object_scale
        self.object_scale = cfg.object_scale
        self.class_scale = cfg.class_scale
        self.coord_scale = cfg.coord_scale
        self.cuda_flag = True

    def forward(self, preds, targets, warm=False):
        feat, box_pred, box_conf, box_prob = preds
        num, f_size = feat.size(0), int(math.sqrt(feat.size(1)))
        box_match = torch.cat((F.sigmoid(feat[..., :2]), feat[..., 2:4]), -1)
        # TODO: simplify the mask
        coord_mask = torch.zeros_like(box_conf)
        conf_mask = torch.ones_like(box_conf)
        pos_mask = torch.zeros_like(box_conf)
        m_boxes = torch.zeros_like(box_conf).repeat(1, 1, 1, 5)
        if self.cuda_flag and feat.is_cuda:
            self.anchors = self.anchors.cuda()
        for idx in range(num):
            build_mask(box_pred[idx], targets[idx], self.anchors, f_size, coord_mask, conf_mask,
                       pos_mask, m_boxes, idx, self.object_scale, warm)
        loc_loss = F.mse_loss(coord_mask * box_match, coord_mask * m_boxes[..., :4],
                              size_average=False) / 2
        conf_loss = F.mse_loss(conf_mask * box_conf, conf_mask * pos_mask, size_average=False) / 2
        class_loss = F.cross_entropy(box_prob[pos_mask.byte().repeat(1, 1, 1, self.nclass)].view(-1, self.nclass),
                                     m_boxes[..., 4:5][pos_mask.byte()].view(-1).long(), size_average=False)
        return conf_loss, class_loss, loc_loss


if __name__ == '__main__':
    from yolo.config import yolo_voc as cfg
    from yolo.utils_yolo.priorbox import PriorBox

    priors = PriorBox(cfg)()
    loss = YoloLoss(cfg)
    feat = torch.ones((1, 169, 5, 25))
    box_xy, box_wh = F.sigmoid(feat[..., :2]), feat[..., 2:4].exp()
    box_xy += priors[..., 0:2]
    box_wh *= priors[..., 2:]
    box_conf, score_pred = F.sigmoid(feat[..., 4:5]), feat[..., 5:]
    box_pred = torch.cat([box_xy, box_wh], 3) / 13
    preds = (feat, box_pred, box_conf, score_pred)
    target = [torch.Tensor([[0.1, 0.2, 0.3, 0.4, 3], [0.3, 0.4, 0.5, 0.6, 3]])]
    res = loss(preds, target, 0)
    print(res)
