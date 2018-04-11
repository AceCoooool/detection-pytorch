import torch
from utils.utils_box import nms
from ssd.config import v2
from ssd.utils_ssd.box_utils import decode


class Detect(object):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh, cfg=v2):
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def __call__(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors, self.num_classes)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i]
            # each class:
            # step1---delete score<conf_thresh
            # step2---non-maximum suppression
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[:, cl].gt(self.conf_thresh)
                scores = conf_scores[:, cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        # sort the score --- note: flt shares same "memory" as output
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank > self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
