import torch

from yolo.utils_yolo.box_utils import filter_box
from utils.utils_box import point_form
from utils.utils_box import nms


class Detect(object):
    def __init__(self, cfg, eval=False, top_k=200):
        self.class_num = cfg.class_num
        self.feat_size = cfg.feat_size
        self.top_k = top_k
        if eval:
            self.nms_t, self.score_t = cfg.eval_nms_threshold, cfg.eval_score_threshold
        else:
            self.nms_t, self.score_t = cfg.nms_threshold, cfg.score_threshold

    def __call__(self, loc, conf, prob):
        num = loc.size(0)
        loc = point_form(loc)
        output = torch.zeros(num, self.class_num, self.top_k, 5)
        for i in range(num):
            loc_t, score_t, label_t = filter_box(loc[i], conf[i], prob[i], self.score_t)
            for c in range(self.class_num):
                idx = label_t == c
                if idx.sum() == 0:
                    continue
                c_loc = loc_t[idx]
                c_score = score_t[idx]
                ids, count = nms(c_loc, c_score, self.nms_t, self.top_k)
                output[i, c, :count] = torch.cat((c_score[ids[:count]].unsqueeze(1), c_loc[ids[:count]]), 1)
        # sort the score --- note: flt shares same "memory" as output
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank > self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
