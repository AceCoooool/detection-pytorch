import torch
from torch import nn
import torch.nn.functional as F

from ssd.utils_ssd.box_utils import match, log_sum_exp


# evaluate conf_loss and loc_loss
class MultiBoxLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = cfg.num_classes
        self.threshold = cfg.overlap_thresh
        self.negpos_ratio = cfg.neg_pos
        self.variance = cfg.variance

    def forward(self, preds, targets):
        loc_data, conf_data, priors = preds
        num = loc_data.size(0)
        num_priors = priors.size(0)
        # match priors (priors->nearest target)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if loc_data.is_cuda:
            loc_t, conf_t = loc_t.cuda(), conf_t.cuda()
        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        pos = conf_t > 0
        # location loss
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # evaluate each priors's loss (the same as the paper)
        batch_conf = conf_data
        loss_c = (log_sum_exp(batch_conf) - batch_conf.gather(2, conf_t.unsqueeze(2))).squeeze(2)
        # hard negative mining: note: the batch size of each iteration is not the same
        # find the "max loss" background
        loss_c[pos] = 0  # filter out pos boxes
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # size: [num, 1]
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # confidence loss (pos:neg=1:3)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weightd = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weightd, size_average=False)

        return loss_l / num_pos.sum(), loss_c / num_pos.sum()
