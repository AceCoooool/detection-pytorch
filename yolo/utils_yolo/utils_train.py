import torch
from utils.utils_box import jaccard, point_form


def build_mask(box_pred, target, anchors, f_size, coord_mask, conf_mask, pos_mask, m_boxes, idx, scale, warm):
    box_pred = point_form(box_pred.view(-1, 4))
    overlap = jaccard(box_pred, target[:, :4])
    best_truth_overlap, best_truth_idx = overlap.max(1)
    # TODO: this is 0.6 in original paper
    conf_mask[idx][(best_truth_overlap > 0.5).view_as(conf_mask[idx])] = 0
    if warm:
        coord_mask[idx].fill_(1)
        m_boxes[idx, ..., 0:2] = 0.5
    t_xy = (target[:, :2] + target[:, 2:4]) * f_size / 2
    t_wh = (target[:, 2:4] - target[:, :2]) * f_size
    xy = torch.floor(t_xy).long()
    pos = xy[:, 1] * f_size + xy[:, 0]
    wh = t_wh / 2
    target_box = torch.cat((-wh, wh), dim=1)
    wh = anchors / 2
    anchor_box = torch.cat((-wh, wh), dim=1)
    overlap = jaccard(target_box, anchor_box)
    best_prior_overlap, best_prior_idx = overlap.max(1)
    coord_mask[idx, pos, best_prior_idx] = 1
    pos_mask[idx, pos, best_prior_idx] = 1
    conf_mask[idx, pos, best_prior_idx] = scale
    m_boxes[idx, pos, best_prior_idx] = torch.cat(
        (t_xy - xy.float(), torch.log(t_wh / anchors[best_prior_idx, :]), target[:, 4:5]), dim=1)
