import torch


# remove the proposal detector which is less than the threshold
def filter_box(boxes, box_conf, box_prob, threshold=.5):
    box_scores = box_conf.repeat(1, 1, box_prob.size(2)) * box_prob
    box_class_scores, box_classes = torch.max(box_scores, dim=2)
    prediction_mask = box_class_scores > threshold
    prediction_mask4 = prediction_mask.unsqueeze(2).expand_as(boxes)

    boxes = torch.masked_select(boxes, prediction_mask4).contiguous().view(-1, 4)
    scores = torch.masked_select(box_class_scores, prediction_mask)
    classes = torch.masked_select(box_classes, prediction_mask)
    return boxes, scores, classes
