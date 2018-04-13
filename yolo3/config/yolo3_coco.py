class_num = 80

anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

# test
use_office = True  # TODO: delete the use_office
test_cuda = False
trained_model = '../weights/yolo3/yolo3.pth'
output_folder = '../results/yolo3'

# -----eval set-----
eval_score_threshold = 1e-4
eval_nms_threshold = 0.3

# demo parameter
score_threshold = 0.5
nms_threshold = 0.3
iou_threshold = 0.6
