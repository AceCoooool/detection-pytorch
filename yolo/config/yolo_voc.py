from dataset import config as cfg

tiny = False  # for tiny-yolo
voc = True
voc_root = cfg.voc_root
# anchor and classes information
anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']
anchor_num = 5
class_num = len(classes)

# input image information
image_size = (416, 416)
feat_size = image_size[0] // 32

# multi-scale for training
multi_scale_img_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
multi_len = len(multi_scale_img_size) - 1

# -----training set----
use_multi = False
multi_epoch = 5
momentum = 0.9
batch_size = 32
# Note: the loss without avg, so we need to divide the bath_size
lr = 1e-3 / batch_size
weight_decay = 1e-4
gamma = 0.1
num_workers = 4
warm_epoch = 2
warm_lr = 1e-4 / batch_size
epoch_num = 160
stepvalues = (80, 120)
cuda = True
resume = 0
resume_weights = '../weights/yolo/yolo_100.pth'
save_folder = '../weights/yolo/'
basenet = 'darknet.pth'

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]

no_object_scale = 1
object_scale = 5
class_scale = 1
coord_scale = 1

# -----testing set-----
test_cuda = True
# Note: the office weights not minus means
# trained_model = '../weights/yolo/single.pth'
trained_model = '../weights/yolo/yolo_160.pth'
output_folder = '../results/yolo'

# -----eval set-----
eval_score_threshold = 1e-4
eval_nms_threshold = 0.3

# demo parameter
score_threshold = 0.5
nms_threshold = 0.3
iou_threshold = 0.6
