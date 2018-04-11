import colorsys
import random
from dataset import config as cfg

# create model information
tiny, voc, num = False, True, 18
flag = [1, 1, 0] * 3 + [1, 0, 1] * 2 + [0, 1, 0]
size_flag = [3, 6, 9, 11, 14, 16]
pool = [0, 1, 4, 7, 13]

# anchor and classes information
anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']
anchor_num = 5
class_num = len(classes)

# color for draw boxes
hsv_tuples = [(x / class_num, 1., 1.) for x in range(class_num)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

voc_root = cfg.voc_root

# input image information
image_size = (416, 416)
feat_size = image_size[0] // 32

multi_scale_img_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
multi_len = len(multi_scale_img_size) - 1
multi_epoch = 2

# -----training set----
momentum = 0.9
batch_size = 1
# Note: the loss without avg, so we need to divide the bath_size
lr = 1e-3 / batch_size
weight_decay = 1e-4
gamma = 0.1
num_workers = 4
warm_epoch = 3
warm_lr = 1e-4 / batch_size
epoch_num = 160
stepvalues = (60, 90)
cuda = True
resume = False
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
use_office = True
trained_model = '../weights/yolo/yolo-voc.pth'
output_folder = '../results/yolo'

# -----eval set-----
eval_score_threshold = 1e-3
eval_nms_threshold = 0.3

# demo parameter
score_threshold = 0.5
nms_threshold = 0.3
iou_threshold = 0.6
