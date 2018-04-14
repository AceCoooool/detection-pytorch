from dataset.config import VOC_CLASSES as classes
from dataset.config import voc_root

bone = ['vgg', 'res101'][False]  # two base network you can choose
# Note: you can change by yourself
# ----training setting----
voc_root = voc_root
resume = False
batch_size = 32  # 32
if bone == 'vgg':
    basenet = 'vgg_rfc.pth'
else:
    basenet = 'res_rfc.pth'
num_workers = 4
epoch_num = 230
cuda = True
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
log_iters = True
visdom = False  # TODO: unfinished
send_images_to_visdom = False  # TODO: unfinished
save_folder = '../weights/ssd/'  # Location to save checkpoint models

# ----test setting----
test_cuda = True
if bone == 'vgg':
    trained_model = '../weights/ssd/vgg_final.pth'
else:
    trained_model = '../weights/ssd/resnet_final.pth'

output_folder = '../results/ssd'
top_k = 200
conf_thresh = 0.01
nms_thresh = 0.45

# ----pre setting----
overlap_thresh = 0.5
neg_pos = 3
variance = [0.1, 0.2]
stepvalues = (150, 190)
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
ssd_dim = 300
means = (104, 117, 123)  # bgr form
num_classes = len(classes) + 1


# SSD 300 config
v2 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'v2',
}
