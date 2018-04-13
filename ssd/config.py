import os.path
from dataset.config import VOC_CLASSES as classes

home = os.path.expanduser("~")

# ----training setting----
resume = False  # Resume from checkpoint
batch_size = 32  # Batch size for training: 16
basenet = 'vgg_rfc.pth'  # pretrained base model
num_workers = 4  # Number of workers used in dataloading
epoch_num = 230  # Number of training iterations: 120000
cuda = True  # Use cuda to train model
lr = 1e-3  # initial learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # Weight decay for SGD
gamma = 0.1  # Gamma update for SGD
log_iters = True  # Print the loss at each iteration
visdom = False  # Use visdom to for loss visualization
send_images_to_visdom = False  # Sample a random image from each 10th batch, send it to visdom after augmentations step
save_folder = '../weights/ssd/'  # Location to save checkpoint models
voc_root = os.path.join(home, "data/VOCdevkit/")

# ----test setting----
bone = 'vgg'
test_cuda = True
if bone == 'vgg':
    trained_model = '../weights/ssd/vgg_final.pth'
else:
    trained_model = '../weights/ssd/resnet_final.pth'
output_folder = '../results/ssd'
top_k = 200
conf_thresh = 0.01
nms_thresh = 0.45

# ----eval setting----


# ----pre setting----
overlap_thresh = 0.5
neg_pos = 3
variance = [0.1, 0.2]
stepvalues = (150, 190)
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
ssd_dim = 300
means = (104, 117, 123)  # bgr form
num_classes = len(classes) + 1

# ----box information----
extras_vgg = {'300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]}
extras_res = {'300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]}
l_vgg = [23, 512]

mbox_vgg = {'300': [(512, 4), (1024, 6), (512, 6), (256, 6), (256, 4), (256, 4)]}
mbox_res = {'300': [(512, 4), (2048, 6), (512, 6), (256, 6), (256, 4), (256, 4)]}
l_res = [11, 512]

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
