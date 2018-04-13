import os.path
import time
import importlib
import torch
from torch import optim
from xml.etree import ElementTree as ET

from yolo.config import yolo_voc as cfg
from yolo.yolov2 import build_yolo
from yolo.utils_yolo.yolo_loss import YoloLoss
from utils.utils_train import weights_init

use_cv2 = importlib.util.find_spec('cv2') is not None
if use_cv2:
    import cv2
    from dataset.voc0712_cv import BaseTransform, AnnotationTransform
else:
    from PIL import Image
    from dataset.voc0712_pil import BaseTransform, AnnotationTransform

if not os.path.exists(cfg.save_folder):
    os.mkdir(cfg.save_folder)

net = build_yolo('train')

if cfg.resume:
    print('Resuming training, loading {}...'.format(cfg.resume))
    net.load_state_dict(torch.load(cfg.resume))
else:
    darknet_weights = torch.load(cfg.save_folder + cfg.basenet)
    print('Loading base network...')
    net.darknet.load_state_dict(darknet_weights)

if not cfg.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    net.conv.apply(weights_init)
    net.conv1.apply(weights_init)
    net.conv2.apply(weights_init)

# net.load_state_dict(torch.load(cfg.trained_model))

optimizer = optim.SGD(net.parameters(), lr=cfg.lr,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)

criterion = YoloLoss(cfg)

net = net.cuda() if cfg.cuda else net
net.train()
loc_loss, conf_loss, prob_loss = 0, 0, 0
epoch = 0
step_index = 0

img_path = '../dataset/000004.jpg'
anno_path = '../dataset/000004.xml'

print("Training YOLO on Single Image")

if use_cv2:
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    img, _, _ = BaseTransform(416, mean=(0, 0, 0), scale=True)(image)
    img = img[:, :, (2, 1, 0)]
else:
    image = Image.open(img_path)
    w, h = image.size
    img, _, _ = BaseTransform(416, mean=(0, 0, 0), scale=True)(image)

anno = ET.parse(anno_path).getroot()
x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
target = AnnotationTransform()(anno, w, h)
target = [torch.Tensor(target)]

x.requires_grad = True
if cfg.cuda:
    x = x.cuda()
    target = [t.cuda() for t in target]

for i in range(1000):
    t0 = time.time()
    out = net(x)
    optimizer.zero_grad()
    conf_loss, class_loss, loc_loss = criterion(out, target)
    loss = conf_loss + class_loss + loc_loss
    loss.backward()
    optimizer.step()
    t1 = time.time()
    if i % 10 == 0:
        print('Timer: %.4f sec.' % (t1 - t0))
        print('epoch ' + repr(epoch) + ', iter ' + repr(i) + ' || loss: %.4f || ' % (loss.item())
              + ' || conf_loss: %.4f || ' % (conf_loss.item()) + ' || class_loss: %.4f || ' % (class_loss.item())
              + ' || loc_loss: %.4f || ' % (loc_loss.item()), end=' ')
torch.save(net.state_dict(), '../weights/yolo/single.pth')
