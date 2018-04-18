import sys

sys.path.append('..')
import os.path
import time
import importlib
import torch
from random import randint
from torch import optim
from torch.utils.data import DataLoader

from yolo.config import yolo_voc as cfg
from yolo.yolov2 import build_yolo
from yolo.utils_yolo.yolo_loss import YoloLoss
from utils.utils_train import weights_init, adjust_learning_rate

use_cv2 = importlib.util.find_spec('cv2') is not None
if use_cv2:
    from dataset.voc0712_cv import VOCDetection, AnnotationTransform, detection_collate
    from dataset.augment_cv import Augmentation
else:
    from dataset.voc0712_pil import VOCDetection, AnnotationTransform, detection_collate
    from dataset.augment_pil import Augmentation

if not os.path.exists(cfg.save_folder):
    os.mkdir(cfg.save_folder)

net = build_yolo('train')

if cfg.resume:
    print('Resuming training, loading {}...'.format(cfg.resume))
    net.load_state_dict(torch.load(cfg.resume_weights))
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

step_index = 0
warm = True if not cfg.resume else False

optimizer = optim.SGD(net.parameters(), lr=cfg.warm_lr if warm else cfg.lr,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)
if cfg.resume > 120:
    step_index = 2
    adjust_learning_rate(optimizer, cfg.lr, cfg.gamma, step_index)
elif cfg.resume > 80:
    step_index = 1
    adjust_learning_rate(optimizer, cfg.lr, cfg.gamma, step_index)

dataset = VOCDetection(cfg.voc_root, cfg.train_sets,
                       Augmentation(size=cfg.image_size[0], mean=(0, 0, 0), scale=True), AnnotationTransform())
data_loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_workers,
                         shuffle=True, collate_fn=detection_collate, pin_memory=True)
epoch_size = len(dataset) // cfg.batch_size
images = torch.randn((cfg.batch_size, 3, cfg.image_size[0], cfg.image_size[0]), requires_grad=True)
images = images.cuda() if cfg.cuda else images

criterion = YoloLoss(cfg)

net = net.cuda() if cfg.cuda else net
net.train()
loc_loss, conf_loss, prob_loss = 0, 0, 0
epoch = 0

print("Training YOLO on VOC")

if warm:
    print('In the warm up phase ... ^_^ ')
for epoch in range(cfg.resume, cfg.epoch_num):
    if epoch == cfg.warm_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.lr
        warm = False
        print('Finish warm up phase ... ^_^ ')
    if epoch in cfg.stepvalues:
        step_index += 1
        adjust_learning_rate(optimizer, cfg.lr, cfg.gamma, step_index)
        loc_loss, conf_loss = 0, 0
    # multi-scale training
    if cfg.use_multi and (epoch + 1) % cfg.multi_epoch == 0:
        idx = randint(0, cfg.multi_len)
        img_size = cfg.multi_scale_img_size[idx]
        print('Using scale of %.d !!!' % img_size)
        # Note: test the largest size before training --- avoid out of memory
        dataset = VOCDetection(cfg.voc_root, cfg.train_sets,
                               Augmentation(size=img_size, mean=(0, 0, 0), scale=True), AnnotationTransform())
        data_loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_workers,
                                 shuffle=True, collate_fn=detection_collate, pin_memory=True)
        images.resize_(cfg.batch_size, 3, img_size, img_size)
    for i, (imgs, targets) in enumerate(data_loader):
        if i == epoch_size:
            break
        images.data.copy_(imgs)
        targets = [anno.cuda() for anno in targets] if cfg.cuda else [anno for anno in targets]
        t0 = time.time()
        out = net(images)
        optimizer.zero_grad()
        conf_loss, class_loss, loc_loss = criterion(out, targets, warm)
        loss = conf_loss + class_loss + loc_loss
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if i % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('epoch ' + repr(epoch) + ', iter ' + repr(i) + ' || loss: %.4f || ' % (loss.item())
                  + ' || conf_loss: %.4f || ' % (conf_loss.item()) + ' || class_loss: %.4f || ' % (class_loss.item())
                  + ' || loc_loss: %.4f || ' % (loc_loss.item()), end=' ')
    if epoch % 20 == 0 and epoch != 0:
        print('Saving state, epoch: ', epoch)
        torch.save(net.state_dict(), '../weights/yolo/yolo_' + repr(epoch) + '.pth')
torch.save(net.state_dict(), '../weights/yolo/final.pth')
