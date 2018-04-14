import sys

sys.path.append('..')
import os.path
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import importlib
from ssd import config as cfg
from ssd.ssd300 import build_ssd
from ssd.utils_ssd.multiloss import MultiBoxLoss
from utils.utils_train import weights_init, adjust_learning_rate

use_cv2 = importlib.util.find_spec('cv2') is not None
if use_cv2:
    from dataset.voc0712_cv import VOCDetection, AnnotationTransform, detection_collate
    from dataset.augment_cv import Augmentation
else:
    from dataset.voc0712_pil import VOCDetection, AnnotationTransform, detection_collate
    from dataset.augment_pil import Augmentation

# TODO: add argparse form

if not os.path.exists(cfg.save_folder):
    os.mkdir(cfg.save_folder)

net = build_ssd('train')
if cfg.resume:
    print('Resuming training, loading {}...'.format(cfg.resume))
    net.load_state_dict(torch.load(cfg.resume))
else:
    bone_weights = torch.load(cfg.save_folder + cfg.basenet)
    print('Loading base network...')
    net.bone.load_state_dict(bone_weights)

if not cfg.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=cfg.lr,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)

criterion = MultiBoxLoss(cfg)

net.train()
loc_loss, conf_loss = 0, 0
epoch = 0
step_index = 0
print('Loading Dataset...')

dataset = VOCDetection(cfg.voc_root, cfg.train_sets, Augmentation(), AnnotationTransform())
data_loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_workers,
                         shuffle=True, collate_fn=detection_collate, pin_memory=True)
epoch_size = len(dataset) // cfg.batch_size
print("Training SSD on VOC")

batch_iterator = None
images = torch.randn((cfg.batch_size, 3, 300, 300), requires_grad=True)

if cfg.cuda:
    net = net.cuda()
    images = images.cuda()

for epoch in range(cfg.epoch_num):
    if epoch in cfg.stepvalues:
        step_index += 1
        adjust_learning_rate(optimizer, cfg.lr, cfg.gamma, step_index)
    for i, (imgs, targets) in enumerate(data_loader):
        if i == epoch_size:
            break
        images.data.copy_(imgs)
        targets = [anno.cuda() for anno in targets] if cfg.cuda else [anno for anno in targets]
        t0 = time.time()
        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_c + loss_l
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if i % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('epoch ' + repr(epoch) + ' iter ' + repr(i) + ' || Loss: %.4f || ' % (loss.item()), end=' ')
    if epoch % 20 == 0 and epoch != 0:
        print('Saving state, epoch: ', epoch)
        torch.save(net.state_dict(), '../weights/ssd/ssd300_' + repr(epoch) + '.pth')
torch.save(net.state_dict(), '../weights/ssd/final.pth')
