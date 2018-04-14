import sys

sys.path.append('..')
import numpy as np
import torch
import os
import importlib
from utils.extras import get_output_dir
from dataset.config import COCO_CLASSES as labelmap
from yolo3.config import yolo3_coco as cfg
from yolo3.yolov3 import build_yolo3

use_cv2 = importlib.util.find_spec('cv2') is not None
if use_cv2:
    import cv2
    from dataset.voc0712_cv import BaseTransform, draw_box
else:
    from PIL import Image
    from dataset.voc0712_pil import BaseTransform, draw_box


def demo(img_list, save_path=None):
    net = build_yolo3('test', cfg)
    net.load_state_dict(torch.load(cfg.trained_model))
    if cfg.test_cuda: net = net.cuda()
    net.eval()
    transform = BaseTransform(size=416, mean=(0, 0, 0), scale=True)
    for img in img_list:
        if use_cv2:
            image = cv2.imread(img)
            h, w, _ = image.shape
        else:
            image = Image.open(img)
            w, h = image.size
        scale = np.array([w, h, w, h])
        x, _, _ = transform(image)
        x = x[:, :, (2, 1, 0)] if use_cv2 else x
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        if cfg.test_cuda: x = x.cuda()
        with torch.no_grad():
            y = net(x)
        for i in range(y.size(1)):
            idx = (y[0, i, :, 0] > 0.5)
            dets = y[0, i][idx].view(-1, 5)
            if dets.numel() == 0:
                continue
            print('Find {} {} for {}.'.format(dets.size(0), labelmap[i], img.split('/')[-1]))
            score, loc = dets[:, 0], dets[:, 1:].cpu().numpy() * scale
            for k in range(len(score)):
                label = '{} {:.2f}'.format(labelmap[i], score[k])
                draw_box(image, label, loc[k], i)
        if use_cv2:
            cv2.imwrite(os.path.join(save_path, img.split('/')[-1]), image)
        else:
            image.save(os.path.join(save_path, img.split('/')[-1]), quality=90)


if __name__ == '__main__':
    img_list = ['../dataset/000004.jpg']
    file = get_output_dir(cfg.output_folder, 'demo')
    demo(img_list, file)
