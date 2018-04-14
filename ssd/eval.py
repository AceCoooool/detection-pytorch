import sys

sys.path.append('..')
import torch
import os.path
import numpy as np
import pickle

import ssd.config as cfg
from ssd.ssd300 import build_ssd
from dataset.voc0712_cv import VOCDetection, BaseTransform, AnnotationTransform
from utils.extras import Timer, get_output_dir
from utils.utils_eval import write_voc_results_file, do_python_eval


# ----save the pred boxes+info to .pkl-----
# all detections are collected into:
# all_boxes[cls][image] = N x 5 array of detections in
# (x1, y1, x2, y2, score)
def generate_boxes(dataset, net, det_file):
    img_num = len(dataset)
    # TODO: delete
    # img_num = 50
    all_boxes = [[[] for _ in range(img_num)] for _ in range(cfg.num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}

    x = torch.randn((1, 3, 300, 300))
    x = x.cuda() if cfg.test_cuda else x
    for i in range(img_num):
        im, gt, h, w = dataset.pull_item(i)
        x.copy_(im.unsqueeze(0))
        _t['im_detect'].tic()
        with torch.no_grad():
            y = net(x)
        detect_time = _t['im_detect'].toc(avg=False)
        # "store" to each class
        for j in range(1, y.size(1)):
            dets = y[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0::2] *= w
            boxes[:, 1::2] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.c_[boxes.cpu().numpy(), scores]
            all_boxes[j][i] = cls_dets
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, img_num, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    return all_boxes


if __name__ == '__main__':
    # load net
    net = build_ssd('test', bone='vgg')
    net.load_state_dict(torch.load(cfg.trained_model))
    net.eval()
    if cfg.test_cuda:
        net = net.cuda()
    # load dataset
    dataset = VOCDetection(cfg.voc_root, [('2007', 'test')], BaseTransform(), AnnotationTransform())
    # save file
    output_dir = get_output_dir(cfg.output_folder, 'eval')
    det_file = os.path.join(output_dir, 'detections.pkl')
    # generate boxes
    print('predict boxes, and save to .pkl')
    box_list = generate_boxes(dataset, net, det_file)
    print('Evaluating detections')
    write_voc_results_file(box_list, dataset)
    do_python_eval(get_output_dir(cfg.save_folder, 'eval'))
