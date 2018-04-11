import torch
import os.path
import numpy as np
import pickle
import importlib

import yolo.config.yolo_voc as cfg
from yolo.yolov2 import build_yolo
from utils.extras import Timer, get_output_dir
from utils.utils_eval import write_voc_results_file, do_python_eval

use_cv2 = importlib.util.find_spec('cv2') is not None
if use_cv2:
    from dataset.voc0712_cv import VOCDetection, BaseTransform, AnnotationTransform
else:
    from dataset.voc0712_pil import VOCDetection, BaseTransform, AnnotationTransform


# ----save the pred boxes+info to .pkl-----
# all detections are collected into:
# all_boxes[cls][image] = N x 5 array of detections in
# (x1, y1, x2, y2, score)
def generate_boxes(dataset, net, det_file):
    img_num = len(dataset)
    # TODO: delete
    # img_num = 50
    all_boxes = [[[] for _ in range(img_num)] for _ in range(cfg.class_num + 1)]

    _t = {'im_detect': Timer(), 'misc': Timer()}

    x = torch.randn((1, 3, 416, 416))
    x = x.cuda() if cfg.test_cuda else x
    for i in range(img_num):
        im, gt, h, w = dataset.pull_item(i)
        x.copy_(im.unsqueeze(0))
        _t['im_detect'].tic()
        with torch.no_grad():
            y = net(x)
        detect_time = _t['im_detect'].toc(avg=False)
        # "store" to each class
        for j in range(0, y.size(1)):
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
            all_boxes[j + 1][i] = cls_dets
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, img_num, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    return all_boxes


def pred_file(filename, files, office):
    # Note: delete all the exists files
    [os.remove(file) for file in files if os.path.isfile(file)]
    print('Loading trained network ...')
    net = build_yolo('test', cfg, eval=True)
    net.load_state_dict(torch.load(cfg.trained_model))
    net.eval()
    net = net.cuda() if cfg.test_cuda else net
    print('Loading VOC dataset ...')
    if office:
        mean = (0, 0, 0)
    else:
        mean = (104, 117, 123) if use_cv2 else (123, 117, 104)
    transform = BaseTransform(size=416, mean=mean, scale=True)
    dataset = VOCDetection(cfg.voc_root, [('2007', 'test')], transform, AnnotationTransform())
    num_images = len(dataset)
    _t = {'im_detect': Timer(), 'misc': Timer()}

    x = torch.randn((1, 3, 416, 416))
    x = x.cuda() if cfg.test_cuda else x
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        im, gt, h, w = dataset.pull_item(i)
        idx = dataset.ids[i]
        x.copy_(im.unsqueeze(0))
        _t['im_detect'].tic()
        with torch.no_grad():
            y = net(x)
        detect_time = _t['im_detect'].toc(avg=False)
        # TODO: unfinish
        for k in range(0, y.size(1)):
            dets = y[0, k, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0::2] *= w
            boxes[:, 1::2] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.c_[boxes.cpu().numpy(), scores]
            for j in range(cls_dets.shape[0]):
                with open(filename, mode='a') as f:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                            .format(idx, cls_dets[j, 4],
                                    cls_dets[j, 0], cls_dets[j, 1], cls_dets[j, 2], cls_dets[j, 3]))
                with open(files[k], mode='a') as f:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                            .format(idx, cls_dets[j, 4],
                                    cls_dets[j, 0], cls_dets[j, 1], cls_dets[j, 2], cls_dets[j, 3]))


if __name__ == '__main__':
    # load net
    net = build_yolo('test', cfg, eval=True)
    net.load_state_dict(torch.load(cfg.trained_model))
    net.eval()
    if cfg.test_cuda:
        net = net.cuda()
    # load dataset
    if cfg.use_office:
        mean = (0, 0, 0)
    else:
        mean = (104, 117, 123) if use_cv2 else (123, 117, 104)
    transform = BaseTransform(size=416, mean=mean, scale=True)
    dataset = VOCDetection(cfg.voc_root, [('2007', 'test')], transform, AnnotationTransform())
    # save file
    output_dir = get_output_dir(cfg.output_folder, 'eval')
    det_file = os.path.join(output_dir, 'detections.pkl')
    # generate boxes
    print('predict boxes, and save to .pkl')
    box_list = generate_boxes(dataset, net, det_file)
    print('Evaluating detections')
    write_voc_results_file(box_list, dataset)
    do_python_eval(get_output_dir(cfg.output_folder, 'eval'))
