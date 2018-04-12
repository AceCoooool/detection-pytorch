import torch
import os.path
import numpy as np
import importlib

import yolo.config.yolo_voc as cfg
from yolo.yolov2 import build_yolo
from utils.extras import Timer, get_output_dir
from utils.utils_eval import voc_eval

use_cv2 = importlib.util.find_spec('cv2') is not None
# use_cv2 = False
if use_cv2:
    from dataset.voc0712_cv import VOCDetection, BaseTransform, AnnotationTransform
else:
    from dataset.voc0712_pil import VOCDetection, BaseTransform, AnnotationTransform


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
        idx = dataset.ids[i][1]
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


# evaluate map
def test_map(files):
    annopath = os.path.join(cfg.voc_root, 'VOC2007', 'Annotations', '%s.xml')
    imagesetfile = os.path.join(cfg.voc_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
    cachedir = os.path.join(cfg.voc_root, 'annotations_cache')
    aps = []
    for i, cls in enumerate(cfg.classes):
        if cls == '__background__':
            continue
        filename = files[i]
        rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir,
                                 ovthresh=0.5, use_07_metric=False)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    mAP = np.mean(aps).item()
    return mAP


if __name__ == '__main__':
    output_dir = get_output_dir(cfg.output_folder, 'eval')
    filename = output_dir + '/all.txt'
    files = [output_dir + '/' + name + '.txt' for name in cfg.classes]
    if not os.path.isfile(filename):
        pred_file(filename, files, cfg.use_office)
    map = test_map(files)
