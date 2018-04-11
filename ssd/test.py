import torch
from dataset.config import VOC_CLASSES as labelmap
from utils.extras import get_output_dir
import ssd.config as cfg
from ssd.ssd300 import build_ssd

import importlib

use_cv2 = importlib.util.find_spec('cv2') is not None
if use_cv2:
    from dataset.voc0712_cv import VOCDetection, AnnotationTransform, BaseTransform
else:
    from dataset.voc0712_pil import VOCDetection, AnnotationTransform, BaseTransform


# TODO: add argparse

def test(net, testset, filename, transform=BaseTransform()):
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}...'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x, _, _ = transform(img)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        with open(filename, mode='a') as f:
            f.write('\n ground truth for: ' + img_id + '\n')
            for box in annotation:
                f.write('label: ' + ' || '.join(str(b) for b in box[:-1]) + ' || ' + labelmap[box[-1]] + '\n')
        if cfg.test_cuda:
            x = x.cuda()

        with torch.no_grad():
            y = net(x)
        if use_cv2:
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])
        else:
            scale = torch.Tensor([img.size[0], img.size[1],
                                  img.size[0], img.size[1]])
        pred_num = 0
        for i in range(1, y.size(1)):
            j = 0
            while y[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('Predictions: ' + '\n')
                score = y[0, i, j, 0].item()
                label_name = labelmap[i - 1]
                pt = (y[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: ' + ' || '.join('{:4.1f}'.format(c) for c in coords)
                            + ' || ' + label_name + ' || ' + '{:4.2f}'.format(score) + '\n')
                j += 1


if __name__ == '__main__':
    file = get_output_dir(cfg.output_folder, 'test')
    filename = file + '/test.txt'
    # load network
    net = build_ssd('test', bone='vgg')
    net.load_state_dict(torch.load(cfg.trained_model))
    net.eval()
    print('Finished loading model !')
    if cfg.test_cuda:
        net = net.cuda()
    # load dataset
    testset = VOCDetection(cfg.voc_root, [('2007', 'test')], None, AnnotationTransform())
    test(net, testset, filename)
