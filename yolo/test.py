import importlib
import torch
from dataset.config import VOC_CLASSES as labelmap
from utils.extras import get_output_dir
from yolo.config import yolo_voc as cfg
from yolo.yolov2 import build_yolo

use_cv2 = importlib.util.find_spec('cv2') is not None
# Note: PIL.Image is better for official weights (I guess due to cv2 use BGR)
# use_cv2 = False
if use_cv2:
    from dataset.voc0712_cv import VOCDetection, AnnotationTransform, BaseTransform
else:
    from dataset.voc0712_pil import VOCDetection, AnnotationTransform, BaseTransform


def test(net, testset, filename, transform):
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
        for i in range(0, y.size(1)):
            j = 0
            while y[0, i, j, 0] >= cfg.score_threshold:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('Predictions: ' + '\n')
                score = y[0, i, j, 0].item()
                label_name = labelmap[i]
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
    open(filename, 'w').close()   # clean the txt file
    # load network
    net = build_yolo('test')
    net.load_state_dict(torch.load(cfg.trained_model))
    net.eval()
    print('Finished loading model !')
    if cfg.test_cuda:
        net = net.cuda()
    # load dataset
    testset = VOCDetection(cfg.voc_root, [('2007', 'test')], None, AnnotationTransform())
    if cfg.use_office:
        mean = (0, 0, 0)
    else:
        mean = (104, 117, 123) if use_cv2 else (123, 117, 104)
    transform = BaseTransform(size=416, mean=mean, scale=True)
    test(net, testset, filename, transform)
