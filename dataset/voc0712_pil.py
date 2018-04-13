"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
from dataset.augment_pil import ToNumpy
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils import data
import xml.etree.ElementTree as ET  # parse xml file
import dataset.config as cfg


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (list with tuple-string): imageset to use (eg. [('2007', 'train')])
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_set:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, item):
        img, gt, h, w = self.pull_item(item)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id)
        width, height = img.size

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.c_[boxes, np.expand_dims(labels, axis=1)]

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    # Note: back PIL.Image form
    def pull_image(self, index):
        img_id = self.ids[index]
        return Image.open(self._imgpath % img_id)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)  # back original size
        return img_id[1], gt


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False, cfg=cfg):
        self.class_to_ind = class_to_ind or dict(zip(cfg.VOC_CLASSES, range(len(cfg.VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # each elem: [xmin, ymin, xmax, ymax, label_ind]
        return res


# basic transform: norm+scale
class BaseTransform(object):
    # image: PIL.Image form (output image is np.array)
    def __init__(self, size=300, mean=(123, 117, 104), scale=False):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = scale

    def __call__(self, image, boxes=None, labels=None):
        image = image.resize((self.size, self.size))
        image, _, _ = ToNumpy()(image)
        image -= self.mean
        image = image / 255.0 if self.scale else image
        return image, boxes, labels


# image: PIL.Image, label: string (class+score), boxe: tuple, c: int
def draw_box(image, label, box, c):
    w, h = image.size
    thickness = (w + h) // 300
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label)
    left, top, right, bottom = box
    top, left = max(0, np.round(top).astype('int32')), max(0, np.round(left).astype('int32'))
    right, bottom = min(w, np.round(right).astype('int32')), min(h, np.round(bottom).astype('int32'))
    print(label, (left, top), (right, bottom))
    text_orign = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=cfg.colors[c])
    draw.rectangle([tuple(text_orign), tuple(text_orign + label_size)], fill=cfg.colors[c])
    draw.text(text_orign, label, fill=(0, 0, 0))
    del draw


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    from dataset.augment_pil import Augmentation

    root = '/home/ace/data/VOCdevkit'
    image_set = [('2007', 'trainval'), ('2012', 'trainval')]
    target_transform = AnnotationTransform()
    dataset = VOCDetection(root, image_set, transform=Augmentation(), target_transform=target_transform)
    img, gt = dataset[0]
    print(img)
    print(gt)
