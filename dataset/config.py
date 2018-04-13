import os
import colorsys
import random

home = os.path.expanduser("~")
voc_root = os.path.join(home, "data/VOCdevkit/")
annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = voc_root + 'VOC' + YEAR

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class_num = len(VOC_CLASSES)

# color for draw boxes
hsv_tuples = [(x / class_num, 1., 1.) for x in range(class_num)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.
