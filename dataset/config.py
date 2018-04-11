import os

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
