import torch
import time
import os.path
import numpy as np
from collections import OrderedDict


# a simple timer
class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, avg=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if avg:
            return self.average_time
        else:
            return self.diff


# return the directory where experimental artifact are places
def get_output_dir(root, name):
    filedir = os.path.join(root, name)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


# convert weight from "pre-trained model" (with different key name)
# new, older is the OrderedDict type
def weight_convert(new, older, root, l=None):
    newkeys = list(new.keys())
    if l is None:
        d2 = OrderedDict([(newkeys[i], v) for i, (k, v) in enumerate(older.items())])
    else:
        d2 = OrderedDict([(newkeys[i], v) for i, (k, v) in enumerate(older.items()) if i < l])
    torch.save(d2, root)


# convert from npz file (for darknet)
def load_from_npz(net, fname, num_conv=None):
    dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                'bn.weight': 'gamma', 'bn.bias': 'biases',
                'bn.running_mean': 'moving_mean',
                'bn.running_var': 'moving_variance'}
    params = np.load(fname)
    own_dict = net.state_dict()
    keys = list(own_dict.keys())

    for i, start in enumerate(range(0, len(keys), 5)):
        if num_conv is not None and i >= num_conv:
            break
        end = min(start + 5, len(keys))
        for key in keys[start:end]:
            list_key = key.split('.')
            ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
            src_key = '{}-convolutional/{}:0'.format(i, ptype)
            print((src_key, own_dict[key].size(), params[src_key].shape))
            param = torch.from_numpy(params[src_key])
            if ptype == 'kernel':
                param = param.permute(3, 2, 0, 1)
            own_dict[key].copy_(param)


if __name__ == '__main__':
    from torchvision.models import resnet101
    from ssd.ssd300 import build_ssd

    ssd = build_ssd('train', bone='res101')
    net = resnet101(pretrained=True)
    weight_convert(ssd.bone.state_dict(), net.state_dict(), '../weights/ssd/res_feat.pth', len(ssd.bone.state_dict()))
    # from yolo.yolov2 import build_yolo
    # from torch import nn
    #
    # net = build_yolo('train')
    # # print(net.state_dict().keys())
    # # load_from_npz(net, '../weights/darknet19.weights.npz')
    # # torch.save(net.state_dict(), '../weights/darknet.pth')
    # older = torch.load('../weights/yolo-voc.pth')
    # new = net.state_dict()
    # weight_convert(new, older, '../weights/yolo/yolo-voc.pth')
