import io
import os
import argparse
import configparser
import importlib
import torch
import numpy as np
from collections import defaultdict
from yolo.yolov2 import build_yolo


# exchange the [xxx] name to [xxx_num] form
def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def weight2pth(config_path, weights_path, output_path):
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)
    # weights header
    weights_file = open(weights_path, 'rb')
    weights_header = np.ndarray(shape=(4,), dtype='int32', buffer=weights_file.read(16))
    print('Weights Header: ', weights_header)
    # convert config information
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    # network information
    cfgname = config_path.split('/')[-1].split('.')[0]
    cfg = importlib.import_module('config.' + cfgname)
    net = build_yolo(cfg)
    net_dict = net.state_dict()
    keys = list(net_dict.keys())
    key_num, count, prev_filter = 0, 0, 3
    print('loading the weights ...')
    for section in cfg_parser.sections():
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            bn = 'batch_normalize' in cfg_parser[section]
            activation = cfg_parser[section]['activation']
            # three special case
            if section == 'convolutional_20':
                prev_filter = 512
            elif section == 'convolutional_21':
                prev_filter = 1280
            elif section == 'convolutional_0':
                prev_filter = 3
            else:
                prev_filter = weights_shape[0]

            weights_shape = (filters, prev_filter, size, size)
            weights_size = np.product(weights_shape)
            print('conv2d', 'bn' if bn else '  ', activation, weights_shape)
            conv_bias = np.ndarray(
                shape=(filters,),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters
            if bn:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters
                net_dict[keys[key_num + 1]].copy_(torch.from_numpy(bn_weights[0]))
                net_dict[keys[key_num + 2]].copy_(torch.from_numpy(conv_bias))
                net_dict[keys[key_num + 3]].copy_(torch.from_numpy(bn_weights[1]))
                net_dict[keys[key_num + 4]].copy_(torch.from_numpy(bn_weights[2]))
            else:
                net_dict[keys[key_num + 1]].copy_(torch.from_numpy(conv_bias))
            # conv parameter
            conv_weights = np.ndarray(
                shape=weights_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size
            net_dict[keys[key_num]].copy_(torch.from_numpy(conv_weights))
            key_num = key_num + 5 if bn else key_num + 1
        else:
            continue
    # check the convert
    remaining_weights = len(weights_file.read()) // 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))
    # save the net.state_dict
    torch.save(net_dict, os.path.join(output_path, cfgname + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    curdir = os.path.abspath('..')
    cfg_path = os.path.join(curdir, 'config/yolo.cfg')
    weight_path = os.path.join(curdir, 'config/yolo.weights')
    output_path = os.path.join(curdir, 'model')
    # parameters
    parser.add_argument('--cfg_path', default=cfg_path, type=str)
    parser.add_argument('--weight_path', default=weight_path, type=str)
    parser.add_argument('--output_path', default=output_path, type=str)

    config = parser.parse_args()
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    weight2pth(config.cfg_path, config.weight_path, config.output_path)
