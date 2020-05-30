import os
import sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from network import MyNetwork

from parse import parse
from tqdm import trange
import pdb
import itertools

from tf_utils import pre_x_in, topk
from ops import tf_skew_symmetric
from tests import test_process
from config import get_config, print_usage
config, unparsed = get_config()

from read_write_model import read_model, qvec2rotmat
from read_dense import read_array

# root = 'Datasets/Phototourism'
# seq = 'brandenburg_gate'
# src = root + '/' + seq
# print(f'Done')
#
# cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')
# # print(cameras[294])
# print(images[294].xys)
# print(images[295].xys)

mynet = MyNetwork(config)
mynet.restore()
mynet.test_imw(x_in)

# self.x_in: xs_b,  # (?, 1, ?, 4)
# self.y_in: ys_b,  # (?, ?, 2)
# self.R_in: Rs_b,  # (?, 9)
# self.t_in: ts_b,  # (?, 3)

