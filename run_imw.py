import os
import sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from parse import parse
from tqdm import trange
import pdb
import itertools

from tf_utils import pre_x_in, topk
from ops import tf_skew_symmetric
from tests import test_process

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_dense import read_array

root = 'Datasets/Phototourism'
seq = 'brandenburg_gate'
src = root + '/' + seq
print(f'Done')


cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')