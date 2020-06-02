import os
import sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from network import MyNetwork
from PIL import Image
import h5py

from parse import parse
from tqdm import trange
import pdb
import itertools

from tf_utils import pre_x_in, topk
from ops import tf_skew_symmetric
from tests import test_process
from config import get_config, print_usage
config, unparsed = get_config()
from glob import glob

from read_write_model import read_model, qvec2rotmat
from read_dense import read_array

path = 'IMW/sacre_coeur/Images'
path_ks = 'IMW/sacre_coeur/keypoints.h5'

all_ks = h5py.File(path_ks)

# for k, v in all_ks.items():
#     print((k, v.shape))

# https://github.com/vcg-uvic/image-matching-benchmark/blob/master/example/training_data/parse_data.ipynb?fbclid=IwAR2prxKGOvm5mJPdjzH8XHEMR3oiE0IV9KgtshTK3lCyty-g3DjhFp9wGx8

paths = sorted(glob(os.path.join(path, '*')))
for i,p1 in enumerate(paths):
    for p2 in paths[i+1:]:
        img = Image.open(p1)
        name = os.path.splitext(p1)[0]
        w,h = img.width, img.height
        print(name)
        kps = all_ks.get(name)
        print(kps)
        kps[:,0] /= w
        kps[:,1] /= h
        print(kps)
        exit()

        img = Image.open(p2)
        name = os.path.splitext(p2)[0]
        w,h = img.width, img.height
        # cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')
        # print(cameras[294])
        # print(images[294].xys.shape)
        # print(images[295].xys)
        print(p)
        print(w,h)
        x_in = np.concatenate([img, img], axis=1)
        x_in = np.expand_dims(x_in, 0)
        x_in = np.expand_dims(x_in, 0)
        print(x_in)
        print(x_in.shape)

        mynet = MyNetwork(config)
        mynet.restore()
        mynet.test_imw(x_in)

        # self.x_in: xs_b,  # (?, 1, ?, 4)
        # self.y_in: ys_b,  # (?, ?, 2)
        # self.R_in: Rs_b,  # (?, 9)
        # self.t_in: ts_b,  # (?, 3)

        break