import os
import sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from network import MyNetwork
from PIL import Image
import h5py
from tqdm import tqdm

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

# sequence = 'sacre_coeur'
sequence = 'st_peters_square'

path = f'IMW/{sequence}/Images'
all_ks = h5py.File(f'IMW/{sequence}/keypoints.h5')
matches = h5py.File(f'IMW/{sequence}/matches.h5')
h5out = h5py.File(f'IMW/{sequence}/matches-acne.h5', 'w')

# for k, v in matches.items():
#     print((k, v.shape))

# https://github.com/vcg-uvic/image-matching-benchmark/blob/master/example/training_data/parse_data.ipynb?fbclid=IwAR2prxKGOvm5mJPdjzH8XHEMR3oiE0IV9KgtshTK3lCyty-g3DjhFp9wGx8

def get_kps(p):
    img = Image.open(p)
    name = os.path.splitext(os.path.basename(p))[0]
    w,h = img.width, img.height
    kps = all_ks.get(name).value
    # print(kps)
    kps[:,0] /= w
    kps[:,1] /= h
    # print(kps)
    return kps

mynet = MyNetwork(config)
mynet.restore()

# topnum = 1000

# try out_weight >= 1e-5 or out_weight >=1e-7

paths = sorted(glob(os.path.join(path, '*')))
for i,p2 in tqdm(enumerate(paths), total=len(paths)):
    for p1 in paths[i+1:]:
        kps1 = get_kps(p1)
        kps2 = get_kps(p2)

        # cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')
        # print(cameras[294])
        # print(images[294].xys.shape)
        # print(images[295].xys)
        name =  os.path.splitext(os.path.basename(p1))[0] + '-' + os.path.splitext(os.path.basename(p2))[0]
        match = matches.get(name).value
        kps1 = kps1[match[0]]
        kps2 = kps2[match[1]]

        # m = min(kps1.shape[0], kps2.shape[0])
        # kps1 = kps1[:m]
        # kps2 = kps2[:m]
        x_in = np.concatenate([kps1, kps2], axis=1)
        x_in = np.expand_dims(x_in, 0)
        x_in = np.expand_dims(x_in, 0)

        res = mynet.test_imw(x_in)[0]
        # idxs = np.nonzero(res > 1e-7)[0]
        idxs = np.nonzero(res > 1e-5)[0]
        print(len(idxs))
        # idxs = np.argsort(res)[::-1][:topnum]
        kps = match[:,idxs]

        # self.x_in: xs_b,  # (?, 1, ?, 4)
        # self.y_in: ys_b,  # (?, ?, 2)
        # self.R_in: Rs_b,  # (?, 9)
        # self.t_in: ts_b,  # (?, 3)

        h5out.create_dataset(name, data=kps)

h5out.close()