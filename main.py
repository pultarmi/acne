#!/usr/bin/env python3
# Filename: main.py
# License: LICENSES/LICENSE_UVIC_EPFL

# from __future__ import print_function

from multiprocessing import set_start_method
set_start_method("spawn")
from multiprocessing import get_context

from config import get_config, print_usage
from data import load_data
from network import MyNetwork

eps = 1e-10
use3d = False
config = None

config, unparsed = get_config()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")

print('---------------')
print(tf.test.is_gpu_available())
print('---------------')

def main(config):
    # Run propper mode
    if config.run_mode == "train":
        data = {}
        data["train"] = load_data(config, "train")
        data["valid"] = load_data(config, "valid")
        mynet = MyNetwork(config)
        mynet.train(data)
    elif config.run_mode == "test":
        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        # data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")
        mynet = MyNetwork(config)
        mynet.test(data)


if __name__ == "__main__":
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    main(config)
