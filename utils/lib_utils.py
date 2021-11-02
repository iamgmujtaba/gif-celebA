from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import json as simplejson

import time
import os
import shutil

####################################################################
####################################################################

def print_config(config):
    print('#'*60)
    print('Training configuration:')
    for k,v  in vars(config).items():
        print('  {:>20} {}'.format(k, v))
    print('#'*60)

def write_config(config, json_path):
    with open(json_path, 'w') as f:
        f.write(simplejson.dumps(vars(config), indent=4, sort_keys=True))

def output_subdir(config):
    prefix = time.strftime("%Y_%m_%d_%H%M")
    subdir = "{}_{}_{}".format(prefix, config.dataset, config.model)
    return os.path.join(config.save_dir, subdir)

def prepare_output_dirs(config):
    # Set output directories
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')
    config.log_dir = os.path.join(config.save_dir, 'logs')

    # And create them
    if os.path.exists(config.save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.save_dir)

    os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    os.mkdir(config.log_dir)
    return config

####################################################################
####################################################################
#Process an image that we can pass to our networks.
def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x
