import gzip
import os
import urllib.request
import shutil

import tensorflow as tf


def make_dir(PATH):
    try:
        os.makedirs(PATH)
    except OSError:
        pass

def convert_to_dataset(data, batch_size):
    
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
