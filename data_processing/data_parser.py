from os.path import isfile, join
from os import listdir
from numpy.lib.shape_base import dstack
from tqdm import tqdm
from utils import *

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2

HEIGHT = 512
WIDTH = 512
DEPTH = 3

N_CLASSES=9

class DataWriter():
    def __init__(self, src_path, dest_path):
        self.src_path = src_path
        self.dest_path = dest_path
        self.filenames = [f for f in listdir(
            src_path) if isfile(join(src_path, f))]

    @staticmethod
    def parse_single_image(image, label):
        h, w, d = image.shape
        # define the dictionary -- the structure -- of our single example
        data = {
            'image': _bytes_feature(serialize_array(image)),
            'label': _bytes_feature(serialize_array(label))
        }
        # create an Example, wrapping the single features
        out = tf.train.Example(features=tf.train.Features(feature=data))

        return out

    def write_image_to_tfr(self, image, label, filename):
    
        filename = filename+".tfrecords"
        # create a writer that'll store our data to disk
        writer = tf.io.TFRecordWriter(filename)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        out = self.parse_single_image(image=image_rgb, label=label)
        writer.write(out.SerializeToString())

        writer.close()
        print(f"Wrote {filename} elements to TFRecord")

    def write_tfrecords(self):
        for file in tqdm(self.filenames):
            data = np.load(self.src_path + file)
            image, label = data['image'], data['label']
            filename = file[:-3] + "tfrecords"
            self.write_image_to_tfr(image, label, filename)

