from os.path import isfile, join
from os import listdir
from numpy.lib.shape_base import dstack
from tqdm import tqdm

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
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value ist tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def serialize_array(array):
        array = tf.io.serialize_tensor(array)
        return array

    def parse_single_image(self, image, label):
        h, w, d = image.shape
        # define the dictionary -- the structure -- of our single example
        data = {
            'image': self._bytes_feature(self.serialize_array(image)),
            'label': self._bytes_feature(self.serialize_array(label))
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
            filename = self.dest_path + file[:-3] + "tfrecords"
            self.write_image_to_tfr(image, label, filename)

class DataReader():
    def __init__(self):
        def __init__(self, src_path):
            self.src_path = src_path
        
            self.filenames = [f for f in listdir(
                src_path) if isfile(join(src_path, f))]