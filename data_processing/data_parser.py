from os.path import isfile, join
from os import listdir
from numpy.lib.shape_base import dstack
from tqdm import tqdm

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2
AUTOTUNE = tf.data.experimental.AUTOTUNE
HEIGHT = 512
WIDTH = 512
DEPTH = 3
BATCH_SIZE = 32
N_CLASSES=9
BUFFER_SIZE=64

class DataWriter():
    def __init__(self, src_path, dest_path, batch_size=25):
        self.src_path = src_path
        self.dest_path = dest_path
        self.filenames = [f for f in listdir(
            src_path) if isfile(join(src_path, f))]
        self.filenames = np.random.shuffle(self.filenames)
        self.batch_size = batch_size
        self.n_samples = len(self.filenames)

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

    def write_batch_tfrecords(self):
        n_batches = self.n_samples // self.batch_size
        for i in tqdm(range(n_batches+1)):
            filename = f'record_{i}.tfrecords'
            writer = tf.io.TFRecordWriter(filename)
            start, end = self.batch_size*i, self.batch_size*(i+1) if self.batch_size*(i+1) < self.n_samples else self.n_samples
            for file in self.filenames[start: end]:
                data = np.load(self.src_path + file)
                image, label = data['image'], data['label']
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                out = self.parse_single_image(image=image_rgb, label=label)
                writer.write(out.SerializeToString())
            writer.close()
            print(f"Wrote batch {i} to TFRecord")


class DataReader():

    def __init__(self, src_path):
        self.src_path = src_path
    
        self.filenames = [self.src_path + f for f in listdir(
            src_path) if isfile(join(src_path, f))]

    def parse_tfr_element(self, element):
        data = {
            'label':tf.io.FixedLenFeature([], tf.string),
            'image' : tf.io.FixedLenFeature([], tf.string),
            }

            
        content = tf.io.parse_single_example(element, data)
        raw_label = content['label']
        raw_image = content['image']
        
        
        image = tf.io.parse_tensor(raw_image, out_type=tf.float32)
        image = tf.reshape(image, shape=[HEIGHT,WIDTH,DEPTH])

        label = tf.io.parse_tensor(raw_label, out_type=tf.float32)
        label = tf.reshape(label, shape=[HEIGHT,WIDTH])
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=N_CLASSES)
        return (image, label)

    def get_dataset_small(self, filenames=None):
        #create the dataset
        filenames = self.filenames if filenames is None else filenames
        dataset = tf.data.TFRecordDataset(filenames)

        #pass every single feature through our mapping function
        dataset = dataset.map(
            self.parse_tfr_element
        )
            
        return dataset
    def load_dataset(self, filenames=None):
        filenames = self.filenames if filenames is None else filenames
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(
            self.parse_tfr_element, num_parallel_calls=AUTOTUNE
        )
        # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
        return dataset
    
    def get_dataset(self, filenames=None):
        filenames = self.filenames if filenames is None else filenames
        dataset = self.load_dataset(filenames)
        # dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset
