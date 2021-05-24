from scipy.ndimage.interpolation import zoom
from os.path import isfile, join
from os import listdir
from tqdm import tqdm

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import h5py

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 24
N_CLASSES = 9
BUFFER_SIZE = 72

DATA_GC_URI_TRAIN = {
    512: 'gs://aga_bucket/synapse-train-512/',
    224: 'gs://aga_bucket/synapse-train-224/',
}

DATA_GC_URI_TEST = {
    512: 'gs://aga_bucket/synapse-test-512/',
    224: 'gs://aga_bucket/synapse-test-224/',
}


class DataWriter():
    def __init__(self, src_path, dest_path="/", batch_size=25, height=512, width=512):
        self.src_path = src_path
        self.dest_path = dest_path
        self.filenames = [f for f in listdir(
            src_path) if isfile(join(src_path, f))]
        np.random.shuffle(self.filenames)
        self.batch_size = batch_size
        self.n_samples = len(self.filenames)
        self.height = height
        self.width = width

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
        filename = filename+"tfrecords"
        # create a writer that'll store our data to disk
        writer = tf.io.TFRecordWriter(filename)

        image = np.stack([image, image, image], axis=-1)
        out = self.parse_single_image(image=image, label=label)
        writer.write(out.SerializeToString())

        writer.close()
        print(f"Wrote {filename} elements to TFRecord")

    def write_tfrecords(self):
        for file in tqdm(self.filenames):
            data = np.load(self.src_path + file)
            image, label = data['image'], data['label']
            filename = self.dest_path + file[:-3]
            self.write_image_to_tfr(image, label, filename)

    def process_data(self, image, label):

        image = np.stack([image, image, image], axis=-1)
        w, h, c = image.shape
        if w != self.width or h != self.height:
            image = zoom(
                image, (self.width / w, self.height / h, 1), order=3)
            label = zoom(label, (self.width /
                                 w, self.height / h), order=0)
        return image, label

    def write_batch_tfrecords(self):
        n_batches = self.n_samples // self.batch_size
        for i in tqdm(range(n_batches+1)):
            filename = self.dest_path + f'record_{i}.tfrecords'
            writer = tf.io.TFRecordWriter(filename)
            start, end = self.batch_size*i, self.batch_size * \
                (i+1) if self.batch_size * \
                (i+1) < self.n_samples else self.n_samples
            for file in self.filenames[start: end]:
                data = np.load(self.src_path + file)
                image, label = self.process_data(data['image'], data['label'])
                out = self.parse_single_image(image=image, label=label)
                writer.write(out.SerializeToString())
            writer.close()
            print(f"Wrote batch {i} to TFRecord")

    def write_test_tfrecords(self):
        for filename in tqdm(self.filenames):
            data = h5py.File(self.src_path + filename, mode='r')
            image3d, label3d = data['image'][:].astype(
                'float32'), data['label'][:].astype('float32')
            writer = tf.io.TFRecordWriter(
                self.dest_path + filename[:-7] + '.tfrecords')
            for image, label in zip(image3d, label3d):
                image, label = self.process_data(image, label)
                out = self.parse_single_image(image=image, label=label)
                writer.write(out.SerializeToString())
            writer.close()
            print(f"Wrote {filename} to TFRecord")

    def write_test_list(self):
        testdataset = []
        for filename in tqdm(self.filenames):
            data = h5py.File(self.src_path + filename, mode='r')
            image3d, label3d = data['image'][:].astype(
                'float32'), data['label'][:].astype('float32')
            image3d_processed, label3d_processed = [], []
            for image, label in zip(image3d, label3d):
                image, label = self.process_data(image, label)
                label = tf.one_hot(label, depth=N_CLASSES).numpy()
                image3d_processed.append(image)
                label3d_processed.append(label)
            testdataset.append(
                {'image': np.array(image3d_processed), 'label': np.array(label3d_processed)})
        return testdataset


class DataReader():

    def __init__(self, src_path="", height=512, width=512, depth=3):
        self.src_path = src_path
        self.filenames = [self.src_path + f for f in listdir(
            src_path) if isfile(join(src_path, f))]
        self.height = height
        self.width = width
        self.depth = depth

    def parse_tfr_element(self, element):
        data = {
            'label': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
        }

        content = tf.io.parse_single_example(element, data)
        raw_label = content['label']
        raw_image = content['image']

        image = tf.io.parse_tensor(raw_image, out_type=tf.float32)
        image = tf.reshape(image, shape=[self.height, self.width, self.depth])

        label = tf.io.parse_tensor(raw_label, out_type=tf.float32)
        label = tf.reshape(label, shape=[self.height, self.width])
        # label = tf.cast(label, tf.int32)
        # label = tf.one_hot(label, depth=N_CLASSES)
        return (image, label)

    def get_dataset_small(self, filenames=None):
        # create the dataset
        filenames = self.filenames if filenames is None else filenames
        dataset = tf.data.TFRecordDataset(filenames)
        # pass every single feature through our mapping function
        dataset = dataset.map(
            self.parse_tfr_element
        )
        return dataset

    def load_dataset(self, filenames=None):
        filenames = self.filenames if filenames is None else filenames
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=AUTOTUNE)  # automatically interleaves reads from multiple files
        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(
            self.parse_tfr_element, num_parallel_calls=AUTOTUNE
        )

        # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
        return dataset

    def get_dataset(self, filenames=None):
        filenames = self.filenames if filenames is None else filenames
        dataset = self.load_dataset(filenames)
        dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

    def load_dataset_tpu(self, filenames):
      # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
        records = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=AUTOTUNE)
        return records.map(self.parse_tfr_element, num_parallel_calls=AUTOTUNE)

    def get_training_dataset(self, train_fns):
        dataset = self.load_dataset_tpu(train_fns)

        # Create some additional training images by randomly flipping and
        def data_augment(image, label):
            rand1, rand2 = np.random.uniform(size=(2, 1))
            if rand1 > 0.5:
                modified, m_label = self.random_rot_flip(image, label)
            elif rand2 > 0.5:
                modified, m_label = self.random_rotate(image, label)
            else:
                modified, m_label = image, label
            m_label = tf.cast(m_label, tf.int32)
            m_label = tf.one_hot(m_label, depth=N_CLASSES)
            return modified, m_label
        augmented = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
        return augmented.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    def random_rotate(self, image, label):
        rot = np.random.uniform(-20*np.pi/180, 20*np.pi/180)
        modified = tfa.image.rotate(image, rot)
        m_label = tfa.image.rotate(label, rot)
        return modified, m_label

    @tf.function
    def random_rot_flip(self, image, label):
        m_label = tf.reshape(label, (self.width, self.height, 1))
        axis = np.random.randint(0, 2)
        if axis == 1:
            # vertical flip
            modified = tf.image.flip_left_right(image=image)
            m_label = tf.image.flip_left_right(image=m_label)
        else:
            # horizontal flip
            modified = tf.image.flip_up_down(image=image)
            m_label = tf.image.flip_up_down(image=m_label)
        # rot 90
        k_90 = np.random.randint(4)
        modified = tf.image.rot90(image=modified, k=k_90)
        m_label = tf.image.rot90(image=m_label, k=k_90)

        m_label = tf.reshape(m_label, (self.width, self.height))
        return modified, m_label

    def one_hot_encode(self, image, label):
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=N_CLASSES)
        return (image, label)

    def get_dataset_training(self, image_size=224, validation=True):
        gcs_pattern = DATA_GC_URI_TRAIN[image_size] + "*.tfrecords"
        filenames = tf.io.gfile.glob(gcs_pattern)
        if validation:
            filenames.remove(
                DATA_GC_URI_TRAIN[image_size] + "record_4.tfrecords")
            filenames.remove(
                DATA_GC_URI_TRAIN[image_size] + "record_11.tfrecords")
            train_fns = filenames
            validation_fns = [DATA_GC_URI_TRAIN[image_size] + "record_4.tfrecords",
                              DATA_GC_URI_TRAIN[image_size] + "record_11.tfrecords"]

            validation_dataset = self.load_dataset(
                validation_fns).map(self.one_hot_encode, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

            training_dataset = self.get_training_dataset(train_fns)
            return training_dataset, validation_dataset

        return self.get_training_dataset(filenames)

    def get_test_data(self, image_size=224, use_self_filenames=True, batch_size=None, shuffle=False):
        if not use_self_filenames:
            gcs_pattern = DATA_GC_URI_TEST[image_size] + "*.tfrecords"
            filenames = tf.io.gfile.glob(gcs_pattern)
        else:
            filenames = self.filenames
        test_dataset = self.load_dataset_tpu(filenames).map(
            self.one_hot_encode, num_parallel_calls=AUTOTUNE)

        if shuffle:
            test_dataset = test_dataset.shuffle(BUFFER_SIZE)

        if not batch_size is None:
            test_dataset  = test_dataset.batch(batch_size)

        test_dataset = test_dataset.prefetch(AUTOTUNE)
        return test_dataset
