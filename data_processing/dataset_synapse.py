import numpy as np
import tensorflow as tf
import cv2
import os
import numpy as np
from scipy.ndimage.interpolation import zoom


def load_data(data_dir, n_files, output_size=512):
    image_3d, label_3d = [], []
    for _, _, files in os.walk(data_dir):
        for counter, file in enumerate(files):
            if counter >= n_files:
                break
            data = np.load(data_dir + file)
            image = cv2.cvtColor(data['image'], cv2.COLOR_GRAY2RGB)
            label = data['label']
            w, h, c = image.shape
            if w != output_size or h != output_size:
                image = zoom(
                    image, (output_size / w, output_size / h, 1), order=3)
                label = zoom(label, (output_size /
                                             w, output_size / h), order=0)
            image_3d.append(image)
            label_3d.append(tf.one_hot(label, depth=9))

    return np.array(image_3d), np.array(label_3d)
