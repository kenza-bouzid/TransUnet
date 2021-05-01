import numpy as np
import tensorflow as tf
import cv2
import os
import numpy as np

def load_data(data_dir, n_files):
    image_3d, label_3d = [], []    
    for _, _, files in os.walk(data_dir):
        for counter, file in enumerate(files):
            if counter >= n_files:
                break
            data = np.load(data_dir + file)
            image = cv2.cvtColor(data['image'], cv2.COLOR_GRAY2RGB)
            # image = tf.keras.applications.imagenet_utils.preprocess_input(
            # image, data_format=None, mode="tf") 
            image_3d.append(image)
            label_3d.append(tf.one_hot(data['label'], depth=9))


    return np.array(image_3d), np.array(label_3d)