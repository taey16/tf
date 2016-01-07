
import tensorflow as tf
import numpy as np
from input_data import input_data


batch_size = 32
IMAGE_PIXELS = 28*28
data_set = input_data.read_data_set('MNIST_data', one_hot=True)


images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
