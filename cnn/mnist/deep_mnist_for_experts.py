
import input_data
import tensorflow as tf
import numpy as np
import inspect

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

use_conv = True
use_sgd = False
batch_size = 32
max_step = 20000
lr = 0.001
epsilon = 1e-4
log_device_placement = False

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#sess = tf.InteractiveSession()
sess = tf.Session(
  config=tf.ConfigProto(
    log_device_placement=log_device_placement
  )
)
x = tf.placeholder(np.float32, shape=[None, 784])
y_= tf.placeholder(np.float32, shape=[None, 10])
#print(inspect.getmembers(sess))
#print('\n')
#print(inspect.getmembers(x))


keep_prob = tf.placeholder(np.float32)
dropout_ratio = 0.5
if use_conv:
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  x_image = tf.reshape(x, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
else:
  W_fc1 = tf.Variable(tf.zeros([784, 10]))
  b_fc1 = tf.Variable(tf.zeros([10]))
  #W = tf.Variable(tf.zeros([10, 10]))
  #b = tf.Variable(tf.zeros([10]))
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  #h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
  #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  #y = tf.nn.softmax(tf.matmul(h_fc1_drop,W) + b)

  y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, np.float32))

if use_sgd:
  train_step = \
    tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
else:
  train_step = \
    tf.train.AdamOptimizer(epsilon).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())


with sess:
  for step in range(max_step):
    batch = mnist.train.next_batch(batch_size)
    #_, train_loss = train_step.run(
    _, train_loss = sess.run([train_step, cross_entropy],
      feed_dict={x: batch[0], 
                 y_:batch[1], 
                 keep_prob: dropout_ratio})

    train_accuracy = accuracy.eval(
      feed_dict={x: batch[0], 
                 y_:batch[1], 
                 keep_prob: 1.0})
    print("step %d, trn loss: %f, acc: %f " %
     (step, np.log(train_loss) / batch_size, train_accuracy * 100))

    if step % 40 == 0:
      test_accuracy = accuracy.eval(
        feed_dict={x: mnist.test.images, 
                   y_:mnist.test.labels, keep_prob: 1.0})
      print("  step %d, tst acc: %f" %
        (step, test_accuracy))


