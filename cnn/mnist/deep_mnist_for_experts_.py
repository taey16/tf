
import input_data
import tensorflow as tf
import numpy as np
import inspect

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.01, name=name)
  return tf.Variable(initial)


def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape, name=name)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

use_conv = True
use_sgd = True
batch_size = 32
test_batch_size = 10000
max_step = 20000
lr = 0.0001
mom = 0.9
epsilon = 1e-4
log_device_placement = False

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
target= tf.placeholder(tf.float32, shape=[None, 10])
#print(inspect.getmembers(sess))
#print('\n')
#print(inspect.getmembers(x))


keep_prob = tf.placeholder(tf.float32)
dropout_ratio = 0.5
if use_conv:
  ## batch_size: -1 hold batch size
  x_image = tf.reshape(x, [-1,28,28,1])
  with tf.name_scope('conv1'):
    weight = weight_variable([5, 5, 1, 32], 'weight')
    bias = bias_variable([32], 'bias')
    h_conv1 = tf.nn.relu(conv2d(x_image,weight) + bias)
  h_pool1 = max_pool_2x2(h_conv1)
  with tf.name_scope('conv2'):
    weight = weight_variable([5, 5, 32, 64], 'weight')
    bias = bias_variable([64], 'bias')
    h_conv2 = tf.nn.relu(conv2d(h_pool1,weight) + bias)
  h_pool2 = max_pool_2x2(h_conv2)
  ## batch_size: -1 hold batch size
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  with tf.name_scope('fc1'):
    weight = weight_variable([7 * 7 * 64, 1024], 'weight')
    bias = bias_variable([1024], 'bias')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight) + bias)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  with tf.name_scope('fc2'):
    weight = weight_variable([1024, 10], 'weight')
    bias = bias_variable([10], 'bias')
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, weight) + bias)
else:
  W_fc1 = weight_variable([784, 784], 'W_fc1')
  b_fc1 = weight_variable([784], 'b_fc1')
  W = weight_variable([784, 10], 'W')
  b = weight_variable([10], 'b')
  #W = tf.Variable(tf.zeros([784, 10]))
  #b = tf.Variable(tf.zeros([10]))
  #y = tf.nn.softmax(tf.matmul(x,W) + b)

  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  y = tf.nn.softmax(tf.matmul(h_fc1_drop,W) + b)


cross_entropy = -tf.reduce_sum(target * tf.log(y))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if use_sgd:
  train_step = \
    tf.train.MomentumOptimizer(lr, mom).minimize(cross_entropy)
else:
  train_step = \
    tf.train.AdamOptimizer(epsilon).minimize(cross_entropy)

#sess = tf.InteractiveSession()
sess = tf.Session(
  config=tf.ConfigProto(
    log_device_placement=log_device_placement
  )
)
sess.run(tf.initialize_all_variables())

test_feed_dict = \
  {x: mnist.test.images,
   target: mnist.test.labels,
   keep_prob: 1}
for step in range(max_step):
  batch = mnist.train.next_batch(batch_size)
  train_feed_dict = \
    {x: batch[0], 
     target:batch[1], 
     keep_prob: dropout_ratio}

  try:
    _, train_loss, train_accuracy = sess.run(
      [train_step, cross_entropy, accuracy],
      feed_dict=train_feed_dict)
    #train_accuracy = sess.run(accuracy, 
    #  feed_dict = train_feed_dict)
  except Exception as err:
    print('Exception, ', err)

  print("step %d, trn loss: %f, acc: %f " %
   (step, train_loss / batch_size, train_accuracy * 100))

  if step % 120 == 0:
    test_accuracy, test_loss = sess.run([accuracy, cross_entropy], 
      feed_dict=test_feed_dict)
    print("  step %d, tst loss: %f tst acc: %f" %
      (step, test_loss / test_batch_size, test_accuracy * 100))


