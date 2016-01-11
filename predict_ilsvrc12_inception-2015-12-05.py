
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import sys, os, re, time


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          'imagenet', 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          'imagenet', 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup, self.node_id_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name, node_id_to_uid

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

  def id_to_uid(self, node_id):
    if node_id not in self.node_id_lookup:
      return ''
    return self.node_id_lookup[node_id]


def create_graph(graph_def_pb='classify_image_graph_def.pb'):
  with gfile.FastGFile(graph_def_pb, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


filename_list = [entry.strip().split(' ') for entry in open( 
  '/storage/ImageNet/ILSVRC2012/val_synset.txt', 
  'r'
)]
synset_label_map = [entry.strip().split(' ')[0] for entry in open(
  '/storage/ImageNet/ILSVRC2012/synset_words.txt',
  'r'
)]
synset_label_dic = {}
for id, synset in enumerate(synset_label_map):
  synset_label_dic[synset] = id

path_prefix = '/storage/ImageNet/ILSVRC2012/val/%s'
num_top_predictions = 5

graph_def_pb = 'imagenet/classify_image_graph_def.pb'
create_graph(graph_def_pb)
node_lookup = NodeLookup()

sess = tf.Session(
  config=tf.ConfigProto(
    log_device_placement=False
  )
)
softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

filepath = 'imagenet/cropped_panda.jpg'
image_data = gfile.FastGFile(filepath, 'rb').read()
predictions = sess.run(softmax_tensor, 
  {'DecodeJpeg/contents:0': image_data})
predictions = np.squeeze(predictions)
top_k = predictions.argsort()[-num_top_predictions:][::-1]

#import pdb; pdb.set_trace()
for node_id in top_k:
  human_string = node_lookup.id_to_string(node_id)
  uid = node_lookup.id_to_uid(node_id)
  score = predictions[node_id]
  print('(predicted: %d, score = %.5f), %s' % (node_id, score, human_string))

#import pdb; pdb.set_trace()
top_1 = 0
top_5 = 0
for n, item in enumerate(filename_list):
  filepath = path_prefix % item[0]
  label = int(item[1])

  if not gfile.Exists(filepath):
    tf.logging.fatal('File does not exist %s', filepath)
  image_data = gfile.FastGFile(filepath, 'rb').read()

  #import pdb; pdb.set_trace()
  start_predict = time.time()
  predictions = sess.run(softmax_tensor, 
    {'DecodeJpeg/contents:0': image_data})
  elapsed_predict = time.time() - start_predict

  predictions = np.squeeze(predictions)
  top_k = predictions.argsort()[-num_top_predictions:][::-1]

  for k, node_id in enumerate(top_k):
    uid = node_lookup.id_to_uid(node_id)
    if k == 0 and synset_label_dic[uid] == label:
      top_1 += 1.0
      top_5 += 1.0
      break
    if k > 0 and synset_label_dic[uid] == label:
      top_5 += 1.0
      break

  print('%s, top@1: %d/%d = %.4f, top@5: %d/%d = %.4f in %.3f sec.' % \
    (n+1, 
     top_1, n+1, top_1/(n+1)*100, 
     top_5, n+1, top_5/(n+1)*100, 
     elapsed_predict))
  sys.stdout.flush()

