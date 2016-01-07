
import os
import zipfile
import collections
import numpy as np
import random

class DataLoader:

  data = None
  count= None
  dictionary = None
  reverse_dictionary = None
  data_index = 0


  def maybe_download(self, filename, expected_bytes):
    url = 'http://mattmahoney.net/dc/'
    """
    Download a file if not present, 
       and make sure it's the right size.
    """
    if not os.path.exists(filename):
      filename, _ = urllib.request.urlretrieve(
        url + filename, filename
      )
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
      print('Found and verified', filename)
    else:
      print(statinfo.st_size)
      raise Exception(
        'Failed to verify ' + 
        filename + 
        '. Can you get to it with a browser?')
    return filename


  # Read the data into a string.
  def read_data(self, filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
      return f.read(name).split()
    f.close()


  def build_dataset(self, words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(
      collections.Counter(words).most_common(vocabulary_size - 1)
    )
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count = unk_count + 1
      data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = \
      dict(zip(dictionary.values(), dictionary.keys()))

    self.data = data
    self.count= count
    self.dictionary = dictionary
    self.reverse_dictionary = reverse_dictionary
    return data, count, dictionary, reverse_dictionary


  # Step 4: 
  # Function to generate a training batch for the skip-gram model.
  def generate_batch(self, batch_size, num_skips, skip_window):
    assert(self.data <> None)
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size),  dtype=np.int32)
    labels= np.ndarray(shape=(batch_size,1),dtype=np.int32)
    # [ skip_window target skip_window ]
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(self.data[self.data_index])
      self.data_index = (self.data_index + 1) % len(self.data)
    for i in range(batch_size // num_skips):
      # target label at the center of the buffer
      target = skip_window
      targets_to_avoid = [ skip_window ]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(self.data[self.data_index])
      self.data_index = (self.data_index + 1) % len(self.data)

    return batch, labels


  def print_batch(self, batch, labels):
    assert(batch.size <> None) 
    assert(labels.size <> None) 

    for i in range(8):
      print(batch[i], '->', labels[i, 0])
      print(self.reverse_dictionary[batch[i]], '->', 
            self.reverse_dictionary[labels[i, 0]])


