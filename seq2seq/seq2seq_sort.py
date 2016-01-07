
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq

def make_loop_function(input_elements, memory_cell, soft=True):
    """
    Args:
      input_elements: List of [batch_size * cell.input_dim]
    """
    seq_length = len(input_elements)
    input_elements = tf.concat(1, [tf.expand_dims(input_t, 1)
                                   for input_t in input_elements])
    
    def loop_function(output_t, t):
        if soft:
            softmax = tf.nn.softmax(output_t)
            weighted_inputs = input_elements * tf.expand_dims(softmax, 2)
            processed = tf.reduce_sum(weighted_inputs, 1)
        else:
            # TODO
            pass
        
        return processed
    
    return loop_function

tf.ops.reset_default_graph()

seq_length = 5
batch_size = 64

vocab_size = 10
embedding_dim = 50

memory_dim = 100

inp = [tf.placeholder(tf.int32, shape=(batch_size,),
                      name="inp%i" % t)
       for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(batch_size,),
                        name="labels%i" % t)
          for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]
prev_mem = tf.zeros((batch_size, memory_dim))

cell = rnn_cell.GRUCell(memory_dim)
cell = rnn_cell.EmbeddingWrapper(cell, vocab_size)

enc_outputs, enc_states = rnn.rnn(cell, inp, dtype=tf.float32)

with tf.variable_scope("RNN/EmbeddingWrapper", reuse=True):
    embeddings = tf.get_variable("embedding")
    inp_embedded = [tf.nn.embedding_lookup(embeddings, inp_t)
                    for inp_t in inp]

cell = rnn_cell.GRUCell(memory_dim)
attn_states = tf.concat(1, [tf.reshape(e, [-1, 1, cell.output_size])
                            for e in enc_outputs])
dec_inp = [tf.zeros((batch_size, cell.input_size), dtype=tf.float32)
           for _ in range(seq_length)]

dec_outputs, dec_states = seq2seq.attention_decoder(dec_inp, enc_states[-1],
                                                    attn_states, cell, output_size=seq_length,
                                                    loop_function=make_loop_function(inp_embedded, cell))
loss = seq2seq.sequence_loss(dec_outputs, labels, weights, seq_length)

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
summary_op = loss # tf.merge_all_summaries()

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

def train_batch(batch_size):
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]
    y = [np.argsort(x) for x in X] # [np.arange(seq_length) for _ in X]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    y = np.array(y).T

    feed_dict = {inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: y[t] for t in range(seq_length)})

    loss = sess.run([train_op, summary_op], feed_dict)[1]
    print loss
    return loss

train_batch(batch_size)

loss_history = [train_batch(batch_size) for _ in xrange(100)]


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(loss_history)
