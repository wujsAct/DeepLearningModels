import time
import numpy as np
import tensorflow as tf
from utils import array_pad
from base_model import Model
import os
#from cells import LSTMCell, MultiRNNCellWithSkipConn
from data_utils import QADataset

class DeepLSTM(Model):
  """Deep LSTM model."""
  def __init__(self, hidden_size=256, depth=2, batch_size=32,
               keep_prob=0.1, n_input=1000,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for an Deep LSTM model.
    
    Args:
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      keep_prob: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      n_input: int, The max time unit [1000]
    """
    super(DeepLSTM, self).__init__()

    self.hidden_size = int(hidden_size)
    self.depth = int(depth)
    self.batch_size = int(batch_size)
    #self.hidden_output_size = self.depth * self.hidden_size
    #hidden depth will not change the hidden outputs?
    self.hidden_output_size = self.hidden_size
    self.keep_prob = float(keep_prob)
    self.n_input = int(n_input)
    self.checkpoint_dir = checkpoint_dir

    start = time.clock()
    print(" [*] Building Deep LSTM...")
    self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=0.0,state_is_tuple=True)
    #if not forward_only and self.keep_prob < 1:
    #  self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=keep_prob)
    #self.stacked_cell = MultiRNNCellWithSkipConn([self.cell] * depth)
    self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*depth,state_is_tuple=True)
    
    #self.initial_state = self.stacked_cell.zero_state(batch_size, tf.float32)
    
  def prepare_model(self, data_dir, dataset_name, vocab_size):
    #In this model,vocab_size is the prediction length.
    vocab_fname = os.path.join(data_dir,dataset_name,dataset_name+'.vocab')
    qaData_t = QADataset(data_dir,dataset_name,vocab_fname)
    if not self.vocab:
      self.vocab,self.n_entities = qaData_t.initialize_vocabulary()
      print(" [*] Loading vocab finished.")

    self.vocab_size = len(self.vocab)
    
    self.emb = tf.get_variable("emb", [self.vocab_size, self.hidden_size])
    
    #something wrong with the feature mapping!
    self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.n_input])
    embed_inputs = tf.nn.embedding_lookup(self.emb, self.inputs)
    self.input_length = tf.placeholder(tf.int32, [self.batch_size])
    outputs,states = tf.nn.dynamic_rnn(self.stacked_cell,
                        tf.unpack(embed_inputs),
                        dtype=tf.float32,
                        #initial_state=self.initial_state
                        sequence_length=self.input_length
                        )
    print(np.asarray(self.input_length))
    outputs = tf.pack([tf.slice(outputs, [0,nstarts-1,0], [1, 1, self.hidden_size])
         for idx, nstarts in enumerate(tf.unpack(self.input_length))])
    self.outputs = tf.reshape(outputs, [self.batch_size, self.hidden_size]) 
    #define weights
    self.weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input,self.hidden_size])),
            'out': tf.Variable(tf.random_normal([self.hidden_size,self.n_entities]))
    }
    self.biases = {
            'hidden':tf.Variable(tf.random_normal([self.hidden_size])),
            'out': tf.Variable(tf.random_normal([self.n_entities]))
    }
    
    tf.histogram_summary("weights", self.weights['out'])
    tf.histogram_summary("output", self.outputs)

    self.y = tf.placeholder(tf.float32, [self.batch_size, self.n_entities])
    self.y_ = tf.matmul(self.outputs, self.weights['out']) + self.biases['out']

    self.loss = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.y)
    tf.scalar_summary("loss", tf.reduce_mean(self.loss))

    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", self.accuracy)

    print(" [*] Preparing model finished.")

  def train(self, sess, vocab_size, epoch=25, learning_rate=0.0002,
            momentum=0.9, decay=0.95, data_dir="data", dataset_name="cnn"):
    
    self.prepare_model(data_dir, dataset_name, vocab_size)
    data_max_idx = 380298
    start = time.clock()
    print(" [*] Calculating gradient and loss...")
    self.optim = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(self.loss)
    print(" [*] Calculating gradient and loss finished. Take %.2fs" % (time.clock() - start))

    # Could not use  because the sparse update of RMSPropOptimizer
    # is not implemented yet (2016.01.24).
#    self.optim = tf.train.RMSPropOptimizer(learning_rate,
#                                            decay=decay,
#                                            momentum=momentum).minimize(self.loss)
#    print(" [*] Calculating gradient and loss finished. Take %.2fs" % (time.clock() - start))
    sess.run(tf.initialize_all_variables())

    if self.load(sess, self.checkpoint_dir, dataset_name):
      print(" [*] Deep Bidirect LSTM checkpoint is loaded.")
    else:
      print(" [*] There is no checkpoint for this model.")

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("tmp/deep", sess.graph_def)

    counter = 0
    start_time = time.time()
    for epoch_idx in xrange(epoch):
      #data_loader = load_dataset(data_dir, dataset_name, vocab_size,self.batch_size)
      vocab_fname = os.path.join(data_dir,dataset_name,dataset_name+'.vocab')
      qaData = QADataset(data_dir,dataset_name,vocab_fname)
      data_loader = qaData.load_dataset1(self.batch_size)
      while True:
        try:
          #starttime = time.time()
          inputs,y,input_length = data_loader.next()
          #print 'load data time',time.time() - starttime
          _, summary_str, cost, accuracy = sess.run([self.optim, merged, self.loss, self.accuracy], 
                                                   feed_dict={self.inputs: inputs,
                                                   self.y: y,
                                                   self.input_length:input_length})
          if counter % 2 == 0:
            writer.add_summary(summary_str, counter)
            #about 300s iteration!
            data_idx = (counter+1) * self.batch_size
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                % (epoch_idx, data_idx, data_max_idx, time.time() - start_time, np.mean(cost), accuracy))
          counter += 1
        except StopIteration:
          break
       
      self.save(sess, self.checkpoint_dir, dataset_name)

  def test(self, voab_size):
    self.prepare_model(data_dir, dataset_name, vocab_size)