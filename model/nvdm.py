# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/3/10
"""

import time
import numpy as np
import tensorflow as tf

from base import Model

try:
  linear = tf.nn.rnn_cell.linear
except:
  from tensorflow.python.ops.rnn_cell import _linear as linear

class NVDM(Model):
  """Neural Varational Document Model"""

  def __init__(self, sess,reader,args):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: The name of dataset to use.
      h_dim: The dimension of document representations (h). [50, 200]
    """
    self.sess = sess
    self.reader = reader

    self.h_dim = args.h_dim
    self.embed_dim = args.embed_dim
    self.batch_size = 1
    self.max_iter =args.max_iter
    self.checkpoint_dir = args.checkpoint_dir
    self.n_sample=1
    self.step = tf.Variable(0, trainable=False)  
    self.dataset = args.dataset
    self.learning_rate=args.learning_rate
    self._attrs = ["h_dim", "embed_dim", "max_iter", "dataset",
                   "learning_rate", "decay_rate", "decay_step"]

    self.build_model()
    
  def build_model(self):
    self.x = tf.placeholder(tf.float32, [self.reader.vocab_size], name="input")
    self.x_idx = tf.placeholder(tf.int32, [None], name='x_idx')  # mask paddings

    self.build_encoder()
    self.build_generator()

    self.objective = self.kl +self.recons_loss
    
    # optimizer for alternative update
    optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=0.1)
   
    fullvars = tf.GraphKeys.TRAINABLE_VARIABLES
    print 'fullvars:',fullvars

    enc_vars = tf.get_collection(fullvars,scope='encoder')
    print enc_vars
    
    dec_vars = tf.get_collection(fullvars,scope='generator')
    print dec_vars
    self.lossL2_enc = tf.add_n([ tf.nn.l2_loss(v) for v in enc_vars if 'bias' not in v.name]) * 0.0001
    self.lossL2_dec = tf.add_n([ tf.nn.l2_loss(v) for v in dec_vars if 'bias' not in v.name])
    print 'lossL2_enc:',self.lossL2_enc
    print 'lossL2_dec:',self.lossL2_dec          
    enc_grads = tf.gradients(self.kl+self.lossL2_enc, enc_vars)
    dec_grads = tf.gradients(self.recons_loss+self.lossL2_dec, dec_vars)
    
     
    self.optim_enc = optimizer1.apply_gradients(zip(enc_grads, enc_vars))
    self.optim_dec = optimizer2.apply_gradients(zip(dec_grads, dec_vars))  
    
  def build_encoder(self):
    """Inference Network. q(h|X)"""
    with tf.variable_scope("encoder") as scope_encoder:
      self.l1_w = tf.get_variable(
                    "l1_w",
                    shape=[self.reader.vocab_size,self.embed_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
      self.l2_w = tf.get_variable(
                    "l2_w",
                    shape=[self.embed_dim,self.embed_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
      
      self.mean_w = tf.get_variable(
                    "mean_w",
                    shape=[self.embed_dim,self.h_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
      self.sigma_w = tf.get_variable(
                    "sigma_w",
                    shape=[self.embed_dim,self.h_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
      
      self.l1 = tf.nn.relu(tf.matmul(tf.expand_dims(self.x,0),self.l1_w))
      self.l2 = tf.nn.relu(tf.matmul(self.l1,self.l2_w)) 


      self.mean = tf.matmul(self.l2,self.mean_w)
      self.log_sigma = tf.matmul(self.l2,self.sigma_w)
      self.sigma = tf.exp(self.log_sigma)
      
      self.kl = -0.5 * tf.reduce_sum(1 + 2*self.log_sigma - tf.square(self.mean) - tf.exp(2*self.log_sigma))
      #self.kld = self.mask*self.kld

  def build_generator(self):
    """Inference Network. p(X|h)"""
    with tf.variable_scope("generator") as generator_scope:
      if self.n_sample ==1: #single smaple
        eps = tf.random_normal((self.batch_size, self.h_dim), 0, 1, dtype=tf.float32)
        self.h = tf.add(self.mean, tf.mul(self.sigma, eps))  #word vector
        
        self.R = tf.get_variable("R",
                                 shape=[self.reader.vocab_size,self.h_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable("generate_bias", 
                                 shape=[self.reader.vocab_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        
        self.e = tf.matmul(self.h, self.R, transpose_b=True) + self.b
        print 'self.e',self.e
        self.p_x_i = tf.squeeze(tf.nn.softmax(self.e))
        print 'self.p_x_i:',self.p_x_i
        #熵越小的话，该位置包含的信息比较少，即概率越高！
        self.recons_loss = -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.x_idx) + 1e-10))
        #logits = tf.nn.log_softmax(self.e)
        #self.recons_loss = -tf.reduce_sum(tf.mul(logits,self.x),1)
        print ' self.recons_loss:',self.recons_loss
  def train(self, config):
    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_time = time.time()
    start_iter = self.step.eval()

    iterator = self.reader.iterator()
    for step in xrange(start_iter, start_iter + self.max_iter):
      x, x_idx = iterator.next()
      #print 'shape x:',np.shape(x)
      #print x_idx
      #exit(0)
      _, kl, mu, sigma, h,lossL2_enc = self.sess.run(
          [self.optim_enc, self.kl, self.mean, self.sigma, self.h,self.lossL2_enc],
          feed_dict={self.x: x, self.x_idx: x_idx})
      _, recons, mu, sigma, h,lossL2_dec = self.sess.run(
          [self.optim_dec, self.recons_loss, self.mean, self.sigma, self.h,self.lossL2_dec],
          feed_dict={self.x: x, self.x_idx: x_idx})

      if step % 1000 == 0:
        #print("Step: [%4d/%4d] time: %4.4f, loss: %.8f" \
        #    % (step, self.max_iter, time.time() - start_time, loss))
        print("Step: [%4d/%4d] time: %4.4f, loss: %.8f, kl: %.8f, recons_loss: %.8f, lossL2_enc %f,lossL2_dec %f" \
            % (step, self.max_iter, time.time() - start_time, kl + recons, kl, recons,lossL2_enc,lossL2_dec))

      if step % 2000 == 0:
        self.save(self.checkpoint_dir)

        if self.dataset == "ptb":
          #self.sample(10, "government")
          self.sample(10, "bush")
          self.sample(10, "transportation committee")
          self.sample(10, "policy")
        elif self.dataset == "toy":
          self.sample(3, "a")
          self.sample(3, "g")
          self.sample(3, "k")

  def sample(self, sample_size=20, text=None):
    """Sample the documents."""
    p = 1

    if text != None:
      try:
        x, word_idxs = self.reader.get(text)
        #print np.shape(x)
        #print np.shape(word_idxs)
      
      except Exception as e:
        print(e)
        return
    else:
      x, word_idxs = self.reader.random()
    print(" [*] Text: %s" % " ".join([self.reader.idx2word[word_idx] for word_idx in word_idxs]))

    cur_ps = self.sess.run(self.p_x_i, feed_dict={self.x: x})
    #
    word_idxs = np.array(cur_ps).argsort()[-sample_size:][::-1]
    ps = cur_ps[word_idxs]

    for idx, (cur_p, word_idx) in enumerate(zip(ps, word_idxs)):
      p *= cur_p
      print("  [%d] %-20s: %.4f perp : %4.f" % (idx+1, self.reader.idx2word[word_idx], cur_p,-np.log(p)))
    
