# -*- coding: utf-8 -*-
'''
@time: 2016/11/30
@editor: wujs
'''
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
import time

class seqLSTM(object):
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    
    self.input_data = tf.placeholder(tf.float32,[None,args.sentence_length,args.word_dim])
    #need add 
    self.output_data = tf.placeholder(tf.float32,[None,args.sentence_length,args.class_size])
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
    fw_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size,state_is_tuple=True)
    bw_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size,state_is_tuple=True)
    
    if args.dropout:
      fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=self.keep_prob)
      bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=self.keep_prob)
    
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell]*args.num_layers,state_is_tuple=True)
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell]*args.num_layers,state_is_tuple=True)
    start_time = time.time()
    print 'start to do lstm'
    #get sentence length
    used = tf.sign(tf.reduce_max(tf.abs(self.input_data),reduction_indices=2))
    self.length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
    
    output,_,_ = tf.nn.bidirectional_rnn(fw_cell,bw_cell,
                                        tf.unpack(tf.transpose(self.input_data,perm=[1,0,2])),
                                        dtype=tf.float32,sequence_length=self.length
                                        )
    print 'bi-lstm cost time:', time.time()-start_time
    
    if args.dropout:
      output = tf.nn.dropout(output,self.keep_prob)
        
    with tf.name_scope("output"):
      output = tf.reshape(tf.transpose(tf.pack(output),perm=[1,0,2]),[-1,2*args.rnn_size])    
      W = tf.get_variable(
                "W",
                shape=[2*args.rnn_size, args.class_size],
                initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1,shape=[args.class_size]),name="b")         
      prediction = tf.nn.softmax(tf.nn.xw_plus_b(output,W,b,name="scores"))
      self.prediction = tf.reshape(prediction,[-1,args.sentence_length,args.class_size])
      self.loss = self.cost()
       
       
  def cost(self):
    cross_entropy = self.output_data * tf.log(self.prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(self.length, tf.float32)
    return tf.reduce_mean(cross_entropy)

    