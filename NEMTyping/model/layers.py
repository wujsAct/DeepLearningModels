# -*- coding: utf-8 -*-
"""
Created on Fri May 05 14:47:15 2017

@author: wujs
@function: generate MLP,LSTM layers
"""
import tensorflow as tf

class BiLSTM(object):
  '''
  LSTM layers using dynamic rnn
  '''
  def __init__(self,cell_size,num_layers=1,keep_prob=1.0,name='LSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    self.reuse = None
    self.trainable_weights = None
    self.name = name

  #x() equals to x.__call___()
  def __call__(self,x,seq_length=None):  #__call__ is very efficient when the state of instance changes frequently 
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      fw_cell = tf.contrib.rnn.MultiRNNCell([
          tf.contrib.rnn.LSTMCell(self.cell_size,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
          ])
  
      bw_cell = tf.contrib.rnn.MultiRNNCell([
          tf.contrib.rnn.LSTMCell(self.cell_size,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
          ])
  
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
      
      lstm_out,next_state =  tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,x,
                                            dtype=tf.float32,sequence_length=seq_length,scope='LSTM_1')
      
      #shape(lstm_out) = (batch_size,sequence_length,2*cell_size)
      lstm_out = tf.concat(lstm_out, 2)  #concate the forward and backward
      
      
      if self.keep_prob < 1.:
        lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)
        
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
        
    self.reuse =True
    return lstm_out,next_state

class FullyConnection(object):
  def __init__(self,output_size):
    self.output_size = output_size
    
  def __call__(self,inputs,activation_fn):
    out = tf.contrib.layers.fully_connected(inputs,self.output_size, 
                                           activation_fn=activation_fn,
                                           )
    return out
'''
There are a lot of loss function defined in tensorflow!
'''
def classification_loss(flag,labels,logits):
  if flag == 'figer':
    loss = tf.losses.mean_pairwise_squared_error(labels,logits)
  elif flag=='sigmoid':
    loss = tf.losses.sigmoid_cross_entropy(labels,logits)
  else:
    loss = tf.losses.tf.losses.softmax_cross_entropy(labels,logits)  #must one-hot entropy
  
  return loss