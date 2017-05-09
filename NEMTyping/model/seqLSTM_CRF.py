# -*- coding: utf-8 -*-
'''
@time: 2016/11/30
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time
import layers as layers_lib
import numpy as np
class seqLSTM_CRF(Model):
  '''
  self.args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    '''
    @time: 2016/12/20
    @editor: wujs
    @function: also need to return the candidates entity mentions lstm representation
    '''
    super(seqLSTM_CRF, self).__init__()
    self.args = args
    with tf.device('/gpu:1'):
      self.input_data = tf.placeholder(tf.float32,[None,self.args.sentence_length,self.args.word_dim])
    #need add
    #self.output_data = tf.placeholder(tf.float32,[None,self.args.sentence_length,self.args.class_size])
    self.output_data = tf.placeholder(tf.int32,[None,self.args.sentence_length])
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    self.num_examples = tf.placeholder(tf.int32,name='num_examples')
    self.batch_size = self.args.batch_size
    
    self.input_data = tf.nn.l2_normalize(self.input_data,2)  #l2 normalize may has efficient methods.
    if self.args.dropout:
      self.input_data =  tf.nn.dropout(self.input_data,self.keep_prob)
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size,2)

    with tf.device('/gpu:1'):
      self.crf_weights = tf.get_variable("crf_weights",
                                   shape=[2*self.args.rnn_size,self.args.class_size],
                                   initializer=tf.contrib.layers.xavier_initializer())
      self.output,_,self.length= self.layers['BiLSTM'](self.input_data)
      print self.output
    
    with tf.device('/gpu:0'):
      matricized_x_t  = tf.reshape(self.output,[-1,2*self.args.rnn_size])
      
      if self.args.dropout:
        matricized_x_t =  tf.nn.dropout(matricized_x_t,self.keep_prob)
      
      matricized_unary_scores = tf.nn.relu(tf.matmul(matricized_x_t,self.crf_weights))
      
      
      self.unary_scores = tf.reshape(matricized_unary_scores,
                             [self.num_examples,self.args.sentence_length,self.args.class_size])
      print 'unary_scores:',self.unary_scores
      print 'self.output_data:',self.output_data
      print 'self.length:',self.length
      self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        self.unary_scores, self.output_data, self.length)
      
      
      self.loss = tf.reduce_mean(-self.log_likelihood)
      
#      '''
#      @revise time: 2017/2/19 add crf layer to predict sequence label!
#      '''
#      
#      output_f = tf.reshape(self.output,[-1,2*self.args.rnn_size])
#      print output_f      
#      W = tf.get_variable(
#                "W",
#                shape=[2*self.args.rnn_size, self.args.class_size],
#                initializer=tf.contrib.layers.xavier_initializer())
#      b = tf.Variable(tf.constant(0.1,shape=[self.args.class_size]),name="b")
#      prediction = tf.nn.softmax(tf.nn.xw_plus_b(output_f,W,b,name="scores"))
#      self.prediction = tf.reshape(prediction,[-1,self.args.sentence_length,self.args.class_size])
#      
#      cross_entropy = self.output_data * tf.log(self.prediction)
#      cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
#      mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
#      cross_entropy *= mask
#      cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
#      cross_entropy /= tf.cast(self.length, tf.float32)
#      self.loss = tf.reduce_mean(cross_entropy)

      