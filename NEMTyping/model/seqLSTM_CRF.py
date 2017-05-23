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
import adversarial_losses as adv_lib

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
    
    self.input_data = tf.placeholder(tf.float32,[None,self.args.sentence_length,self.args.word_dim])
  
    self.output_data = tf.placeholder(tf.int32,[None,self.args.sentence_length])
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    self.num_examples = tf.placeholder(tf.int32,name='num_examples')
    self.batch_size = self.args.batch_size
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size,2,keep_prob=self.keep_prob)
    self.layers['crfScore'] = layers_lib.FullyConnection(self.args.class_size)
    self.layers['crfLyaer'] = layers_lib.CRF(self.args.class_size)
    
    self.input_data = tf.nn.l2_normalize(self.input_data,1)  #l2 normalize may has efficient methods.
    
    with tf.device('/gpu:1'):
      self.unary_scores,self.transition_params,_,self.length,self.loss_lm = self.cl_loss_from_embedding(self.input_data)
      print 'self.loss_lm:',self.loss_lm
    
    #with tf.device('/cpu:0'):
    #  _,_,_,_,self.adv_loss = self.adversarial_loss()
    #  print 'self.adv_loss:',self.adv_loss
    self.loss = self.loss_lm
    #self.loss = tf.add(self.loss_lm,self.adv_loss)
    
  def cl_loss_from_embedding(self,embedded,return_intermediate=False):
    output,_,length= self.layers['BiLSTM'](embedded)
    matricized_unary_scores = self.layers['crfScore'](tf.reshape(output,[-1,2*self.args.rnn_size]),tf.nn.relu)
    
    unary_scores = tf.reshape(matricized_unary_scores,
                             [self.num_examples,self.args.sentence_length,self.args.class_size])
      
    transition_params,loss = self.layers['crfLyaer'](unary_scores,self.output_data,length)
      
    return unary_scores,transition_params,_,length,loss
      
  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    return adv_lib.adversarial_loss(self.input_data,
                                      self.loss_lm,
                                      self.cl_loss_from_embedding) 
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

      