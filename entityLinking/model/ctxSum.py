# -*- coding: utf-8 -*-
'''
@time: 2016/12/21
@editor: wujs
@function: 简单的求和description 字向量，利用bilinear来求解entity linking problem！
'''
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
import time

class ctxSum(object):
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    with tf.variable_scope("ctxSum") as scope:
      #entity linking 需要修正的是bilinear weights
      self.bilinear_w = tf.get_variable(
                    "bilinear_w",
                    shape=[args.rawword_dim,2*args.rnn_size],
                    initializer=tf.contrib.layers.xavier_initializer())
      print self.bilinear_w 
    
    #for every entity mention, there are 30 candidates entities
    self.ent_mention_linking_tag = tf.placeholder(tf.float32,[None,args.candidate_ent_num])
    self.candidate_ent_linking_feature= tf.placeholder(tf.float32,[None,args.candidate_ent_num,args.rawword_dim])
    
    self.ent_mention_lstm_feature = tf.placeholder(tf.float32,[None,2*args.rnn_size,1])
    
    self.candidate_ent_linking_feature = tf.nn.l2_normalize(self.candidate_ent_linking_feature,1)
    self.ent_mention_lstm_feature = tf.nn.l2_normalize(self.ent_mention_lstm_feature,1)
    
    '''
    @2016/12/22 不知道为啥einsum对不定长的数据就失效了呢！
    '''
    #temp =  tf.einsum('ijk,kl->ijl',self.candidate_ent_linking_feature,self.bilinear_w)
    
    #tf.scan 或者 tf.einsum 都可以解决这个问题！tensorflow 丰富的函数库，可以减少很多工作呢！
    temp = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w),self.candidate_ent_linking_feature,initializer=tf.Variable(tf.random_normal((args.candidate_ent_num,2*args.rnn_size))))
    print temp
    
    self.prediction =  tf.batch_matmul(temp, self.ent_mention_lstm_feature)
    
    self.prediction = tf.nn.softmax(self.prediction[:,:,0])
    
    print 'ctx prediction:',self.prediction
    print 'ctxSum ent_mention_linking_tag', self.ent_mention_linking_tag
    self.linking_loss = tf.contrib.losses.softmax_cross_entropy(logits=self.prediction, onehot_labels=self.ent_mention_linking_tag)
    
    print 'linking_loss:', self.linking_loss
    
    correct_predict = tf.equal(tf.argmax(self.ent_mention_linking_tag,1),tf.argmax(self.prediction,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))
    