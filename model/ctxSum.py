# -*- coding: utf-8 -*-
'''
@time: 2016/12/21
@editor: wujs
@function: 简单的求和description 字向量，利用bilinear来求解entity linking problem！

@time: 2016/12/28
加上其余的特征，将所有predict结果，进行concat操作，然后通过MLP去获得最终的entity linking的结果！
'''
import tensorflow as tf
from base_model import Model
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
import time

class ctxSum(Model):
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    super(ctxSum, self).__init__()
    self.args = args
    self.batch_size = self.args.batch_size
    with tf.variable_scope("ctxSum") as scope:
      #entity linking 需要修正的是bilinear weights          
      self.bilinear_w_descrip = tf.get_variable(
                    "bilinear_w_description",
                    shape=[self.args.rawword_dim,2*self.args.rnn_size],
                    initializer=tf.contrib.layers.xavier_initializer())
      print self.bilinear_w_descrip
                  
      self.bilinear_w_type = tf.get_variable(
                    "bilinear_w_type",
                    shape=[self.args.figer_type_num,2*self.args.rnn_size],
                    initializer=tf.contrib.layers.xavier_initializer())
      print self.bilinear_w_type
      
      self.w_prob = tf.get_variable(
              "w_prob",
              shape=[3,1],
              initializer = tf.contrib.layers.xavier_initializer())
      print self.w_prob
      
      self.bias_prob = tf.Variable(tf.zeros([1],dtype=tf.float32),name='bias_prob')
      print self.w_prob
      '''
      self.w_f_1 = tf.get_variable(
              "w_f_1",
              shape=[30*4,30*2],
              initializer = tf.contrib.layers.xavier_initializer())
      self.bias_f_1 = tf.Variable(tf.zeros([30*2],dtype=tf.float32),name='bias_f_1')
      
      self.w_f_2 = tf.get_variable(
              "w_f_2",
              shape=[30*2,30],    
              initializer = tf.contrib.layers.xavier_initializer())
      self.bias_f_2 = tf.Variable(tf.zeros([30],dtype=tf.float32),name='bias_f_2')
      '''
    #for every entity mention, there are 30 candidates entities
    self.ent_mention_linking_tag = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num])
    self.candidate_ent_linking_feature= tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,self.args.rawword_dim])
    self.candidate_ent_type_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,self.args.figer_type_num])
    self.candidate_ent_prob_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,3])
    self.ent_mention_lstm_feature = tf.placeholder(tf.float32,[None,2*self.args.rnn_size,1])
    self.candidate_ent_coherent_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num])
    
    
    self.candidate_ent_linking_feature = tf.nn.l2_normalize(self.candidate_ent_linking_feature,1)
    self.ent_mention_lstm_feature = tf.nn.l2_normalize(self.ent_mention_lstm_feature,1)
    self.candidate_ent_type_feature = tf.nn.l2_normalize(self.candidate_ent_type_feature,1)
    self.candidate_ent_coherent_feature = tf.nn.l2_normalize(self.candidate_ent_coherent_feature,1)
    self.candidate_ent_prob_feature = tf.nn.l2_normalize(self.candidate_ent_prob_feature,1)
    
    '''
    @2016/12/22 不知道为啥einsum对不定长的数据就失效了呢！
    #temp =  tf.einsum('ijk,kl->ijl',self.candidate_ent_linking_feature,self.bilinear_w)
    #tf.scan 或者 tf.einsum 都可以解决这个问题！tensorflow 丰富的函数库，可以减少很多工作呢！
    '''
    temp_descrip = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_descrip),self.candidate_ent_linking_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,2*self.args.rnn_size))))
    
    temp_type = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_type),self.candidate_ent_type_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,2*self.args.rnn_size))))

    self.prediction_descrip = tf.batch_matmul(temp_descrip,self.ent_mention_lstm_feature)[:,:,0]
    self.prediction_type = tf.batch_matmul(temp_type,self.ent_mention_lstm_feature)[:,:,0]
    self.prediction_prob = tf.scan(lambda a,x: tf.matmul(x,self.w_prob)+self.bias_prob,self.candidate_ent_prob_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,1))))[:,:,0]
    
    '''
    all_feature_1 = tf.concat(1,[self.prediction_descrip,self.prediction_type])
    all_feature_2 = tf.concat(1,[self.candidate_ent_coherent_feature,all_feature_1])
    self.all_feature = tf.concat(1,[self.prediction_prob,all_feature_2])
    print 'self.all_feature:',self.all_feature
    
    self.pred_l1 = tf.nn.sigmoid(tf.matmul(self.all_feature,self.w_f_1) + self.bias_f_1)
    print 'self.pred_l1:',self.pred_l1
    self.prediction = tf.nn.softmax(tf.matmul(self.pred_l1,self.w_f_2) + self.bias_f_2)
    '''
    self.prediction = tf.nn.softmax(tf.add(self.prediction_prob,tf.add(self.candidate_ent_coherent_feature,tf.add(self.prediction_descrip,self.prediction_type))))
    
    print 'ctx prediction:',self.prediction
    print 'ctxSum ent_mention_linking_tag', self.ent_mention_linking_tag
    
    self.linking_loss  = tf.reduce_mean(-tf.reduce_sum(self.ent_mention_linking_tag * tf.log(self.prediction+1e-9), reduction_indices=[1]))
    
    print 'linking_loss:', self.linking_loss
    
    correct_predict = tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.ent_mention_linking_tag,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))