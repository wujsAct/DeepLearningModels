# -*- coding: utf-8 -*-
'''
@time: 2016/12/21
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time

class ctxSum(Model):
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    super(ctxSum, self).__init__()
    self.args = args
    self.batch_size = self.args.batch_size
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    with tf.variable_scope("ctxSum") as scope:
      #entity linking bilinear weights          
      self.bilinear_w_descrip = tf.get_variable(
                    "bilinear_w_description",
                    shape=[int(self.args.rawword_dim),2*self.args.rnn_size],
                    initializer=tf.contrib.layers.xavier_initializer())
      print self.bilinear_w_descrip
      '''
      self.bilinear_w_mentWordv = tf.get_variable(
                    "bilinear_w_mentWordv",
                    shape=[int(self.args.rawword_dim),2*self.args.rnn_size],
                    initializer=tf.contrib.layers.xavier_initializer())
      print self.bilinear_w_mentWordv
      '''
                  
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
      
      self.w1 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w1')
      self.w2 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w2')
      self.w3 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w3')
      self.w4 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w4')
      #self.w5 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w5')
      
      
    #for every entity mention, there are 30 candidates entities
    self.ent_mention_linking_tag = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num])
    self.candidate_ent_linking_feature= tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,int(self.args.rawword_dim)])

    '''
    @2017/2/8
    all embedding vectors!
    #candidate entity surface name vs. entity mention embedding
    '''
    #self.ent_surfacewordv_feature= tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,int(self.args.rawword_dim)])
    
    self.candidate_ent_type_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,self.args.figer_type_num])
    self.candidate_ent_prob_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num,3])
    self.ent_mention_lstm_feature = tf.placeholder(tf.float32,[None,2*self.args.rnn_size,1])
    self.candidate_ent_coherent_feature = tf.placeholder(tf.float32,[None,self.args.candidate_ent_num])
    
  
    #self.ent_surfacewordv_feature = tf.nn.l2_normalize(self.ent_surfacewordv_feature,1)
    
    self.candidate_ent_linking_feature = tf.nn.l2_normalize(self.candidate_ent_linking_feature,2)
    self.ent_mention_lstm_feature = tf.nn.l2_normalize(self.ent_mention_lstm_feature,1)
    self.candidate_ent_type_feature = tf.nn.l2_normalize(self.candidate_ent_type_feature,2)
    self.candidate_ent_coherent_feature = tf.nn.l2_normalize(self.candidate_ent_coherent_feature,1)
    self.candidate_ent_prob_feature = tf.nn.l2_normalize(self.candidate_ent_prob_feature,2)
    
    '''
    @2016/12/22
    '''
    temp_descrip = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_descrip),self.candidate_ent_linking_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,2*self.args.rnn_size))))
    
    #temp_mentWordV = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_mentWordv),self.ent_surfacewordv_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,2*self.args.rnn_size))))
    
    
    temp_type = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_type),self.candidate_ent_type_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,2*self.args.rnn_size))))

    
    self.prediction_descrip = tf.batch_matmul(temp_descrip,self.ent_mention_lstm_feature)[:,:,0]
    
    #self.prediction_mentWordV = tf.batch_matmul(temp_mentWordV,self.ent_mention_lstm_feature)[:,:,0]
    
    self.prediction_type = tf.batch_matmul(temp_type,self.ent_mention_lstm_feature)[:,:,0]
    self.prediction_prob = tf.scan(lambda a,x: tf.matmul(x,self.w_prob)+self.bias_prob,self.candidate_ent_prob_feature,initializer=tf.Variable(tf.random_normal((self.args.candidate_ent_num,1))))[:,:,0]
    
    '''need to imporve the features'''
    
    self.prediction =  tf.nn.softmax(tf.add(self.w1*self.prediction_prob,tf.add(self.w2*self.candidate_ent_coherent_feature,tf.add(self.w3*self.prediction_descrip,self.w4*self.prediction_type))))
    print 'ctx prediction:',self.prediction
    print 'ctxSum ent_mention_linking_tag', self.ent_mention_linking_tag
    
    self.linking_loss  = tf.reduce_mean(-tf.reduce_sum(self.ent_mention_linking_tag * tf.log(self.prediction+1e-9), reduction_indices=[1]))
     
    print 'linking_loss:', self.linking_loss
    
    correct_predict = tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.ent_mention_linking_tag,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))