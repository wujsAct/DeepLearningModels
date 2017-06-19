# -*- coding: utf-8 -*-
'''
@time: 2016/12/21
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
class ctxSumV1(Model):
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    super(ctxSumV1, self).__init__()
    self.args = args
    self.batch_size = self.args.batch_size
    self.candidateEntNum= 50
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NEL')
    self.lowdim =2*self.args.rnn_size
    with tf.device('/gpu:0'):
      with tf.variable_scope("ctxSum") as scope:
        #entity linking bilinear weights          
        self.bilinear_w_descrip = tf.get_variable(
                      "bilinear_w_description",
                      shape=[int(self.args.rawword_dim),self.lowdim],
                      initializer=tf.contrib.layers.xavier_initializer())
        print self.bilinear_w_descrip
      
                    
        self.bilinear_w_type = tf.get_variable(
                      "bilinear_w_type",
                      shape=[self.args.figer_type_num,self.lowdim],
                      initializer=tf.contrib.layers.xavier_initializer())
        print self.bilinear_w_type
        
        self.w_prob = tf.get_variable(
                "w_prob",
                shape=[3,1],
                initializer = tf.contrib.layers.xavier_initializer())
        print self.w_prob
        
        self.bias_prob = tf.Variable(tf.zeros([1],dtype=tf.float32),name='bias_prob')
        print self.bias_prob
        
        self.w1 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w1')
        self.w2 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w2')
        self.w3 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w3')
        self.w4 = tf.Variable(tf.ones([1],dtype=tf.float32),name='w4')
    with tf.device('/gpu:1'):
      #for every entity mention, there are 30 candidates entities
      '''
      @revise: 2017/3/29 every entity is a trainning sample.
      None stands for the candidate entities number
      '''
      self.ent_mention_linking_tag = tf.placeholder(tf.float32,[None])
      print 'self.ent_mention_linking_tag:',self.ent_mention_linking_tag
      self.candidate_ent_linking_feature= tf.placeholder(tf.float32,[None,int(self.args.rawword_dim)])
      '''
      @2017/2/8
      all embedding vectors!
      #candidate entity surface name vs. entity mention embedding
      '''
  
      self.candidate_ent_type_feature = tf.placeholder(tf.float32,[None,self.args.figer_type_num])
      self.candidate_ent_prob_feature = tf.placeholder(tf.float32,[None,3])
      self.candidate_ent_coherent_feature = tf.placeholder(tf.float32,[None])
      
      self.ent_mention_lstm_feature = tf.placeholder(tf.float32,[2*self.args.rnn_size])
      
      self.ent_mention_lstm_feature_e1 = tf.expand_dims(self.ent_mention_lstm_feature,1)
#      print 'ent_mention_lstm_feature:',self.ent_mention_lstm_feature
      self.candidate_ent_linking_feature = tf.nn.l2_normalize(self.candidate_ent_linking_feature,0)
      self.candidate_ent_type_feature = tf.nn.l2_normalize(self.candidate_ent_type_feature,0)
      self.candidate_ent_coherent_feature = tf.nn.l2_normalize(self.candidate_ent_coherent_feature,0)
#  
      '''
      2017/2/24 apply two layer MLPs to encoder entity mention information!
      '''
      
      if True:
        self.candidate_ent_type_feature =  tf.nn.dropout(self.candidate_ent_type_feature,self.keep_prob)
        self.candidate_ent_prob_feature =  tf.nn.dropout(self.candidate_ent_prob_feature,self.keep_prob)
        self.candidate_ent_coherent_feature =  tf.nn.dropout(self.candidate_ent_coherent_feature,self.keep_prob)
    '''
    @2016/12/22
    '''
    with tf.device('/gpu:0'):
      temp_descrip = tf.matmul(self.candidate_ent_linking_feature,self.bilinear_w_descrip)
      self.prediction_descrip = tf.nn.sigmoid(tf.matmul(temp_descrip,self.ent_mention_lstm_feature_e1))
      
      print 'prediction_descrip:',self.prediction_descrip
      temp_type = tf.matmul(self.candidate_ent_type_feature,self.bilinear_w_type)
      self.prediction_type = tf.nn.sigmoid(tf.matmul(temp_type,self.ent_mention_lstm_feature_e1))
      print 'prediction_type:',self.prediction_type
      
      self.prediction_prob = tf.matmul(self.candidate_ent_prob_feature,self.w_prob) + self.bias_prob
      print 'prediction_prob:',self.prediction_prob
                                
      '''need to imporve the features'''
      self.prediction =  tf.nn.softmax(tf.add(self.w1*self.prediction_prob,tf.add(self.w2*self.candidate_ent_coherent_feature,tf.add(self.w3*self.prediction_descrip,self.w4*self.prediction_type))))
      print 'ctx prediction:',self.prediction
      print 'ctxSum ent_mention_linking_tag', self.ent_mention_linking_tag
    
      self.linking_loss  = tf.reduce_mean(-tf.reduce_sum(self.ent_mention_linking_tag * tf.log(self.prediction+1e-9), reduction_indices=[1]))
      #self.linking_loss = tf.reduce_mean(tf.contrib.losses.hinge_loss(self.prediction,self.ent_mention_linking_tag))
      print 'linking_loss:', self.linking_loss
    
      correct_predict = tf.equal(tf.argmax(self.prediction,0),tf.argmax(self.ent_mention_linking_tag,0))
      print correct_predict
      self.accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))