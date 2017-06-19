# -*- coding: utf-8 -*-
'''
@time: 2016/12/21
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time
import numpy as np
class ctxSum(Model):
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args,features):
    super(ctxSum, self).__init__()
    self.args = args
    self.batch_size = self.args.batch_size
    self.candidateEntNum=args.candidate_ent_num
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NEL')
    self.lowdim =2*self.args.rnn_size
    with tf.device('/gpu:0'):
      with tf.variable_scope("ctxSum") as scope:
        '''
        self.lstm_w1 = tf.get_variable(
                      "lstm_w1",
                      shape=[2*self.args.rnn_size,self.args.rnn_size],
                      initializer=tf.contrib.layers.xavier_initializer())
        self.lstm_w2 = tf.get_variable(
                      "lstm_w2",
                      shape=[self.args.rnn_size,self.lowdim],
                      initializer=tf.contrib.layers.xavier_initializer())
        '''
        #entity linking bilinear weights          
        self.bilinear_w_descrip = tf.get_variable(
                      "bilinear_w_description",
                      shape=[int(self.args.rawword_dim),self.lowdim],
                      initializer=tf.contrib.layers.xavier_initializer())
        print self.bilinear_w_descrip
        
        self.bilinear_w_mentWordv = tf.get_variable(
                      "bilinear_w_mentWordv",
                      shape=[int(self.args.rawword_dim),self.lowdim],
                      initializer=tf.contrib.layers.xavier_initializer())
        print self.bilinear_w_mentWordv
        
                    
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
        self.w1 = tf.constant(0,dtype=tf.float32)
        self.w2 = tf.constant(0,dtype=tf.float32)
        self.w2_1 = tf.constant(0,dtype=tf.float32)
        self.w3 = tf.constant(0,dtype=tf.float32)
        self.w4 = tf.constant(0,dtype=tf.float32)
        self.w5 = tf.constant(0,dtype=tf.float32,name='w5')
        if features=='0':  
          self.w1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w1')
        elif features=='1_1':
          self.w1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w1')
          self.w2 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2')
        elif features=='1_2':
          self.w1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w1')
          self.w2 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2')
          self.w2_1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2_1')
        elif features=='2':
          self.w1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w1')
          self.w2 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2')
          self.w2_1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2_1')
          self.w3 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w3')
        elif features=='3':
          self.w1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w1')
          self.w2 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2')
          self.w2_1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2_1')
          self.w3 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w3')
          self.w4 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w4')
        elif features =='4':
          self.w1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w1')
          self.w2 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2')
          self.w2_1 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w2_1')
          self.w3 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w3')
          self.w4 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w4')
          self.w5 = tf.Variable(0.001*tf.ones([1],dtype=tf.float32),name='w5')
        else:
          print 'features number is wrong....'

    with tf.device('/gpu:1'):
      #for every entity mention, there are 30 candidates entities
      self.ent_mention_linking_tag = tf.placeholder(tf.float32,[None,self.candidateEntNum],name='ent_mention_linking_tag')
      print 'self.ent_mention_linking_tag:',self.ent_mention_linking_tag
      self.candidate_ent_linking_feature= tf.placeholder(tf.float32,[None,self.candidateEntNum,int(self.args.rawword_dim)],name='candidate_ent_linking_feature')
      #self.sequence_length = tf.placeholder(tf.int32,[None])
      '''
      @2017/2/8
      all embedding vectors!
      #candidate entity surface name vs. entity mention embedding
      '''
      self.ent_surfacewordv_feature= tf.placeholder(tf.float32,[None,self.candidateEntNum,int(self.args.rawword_dim)])
      
      self.candidate_ent_type_feature = tf.placeholder(tf.float32,[None,self.candidateEntNum,self.args.figer_type_num],name='candidate_ent_type_feature')
      self.candidate_ent_prob_feature = tf.placeholder(tf.float32,[None,self.candidateEntNum,3],name='candidate_ent_prob_feature')
      self.ent_mention_lstm_feature = tf.placeholder(tf.float32,[None,5,2*self.args.rnn_size],name='ent_mention_lstm_feature')
      self.candidate_ent_coherent_feature_ngd= tf.placeholder(tf.float32,[None,self.candidateEntNum],'candidate_ent_coherent_feature')   #we revise (None,self.candidateEntNum,top_k) where top_k is the maximum relevants! ==> in case of indule the relationships.
      self.candidate_ent_coherent_feature_fb= tf.placeholder(tf.float32,[None,self.candidateEntNum],'candidate_ent_coherent_feature')
      self.ent_surfacewordv_feature = tf.nn.l2_normalize(self.ent_surfacewordv_feature,2)
      
      self.candidate_ent_linking_feature = tf.nn.l2_normalize(self.candidate_ent_linking_feature,2)
      #self.ent_mention_lstm_feature = tf.nn.l2_normalize(self.ent_mention_lstm_feature,1)
      self.candidate_ent_type_feature = tf.nn.l2_normalize(self.candidate_ent_type_feature,2)
      self.candidate_ent_coherent_feature_ngd = tf.nn.l2_normalize(self.candidate_ent_coherent_feature_ngd,1)
      self.candidate_ent_coherent_feature_fb = tf.nn.l2_normalize(self.candidate_ent_coherent_feature_fb,1)
      #self.candidate_ent_prob_feature = tf.nn.l2_normalize(self.candidate_ent_prob_feature,2)
    
      print 'tf.reduce_sum(self.ent_mention_lstm_feature,1)[:,:,0]:',tf.reduce_sum(self.ent_mention_lstm_feature,1)
      '''
      2017/2/24 apply two layer MLPs to encoder entity mention information!
      '''
  #    self.ent_mention_lstm_feature_l1 = tf.nn.relu(tf.matmul(tf.nn.l2_normalize(tf.reduce_sum(self.ent_mention_lstm_feature,1),1),self.lstm_w1))
  #    print 'self.ent_mention_lstm_feature_l1:',self.ent_mention_lstm_feature_l1
  #    self.ent_mention_lstm_feature_l2 = tf.expand_dims(tf.nn.relu(tf.matmul(self.ent_mention_lstm_feature_l1,self.lstm_w2)),2)
  #    print 'self.ent_mention_lstm_feature_l1:',self.ent_mention_lstm_feature_l2
      #self.ent_mention_lstm_feature_l2 = tf.nn.l2_normalize(tf.expand_dims(tf.reduce_sum(self.ent_mention_lstm_feature,1),-1),1)
      self.ent_mention_lstm_feature_l2 = tf.expand_dims(tf.reduce_sum(self.ent_mention_lstm_feature,1),-1)
      print 'ent_mention_lstm_feature_l2:',self.ent_mention_lstm_feature_l2
      #self.embedding_x_expanded = tf.expand_dims(self.ent_mention_lstm_feature,-1)
      
      if True:
        self.ent_mention_lstm_feature_l2 =  tf.nn.dropout(self.ent_mention_lstm_feature_l2,self.keep_prob)
        self.candidate_ent_type_feature =  tf.nn.dropout(self.candidate_ent_type_feature,self.keep_prob)
        self.candidate_ent_prob_feature =  tf.nn.dropout(self.candidate_ent_prob_feature,self.keep_prob)
        self.candidate_ent_coherent_feature_ngd =  tf.nn.dropout(self.candidate_ent_coherent_feature_ngd,self.keep_prob)
        self.candidate_ent_coherent_feature_fb =  tf.nn.dropout(self.candidate_ent_coherent_feature_fb,self.keep_prob)
       
    '''
    print 'self.embedding_x_expanded:',self.embedding_x_expanded
    pooled_outputs = []
    filter_sizes = np.array([2,3,4],dtype=np.int32);num_filters=2
    for i,filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" %filter_size):
        #Convolution Layer
        filter_shape = [filter_size,2*self.args.rnn_size,1,num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
        b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
        
        conv = tf.nn.conv2d(
            self.embedding_x_expanded,
            W,
            strides=[1,1,1,1],
            padding='VALID',
            name='conv')  
        #Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
        #Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1,5-filter_size+1,1,1],
            strides=[1,1,1,1],
            padding='VALID',
            name='pool')
        pooled_outputs.append(pooled)
        
    #combin all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    self.h_pool = tf.concat(3,pooled_outputs)
    print 'h_pool:',self.h_pool
    self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])
    print 'self.h_pool_flat:',self.h_pool_flat
    
    self.ent_mention_lstm_feature_l2 = tf.expand_dims(self.h_pool_flat,-1)
    print 'self.ent_mention_lstm_feature_l2:',self.ent_mention_lstm_feature_l2
    '''
    '''
    @2016/12/22
    '''
    with tf.device('/gpu:0'):
      temp_descrip = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_descrip),self.candidate_ent_linking_feature,initializer=tf.Variable(tf.random_normal((self.candidateEntNum,self.lowdim))))
      self.prediction_descrip = tf.nn.sigmoid(tf.einsum('aij,ajk->aik',temp_descrip,self.ent_mention_lstm_feature_l2)[:,:,0])
      
    with tf.device('/gpu:1'):
      '''
      @transfer the result into probability!
      '''
      temp_mentWordV = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_mentWordv),self.ent_surfacewordv_feature,initializer=tf.Variable(tf.random_normal((self.candidateEntNum,self.lowdim))))
      print 'temp_mentWordV:',temp_mentWordV
      self.prediction_mentWordV = tf.einsum('aij,ajk->aik',temp_mentWordV,self.ent_mention_lstm_feature_l2)[:,:,0]
      print 'self.prediction_mentWordV:',self.prediction_mentWordV
      #self.prediction_mentWordV = tf.nn.sigmoid(tf.batch_matmul(temp_mentWordV,self.ent_mention_lstm_feature_l2)[:,:,0])
      
      
      temp_type = tf.scan(lambda a,x: tf.matmul(x,self.bilinear_w_type),self.candidate_ent_type_feature,initializer=tf.Variable(tf.random_normal((self.candidateEntNum,self.lowdim))))
      self.prediction_type = tf.nn.sigmoid(tf.einsum('aij,ajk->aik',temp_type,self.ent_mention_lstm_feature_l2)[:,:,0])
      self.prediction_prob = tf.nn.sigmoid(tf.scan(lambda a,x: tf.matmul(x,self.w_prob)+self.bias_prob,self.candidate_ent_prob_feature,initializer=tf.Variable(tf.random_normal((self.candidateEntNum,1))))[:,:,0])
      
      '''need to imporve the features'''
      self.prediction =  tf.nn.softmax(tf.add(self.w1*self.prediction_prob,tf.add(tf.add(self.w2*self.candidate_ent_coherent_feature_ngd,self.w2_1*self.candidate_ent_coherent_feature_fb),tf.add(self.w3*self.prediction_descrip,tf.add(self.w4*self.prediction_type,self.w5*self.prediction_mentWordV)))))
      print 'ctx prediction:',self.prediction
      print 'ctxSum ent_mention_linking_tag', self.ent_mention_linking_tag
    
      self.linking_loss  = tf.reduce_mean(-tf.reduce_sum(self.ent_mention_linking_tag * tf.log(self.prediction+1e-9), reduction_indices=[1]))
      #self.linking_loss = tf.reduce_mean(tf.contrib.losses.hinge_loss(self.prediction,self.ent_mention_linking_tag))
      print 'linking_loss:', self.linking_loss
    
      correct_predict = tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.ent_mention_linking_tag,1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))