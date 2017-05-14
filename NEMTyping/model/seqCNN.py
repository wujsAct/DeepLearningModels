# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:18:25 2017

@author: wujs
"""

import tensorflow as tf
from base_model import Model
import time
import numpy as np
import cPickle
import layers as layers_lib
import adversarial_losses as adv_lib

class seqCNN(Model):
  '''
  self.args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    '''
    @time: 2016/12/20
    @editor: wujs
    @function: also need to return the candidates entity mentions lstm representation
    '''
    super(seqCNN, self).__init__()
    self.args = args
    self.batch_size=args.batch_size

    self.input_data = tf.placeholder(tf.float32,[self.args.batch_size,self.args.sentence_length,self.args.word_dim],name='inputdata')
    print 'self.input_data:',self.input_data
    
    self.output_data = tf.sparse_placeholder(tf.float32, name='outputdata')
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    
    self.pos_f1 = tf.placeholder(tf.float32,[None,5,1])
    self.pos_f2 = tf.placeholder(tf.float32,[None,10,1])
    self.pos_f3 = tf.placeholder(tf.float32,[None,10,1])
    
    self.entMentIndex = tf.placeholder(tf.int32,[None,5],name='ent_mention_index')
    
    self.entCtxLeftIndex = tf.placeholder(tf.int32,[None,10],name='ent_ctxleft_index')
    self.entCtxRightIndex = tf.placeholder(tf.int32,[None,10],name='ent_ctxright_index')
     
    #self.hier = np.asarray(cPickle.load(open('data/figer/figerhierarchical.p','rb')),np.float32)  #add the hierarchy features
    self.hier = np.asarray(cPickle.load(open('data/OntoNotes/OntoNoteshierarchical.p','rb')),np.float32)
    
    print np.shape(self.hier)[0]
    
    self.layers={}
    self.layers['CNN'] = layers_lib.CNN(filters=[1,2,3],word_embedding_size=self.args.word_dim+1,num_filters=5)
    self.layers['fullyConnect_final'] = layers_lib.FullyConnection(self.args.class_size) # 90 is the row of type hierical 
    
        
    self.dense_outputdata= tf.sparse_tensor_to_dense(self.output_data)
    
    print 'self.dense_outputdata:', self.dense_outputdata
    
    self.prediction,self.loss_lm = self.cl_loss_from_embedding(self.input_data)
    print 'self.loss_lm:',self.loss_lm
      
#    _,self.adv_loss = self.adversarial_loss()
#    print 'self.adv_loss:',self.adv_loss
#
#    self.loss = tf.add(self.loss_lm,self.adv_loss)
    self.loss = self.loss_lm 
    
  def cl_loss_from_embedding(self,embedded,return_intermediate=False):
    with tf.device('/gpu:1'):
      #output,_,_ = self.layers['BiLSTM'](embedded)
      output = tf.concat([tf.reshape(self.input_data,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
      
    self.input_f1 = tf.nn.embedding_lookup(output,self.entMentIndex)
    self.input_f2 = tf.nn.embedding_lookup(output,self.entCtxLeftIndex)
    self.input_f3 = tf.nn.embedding_lookup(output,self.entCtxRightIndex)
    
   
    self.input_f1 = tf.concat([self.input_f1,self.pos_f1],-1)
    print 'input_f1:',self.input_f1
    
    
    self.input_f2 = tf.concat([self.input_f2,self.pos_f2],-1)
    print 'input_f2:',self.input_f2
    
    
    
    self.input_f3 = tf.concat([self.input_f3,self.pos_f3],-1)
    
    
    self.h_pool_f1 = self.layers['CNN'](tf.expand_dims(self.input_f1,-1),5)
    print 'h_pool_f1:',self.h_pool_f1
    
    self.h_pool_f2 = self.layers['CNN'](tf.expand_dims(self.input_f2,-1),10)
    print 'h_pool_f2:',self.h_pool_f2
    
    self.h_pool_f3 = self.layers['CNN'](tf.expand_dims(self.input_f3,-1),10)
    print 'h_pool_f3:',self.h_pool_f3
    
    self.final_input = tf.concat([self.h_pool_f1,self.h_pool_f2,self.h_pool_f3],1)
    
    print 'final_input:',self.final_input
    
    
    prediction = self.layers['fullyConnect_final'](self.final_input,activation_fn=tf.nn.tanh)
      
    loss = tf.reduce_mean(layers_lib.classification_loss('figer',self.dense_outputdata,prediction))
    return prediction,loss
  
  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    return adv_lib.adversarial_loss(self.input_data,
                                      self.loss_lm,
                                      self.cl_loss_from_embedding) 

      