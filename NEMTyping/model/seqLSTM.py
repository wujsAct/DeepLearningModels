# -*- coding: utf-8 -*-
'''
@time: 2016/11/30
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time
import numpy as np
import cPickle
import layers as layers_lib
import adversarial_losses as adv_lib

class seqLSTM(Model):
  '''
  self.args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    '''
    @time: 2016/12/20
    @editor: wujs
    @function: also need to return the candidates entity mentions lstm representation
    '''
    super(seqLSTM, self).__init__()
    self.args = args
    self.batch_size=args.batch_size

    self.input_data = tf.placeholder(tf.float32,[self.args.batch_size,self.args.sentence_length,self.args.word_dim],name='inputdata')
    
    
    self.output_data = tf.sparse_placeholder(tf.float32, name='outputdata')
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    
    self.entMentIndex = tf.placeholder(tf.int32,[None,None],name='ent_mention_index')
    
    self.entCtxLeftIndex = tf.placeholder(tf.int32,[None,None],name='ent_ctxleft_index')
    self.entCtxRightIndex = tf.placeholder(tf.int32,[None,None],name='ent_ctxright_index')
     
    self.figerHier = np.asarray(cPickle.load(open('data/figer/figerhierarchical.p','rb')),np.float32)  #add the hierarchy features
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    self.layers['fullyConnect_ment'] = layers_lib.FullyConnection(90) # 90 is the row of type hierical 
    self.layers['fullyConnect_ctx'] = layers_lib.FullyConnection(self.args.class_size) # 90 is the row of type hierical 
    
    self.dense_outputdata= tf.sparse_tensor_to_dense(self.output_data)
    
    with tf.device('/gpu:1'):
      self.prediction,self.loss_lm = self.cl_loss_from_embedding(self.input_data)
      print 'self.loss_lm:',self.loss_lm
      
    _,self.adv_loss = self.adversarial_loss()
    print 'self.adv_loss:',self.adv_loss

    self.loss = tf.add(self.loss_lm,self.adv_loss)
      
    
  def cl_loss_from_embedding(self,embedded,return_intermediate=False):
    output,_ = self.layers['BiLSTM'](embedded)
    output = tf.concat([tf.reshape(output,[-1,2*self.args.rnn_size]),tf.constant(np.zeros((1,2*self.args.rnn_size),dtype=np.float32))],0)
      
    input_f1 =tf.reduce_sum(tf.nn.embedding_lookup(output,self.entMentIndex),1)
    input_f2 = tf.reduce_sum(tf.nn.embedding_lookup(output,self.entCtxLeftIndex),1)
    input_f3 = tf.reduce_sum(tf.nn.embedding_lookup(output,self.entCtxRightIndex),1)
    input_ctx = tf.concat([input_f2,input_f3],1)
    
    if self.args.dropout:  #dropout position is here!
      input_f1 =  tf.nn.dropout(input_f1,self.keep_prob)
      input_ctx =  tf.nn.dropout(input_ctx,self.keep_prob)
        
    prediction_l1_ment = self.layers['fullyConnect_ment'](input_f1,activation_fn=tf.nn.tanh)
    prediction_metn = tf.matmul(prediction_l1_ment,self.figerHier)
    
    prediction_ctx = self.layers['fullyConnect_ctx'](input_ctx,activation_fn=tf.nn.tanh)
    
    prediction = prediction_metn + prediction_ctx
    loss = tf.reduce_mean(layers_lib.classification_loss('figer',self.dense_outputdata,prediction))
    return prediction,loss
  
  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    return adv_lib.adversarial_loss(self.input_data,
                                      self.loss_lm,
                                      self.cl_loss_from_embedding) 

      