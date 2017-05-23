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
    
    if self.args.datasets =='figer':
      self.hier = np.asarray(cPickle.load(open('data/figer/figerhierarchical.p','rb')),np.float32)  
    else:
      self.hier = np.asarray(cPickle.load(open('data/OntoNotes/OntoNoteshierarchical.p','rb')),np.float32)
    self.pos_f1 = tf.placeholder(tf.float32,[None,5,1])
    self.pos_f2 = tf.placeholder(tf.float32,[None,10,1])
    self.pos_f3 = tf.placeholder(tf.float32,[None,10,1])
    print 'hier shape:',np.shape(self.hier)
    
    print 'self.args.rnn_size:',self.args.rnn_size
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    self.layers['fullyConnect_ment'] = layers_lib.FullyConnection(np.shape(self.hier)[0],name='fullyConnect_ment') # 90 is the row of type hierical 
    self.layers['fullyConnect_ctx'] = layers_lib.FullyConnection(self.args.class_size,name='fullyConnect_ctx') # 90 is the row of type hierical 
    
    self.dense_outputdata= tf.sparse_tensor_to_dense(self.output_data)
    
   
    self.prediction,self.loss_lm = self.cl_loss_from_embedding(self.input_data)
    print 'self.loss_lm:',self.loss_lm
      
    _,self.adv_loss = self.adversarial_loss()
    print 'self.adv_loss:',self.adv_loss

    self.loss = tf.add(self.loss_lm,self.adv_loss)
    #self.loss = self.loss_lm 
    
  def cl_loss_from_embedding(self,embedded,return_intermediate=False):
    with tf.device('/gpu:1'):
      output,_ = self.layers['BiLSTM'](embedded)
      output = tf.concat([tf.reshape(output,[-1,2*self.args.rnn_size]),tf.constant(np.zeros((1,2*self.args.rnn_size),dtype=np.float32))],0)
      
    input_f1 =tf.nn.l2_normalize(tf.reduce_sum(tf.nn.embedding_lookup(output,self.entMentIndex),1),1)
    
    #input_f2 =tf.nn.l2_normalize(tf.reduce_sum(tf.nn.embedding_lookup(output,self.entCtxLeftIndex),1),1)
    
    #input_f3 =tf.nn.l2_normalize(tf.reduce_sum(tf.nn.embedding_lookup(output,self.entCtxRightIndex),1),1)
    
    f2_temp = tf.nn.embedding_lookup(output,self.entCtxLeftIndex)
    f3_temp = tf.nn.embedding_lookup(output,self.entCtxRightIndex)
    
    f2_atten = tf.nn.softmax(tf.einsum('aij,ajk->aik', f2_temp, tf.expand_dims(input_f1,-1)),-1)  #Batch matrix multiplication
    f3_atten = tf.nn.softmax(tf.einsum('aij,ajk->aik', f3_temp, tf.expand_dims(input_f1,-1)),-1) 
    
    input_f2 = tf.einsum('aij,ajk->aik',tf.transpose(f2_temp,[0,2,1]),f2_atten)[:,:,0]
    input_f3 = tf.einsum('aij,ajk->aik',tf.transpose(f3_temp,[0,2,1]),f3_atten)[:,:,0]
    
    print 'f2_input:',input_f2
    print 'f3_input:',input_f3
    
    input_ctx = tf.concat([input_f2,input_f3],1)
    
    if self.args.dropout:  #dropout position is here!
      input_f1 =  tf.nn.dropout(input_f1,self.keep_prob)
      input_ctx =  tf.nn.dropout(input_ctx,self.keep_prob)
        
    prediction_l1_ment = self.layers['fullyConnect_ment'](input_f1,activation_fn=None)
    prediction_ment = tf.matmul(prediction_l1_ment,self.hier)
    
    print 'ment:',prediction_ment
    prediction_ctx = self.layers['fullyConnect_ctx'](input_ctx,activation_fn=None)
    print 'ctx:',prediction_ctx
    prediction = tf.nn.sigmoid(prediction_ment + prediction_ctx)
    
    loss = tf.reduce_mean(layers_lib.classification_loss('figer',self.dense_outputdata,prediction))
    return prediction,loss
  
  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    return adv_lib.adversarial_loss(self.input_data,
                                      self.loss_lm,
                                      self.cl_loss_from_embedding) 

      