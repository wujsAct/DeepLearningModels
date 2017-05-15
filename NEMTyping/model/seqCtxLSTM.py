# -*- coding: utf-8 -*-
'''
@time: 2016/11/30
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time
import cPickle
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn as contrib_rnn
import layers as layers_lib
import adversarial_losses as adv_lib


class seqCtxLSTM(Model):
  def __init__(self,args):
    super(seqCtxLSTM, self).__init__()
    self.args = args
    self.batch_size=args.batch_size
   
    self.input_data = tf.placeholder(tf.float32,[self.args.batch_size,self.args.sentence_length,self.args.word_dim],name='inputdata')

    
    self.output_data = tf.sparse_placeholder(tf.float32, name='outputdata')  #[None, 114]
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    
    self.entMentIndex = tf.placeholder(tf.int32,[None,5],name='ent_mention_index')
    
    self.entCtxLeftIndex = tf.placeholder(tf.int32,[None,10],name='ent_ctxleft_index')
    self.entCtxRightIndex = tf.placeholder(tf.int32,[None,10],name='ent_ctxright_index')
    
    self.pos_f1 = tf.placeholder(tf.float32,[None,5,1])
    self.pos_f2 = tf.placeholder(tf.float32,[None,10,1])
    self.pos_f3 = tf.placeholder(tf.float32,[None,10,1])
    
    self.figerHier = np.asarray(cPickle.load(open('data/figer/figerhierarchical.p','rb')),np.float32)  #add the hierarchy features
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    self.layers['fullyConnect'] = layers_lib.FullyConnection(self.args.class_size)
    
    self.dense_outputdata= tf.sparse_tensor_to_dense(self.output_data)
    

    with tf.device('/gpu:1'):
      self.prediction,self.loss_lm = self.cl_loss_from_embedding(self.input_data)
    
      print 'self.loss_lm:',self.loss_lm
    
      _,self.adv_loss = self.adversarial_loss()
      print 'self.adv_loss:',self.adv_loss

      self.loss = tf.add(self.loss_lm,self.adv_loss)
      
  
  def cl_loss_from_embedding(self,embedded,return_intermediate=False):
    self.reshape_input = tf.concat([tf.reshape(self.input_data,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
      
    input_f1 = tf.nn.l2_normalize(tf.reduce_sum(tf.nn.embedding_lookup(self.reshape_input,self.entMentIndex),1),1)
    print 'input_f1:',input_f1
  
    input_f2,_,_ =self.layers['BiLSTM'](tf.nn.embedding_lookup(self.reshape_input,self.entCtxLeftIndex))
    
    input_f2 = tf.nn.l2_normalize(tf.reduce_sum(input_f2,1),1)
    print 'input_f2:',input_f2
    
    input_f3,_,_ = self.layers['BiLSTM'](tf.nn.embedding_lookup(self.reshape_input,self.entCtxRightIndex))
    input_f3 = tf.nn.l2_normalize(tf.reduce_sum(input_f3,1),1)
    print 'input_f2:',input_f3
    
    
    self.input_total = tf.nn.sigmoid(tf.concat([input_f1,input_f2,input_f3],1))
  
    if self.args.dropout:
      self.input_total =  tf.nn.dropout(self.input_total,self.keep_prob)
        
        
    prediction = tf.nn.sigmoid(self.layers['fullyConnect'](self.input_total,tf.nn.tanh))

    loss = tf.reduce_mean(layers_lib.classification_loss('figer',self.dense_outputdata,prediction))
    
    return prediction,loss
  
  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""
    return adv_lib.adversarial_loss(self.input_data,
                                      self.loss_lm,
                                      self.cl_loss_from_embedding) 
        #elif tag=='LSTM':
#        output = tf.concat([output_f2,output_f3],1)
#        prediction_ctx = tf.contrib.layers.fully_connected(output,self.args.class_size, 
#                                           activation_fn=tf.nn.tanh,
#                                           )
#        
#        prediction_l1_ment = tf.contrib.layers.fully_connected(output_f1,90, 
#                                           activation_fn=tf.nn.tanh,
#                                           )
#      
#        prediction_metn = tf.matmul(prediction_l1_ment,self.figerHier)
#      
#      
#        self.prediction = prediction_ctx + prediction_metn
#        print self.prediction
#        
#        cross_entropy = tf.losses.mean_pairwise_squared_error(self.dense_outputdata,self.prediction)
#        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.dense_outputdata,logits=self.prediction)
#        self.loss = tf.reduce_mean(cross_entropy)
#        print self.loss
         
      
      

      