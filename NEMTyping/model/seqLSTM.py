# -*- coding: utf-8 -*-
'''
@time: 2016/11/30
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time
import numpy as np

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
    
    with tf.device('/gpu:1'):
      self.input_data = tf.placeholder(tf.float32,[self.batch_size,self.args.sentence_length,self.args.word_dim])
    #need add
    #self.output_data = tf.placeholder(tf.float32,[None,self.args.sentence_length,self.args.class_size])
    #self.output_data = tf.sparse_placeholder(tf.float32, [None,self.args.sentence_length,self.args.class_size], name='outputdata')
    #self.output_data = tf.placeholder(tf.int32,[None,self.args.sentence_length])
    
    self.output_data = tf.sparse_placeholder(tf.float32, name='outputdata')  #[None, 114]
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    self.num_examples = tf.placeholder(tf.int32,name='num_examples')

    self.entMentIndex = tf.placeholder(tf.int32,[None,None],name='ent_mention_index')
      
    with tf.variable_scope("seqLSTM_variables") as scope:
      with tf.device('/gpu:1'):
        self.crf_weights = tf.get_variable("crf_weights",
                                     shape=[2*self.args.rnn_size,self.args.class_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
        
        fw_cell = tf.nn.rnn_cell.LSTMCell(self.args.rnn_size,state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(self.args.rnn_size,state_is_tuple=True)
        
        if self.args.dropout:
          fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=self.keep_prob)
          bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=self.keep_prob)
  
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell]*self.args.num_layers,state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell]*self.args.num_layers,state_is_tuple=True)
       
        #get sentence length
        used = tf.sign(tf.reduce_max(tf.abs(self.input_data),reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
        output,_,_ = tf.nn.bidirectional_rnn(fw_cell,bw_cell,
                                            tf.unpack(tf.transpose(self.input_data,perm=[1,0,2])),
                                            dtype=tf.float32,sequence_length=self.length
                                            ,scope=scope)
      with tf.device('/cpu:0'):  #fix run on cpus
        if self.args.dropout:
          output =  tf.nn.dropout(output,self.keep_prob)
        
  
        output = tf.transpose(tf.pack(output),perm=[1,0,2],name='LSTM_output')
        
        self.output = tf.concat(0,[tf.reshape(output,[-1,2*args.rnn_size]),tf.constant([[0.0]*256])])
        print 'outputs:',self.output
#        '''
#        #we need to extract entity mention index; this method may cost a lot of memory!
#        '''
#        #we utilize mask to get entity mentions features
#        features = tf.scan(
#            lambda a,outputx:self.output*outputx,self.entMentIndex,initializer=tf.Variable(tf.random_normal((args.batch_size,self.args.sentence_length,2*self.args.rnn_size))))
#        
#        output_f = tf.reduce_mean(tf.reduce_sum(features,1),1)   #get all the entity mentions features!
#        
#        output_f = tf.nn.l2_normalize(output_f,1)
        '''
        @time: 2017/4/26 very efficient methods to get entity mentions embeddings!
        '''
        output_f = tf.reduce_mean(tf.nn.embedding_lookup(self.output,self.entMentIndex),1)  #reduce_sum vs. reduce_mean
        print 'output_f:',output_f
        if self.keep_prob == 1.0:
          phase = True
        else:
          phase = False
        prediction_l1 = tf.contrib.layers.fully_connected(output_f, self.args.rnn_size, 
                                             activation_fn=None,
                                             scope='layer1')
       
        prediction_l1 = tf.contrib.layers.batch_norm(prediction_l1, 
                                          center=True, scale=True, 
                                          is_training=phase,activation_fn=tf.nn.relu,
                                          scope=scope)
        
        prediction_l2 = tf.contrib.layers.fully_connected(prediction_l1, self.args.class_size, 
                                             activation_fn=None,
                                             scope='layer2')
        self.prediction = tf.nn.softmax(prediction_l2)

        print self.prediction
        
        
        cross_entropy = tf.sparse_tensor_to_dense(self.output_data) * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=[1])
        self.loss = tf.reduce_mean(cross_entropy)
        #tf.argmax(self.prediction,1)
        correct_prediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(tf.sparse_tensor_to_dense(self.output_data),1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
      
      

      