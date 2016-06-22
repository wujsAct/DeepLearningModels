# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 20:10:36 2016

@author: DELL
"""

import tensorflow as tf
import os
from base_model import Model
from deep_bi_lstm import DeepBiLSTM
from data_utils import QADataset
import time
import numpy as np

class AttentiveReader(Model):
  """Attentive Reader."""
  def __init__(self, hidden_size=256,
               query_n_input=50,context_n_input=1000,
               learning_rate=1e-4, batch_size=32,cand_n_input=50,
               dropout=0.1, max_time_unit=100,checkpoint_dir="checkpoint",forward_only=False):
    
    super(AttentiveReader, self).__init__()
    self.checkpoint_dir = checkpoint_dir
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.dropout = dropout
    self.max_time_unit = max_time_unit
    self.query_n_input = query_n_input
    self.ctx_n_input = context_n_input
    self.ctx_output_size =hidden_size
    self.q_ouput_size = hidden_size
    self.CtxQ_emb_lent=hidden_size
    self.ctx_emb_lent = hidden_size
    self.CtxQ_linearJoint_emb_lent = hidden_size
    #self.cand_n_input=cand_n_input
  
    
  def prepare_model(self,data_dir,dataset_name):
    with tf.device("/cpu:0"):
      vocab_fname = os.path.join(data_dir,dataset_name,dataset_name+'.vocab')
      qaData_t = QADataset(data_dir,dataset_name,vocab_fname)
      if not self.vocab:
        self.vocab,self.n_entities = qaData_t.initialize_vocabulary()
        print(" [*] Loading vocab finished.")
        
    self.vocab_size = len(self.vocab)
    self.emb = tf.Variable(tf.truncated_normal(
          [self.vocab_size, self.hidden_size], -0.1, 0.1), name='emb')
    self.ctx_BiLSTM_inputs =tf.placeholder(tf.int32,[self.batch_size,self.ctx_n_input])
    self.ctx_inputs_sequence = tf.placeholder(tf.int32,[self.batch_size])
    
    embed_inputs_ctx = tf.nn.embedding_lookup(self.emb, tf.transpose(self.ctx_BiLSTM_inputs))
    with tf.device("/gpu:0"):
      print 'embed_inputs_ctx',embed_inputs_ctx
      self.lstm_fw_cell_ctx = tf.nn.rnn_cell.LSTMCell(self.hidden_size,state_is_tuple=True)
      self.lstm_bw_cell_ctx = tf.nn.rnn_cell.LSTMCell(self.hidden_size,state_is_tuple=True)
      
      self.ctx_outputs_,_, _ = tf.nn.bidirectional_rnn(self.lstm_fw_cell_ctx,
                                                    self.lstm_bw_cell_ctx ,
                                                    tf.unpack(embed_inputs_ctx),
                                                    dtype=tf.float32,
                                                    #sequence_length=self.ctx_inputs_sequence,
                                                    scope='ctx_scope')  
      
    self.query_BiLSTM_inputs = tf.placeholder(tf.int32,[self.batch_size,self.query_n_input])
    self.query_inputs_sequence = tf.placeholder(tf.int32,[self.batch_size])    
    embed_inputs_q = tf.nn.embedding_lookup(self.emb, tf.transpose(self.query_BiLSTM_inputs))                                       
    with tf.device("/gpu:1"):
      print 'embed_inputs_q:',embed_inputs_q
      self.lstm_fw_cell_q = tf.nn.rnn_cell.LSTMCell(self.hidden_size,state_is_tuple=True)
      self.lstm_bw_cell_q = tf.nn.rnn_cell.LSTMCell(self.hidden_size,state_is_tuple=True)     
      self.q_outputs_,_,_ = tf.nn.bidirectional_rnn(self.lstm_fw_cell_q,
                                                      self.lstm_bw_cell_q,
                                                      tf.unpack(embed_inputs_q),
                                                      dtype=tf.float32,
                                                      #sequence_length=self.query_inputs_sequence,
                                                      scope='q_scope')
    #query outputs u = y(q) || y(1), utilize the op slice to get the result we want     
    self.u_final_real=tf.pack([tf.concat(1,[tf.slice(self.q_outputs_,[lents-1,idx,0],[1,1,self.q_ouput_size])[:,0,:],tf.slice(self.q_outputs_,[0,idx,self.q_ouput_size],[1,1,self.q_ouput_size])[:,0,:]])[0,:]
                         for idx, lents in enumerate(tf.unpack(self.query_inputs_sequence))])
    with tf.device("/gpu:0"):                    
      self.embedCtxQuery={
             'Wrg': tf.Variable(tf.truncated_normal([self.ctx_emb_lent*2,self.hidden_size*2],-0.1,0.1),dtype=tf.float32),
             'Wug': tf.Variable(tf.truncated_normal([self.q_ouput_size*2,self.hidden_size*2],-0.1,0,1),dtype=tf.float32)
      }
      
      #print 'u_final_real:',self.u_final_real[0:0+1,:]
      #ctx outputs:self.ctx_outputs_[:lents,idx,:]
      temp = tf.constant([[0]*(self.q_ouput_size*2)]*self.ctx_n_input,dtype=tf.float32)
      
      self.ctx_outputs_real=tf.pack([tf.concat(0,[tf.slice(self.ctx_outputs_,[0,idx,0],[lents,1,self.q_ouput_size*2])[:,0,:],tf.slice(temp,[0,0],[self.ctx_n_input-lents,self.q_ouput_size*2])])
                           for idx, lents in enumerate(tf.unpack(self.ctx_inputs_sequence))])
      print self.ctx_outputs_real
    
      self.m = [tf.tanh(tf.matmul(tf.slice(self.u_final_real,[i,0],[1,self.q_ouput_size*2]),tf.transpose(tf.slice(self.ctx_outputs_real,[i,0,0],[1,self.ctx_n_input,self.ctx_output_size*2])[0,:,:])))
                for i in range(self.batch_size)]
      #print 'self.m',self.m
      self.s = tf.pack([tf.nn.softmax(self.m[i]) for i in range(self.batch_size)])
      print 'self.s',self.s
    
      self.r =tf.pack([ tf.nn.softmax(tf.matmul(self.s[i,:,:],self.ctx_outputs_real[i,:,:]))  for i in range(self.batch_size)])
      print 'self.r',self.r
      self.g = tf.pack([tf.add(tf.transpose(tf.tanh(tf.matmul(self.r[i,:,:],self.embedCtxQuery['Wrg']))), tf.matmul(self.embedCtxQuery['Wug'],tf.transpose(self.u_final_real[i:i+1,:])))
                for i in range(self.batch_size)])
      print 'self.g',self.g
    
    with tf.device("/gpu:1"):
      self.outputs= self.g[:,:,0]
      self.weights = {
                'out': tf.Variable(tf.random_normal([self.hidden_size*2,self.n_entities]))
        }
      self.biases = {
                'out': tf.Variable(tf.random_normal([self.n_entities]))
        }
        
      self.y = tf.placeholder(tf.int64,[self.batch_size]) 
       
      self.y_ = tf.add(tf.matmul(self.outputs, self.weights['out']),self.biases['out'])
        
        
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_,self.y)  #This function was added after release 0.6.0!
      correct_prediction = tf.equal(self.y, tf.argmax(self.y_, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      
    print(" [*] Preparing model finished.")
                
    
  def train(self,sess,epoch=25,learning_rate=0.0001,data_dir='data',dataset_name='cnn'):
    '''
    2016/6/12
    '''
    
    self.prepare_model(data_dir,dataset_name)
    data_max_idx=280096
    start = time.clock()
    with tf.device("/gpu:1"):
      print(" [*] Calculating gradient and loss...")
      self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)  
      sess.run(tf.initialize_all_variables())
    print(" [*] Calculating gradient and loss finished. Take %.2fs" % (time.clock() - start))
    
    if self.load(sess, self.checkpoint_dir, dataset_name):
      print(" [*] Deep Bidirect LSTM checkpoint is loaded.")
    else:
      print(" [*] There is no checkpoint for this model.")
      
    #merged = tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter("tmp/deep", sess.graph)
    
    counter = 0
    start_time = time.time()
    for epoch_idx in xrange(epoch):
      vocab_fname = os.path.join(data_dir,dataset_name,dataset_name+'.vocab')
      qaData = QADataset(data_dir,dataset_name,vocab_fname)
      data_loader = qaData.load_dataset2(self.batch_size)
      while True:
        try:
          #start = time.time()
          ctx,ctx_seq_length,q,q_seq_length,y = data_loader.next()
          #print 'cnn:',time.time()-start
          _, cost, accuracy = sess.run([self.optim, self.loss, self.accuracy], 
                                                     feed_dict={self.ctx_BiLSTM_inputs:ctx,
                                                     self.ctx_inputs_sequence:ctx_seq_length,
                                                     self.query_BiLSTM_inputs:q,
                                                     self.query_inputs_sequence:q_seq_length,
                                                     self.y:y})
           
          if counter % 10 == 0:
            #writer.add_summary(summary_str, counter)
            #about 300s iteration!
            data_idx = (counter+1) * self.batch_size
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                % (epoch_idx, data_idx, data_max_idx, time.time() - start_time, np.mean(cost), accuracy))
          counter += 1
        except StopIteration:
          break
      self.save(sess, self.checkpoint_dir, dataset_name)
    
    
    
      
