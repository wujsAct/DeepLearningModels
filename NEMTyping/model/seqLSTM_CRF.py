# -*- coding: utf-8 -*-
'''
@time: 2016/11/30
@editor: wujs
'''
import tensorflow as tf
from base_model import Model
import time

class seqLSTM_CRF(Model):
  '''
  self.args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    '''
    @time: 2016/12/20
    @editor: wujs
    @function: also need to return the candidates entity mentions lstm representation
    '''
    super(seqLSTM_CRF, self).__init__()
    self.args = args
    self.input_data = tf.placeholder(tf.float32,[None,self.args.sentence_length,self.args.word_dim])
    #need add
    #self.output_data = tf.placeholder(tf.int32,[None,self.args.sentence_length,self.args.class_size])
    self.output_data = tf.placeholder(tf.int32,[None,self.args.sentence_length])
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_NER')
    self.num_examples = tf.placeholder(tf.int32,name='num_examples')
    self.batch_size = self.args.batch_size
    if self.args.dropout:
      self.input_data =  tf.nn.dropout(self.input_data,self.keep_prob)
      
    with tf.variable_scope("seqLSTM_variables") as scope:
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
      start_time = time.time()
      print 'start to do lstm'
      #get sentence length
      used = tf.sign(tf.reduce_max(tf.abs(self.input_data),reduction_indices=2))
      self.length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
      print 'self.length:',self.length
      output,_,_ = tf.nn.bidirectional_rnn(fw_cell,bw_cell,
                                          tf.unpack(tf.transpose(self.input_data,perm=[1,0,2])),
                                          dtype=tf.float32,sequence_length=self.length
                                          )
      print 'bi-lstm cost time:', time.time()-start_time
      if self.args.dropout:
        output =  tf.nn.dropout(output,self.keep_prob)
      
      #features 
      self.output = tf.transpose(tf.pack(output),perm=[1,0,2])
      print self.output
      
      
      matricized_x_t  = tf.reshape(self.output,[-1,2*self.args.rnn_size])
      matricized_unary_scores = tf.matmul(matricized_x_t,self.crf_weights)
      
      
      self.unary_scores = tf.reshape(matricized_unary_scores,
                             [self.num_examples,self.args.sentence_length,self.args.class_size])
      print 'unary_scores:',self.unary_scores
      print 'self.output_data:',self.output_data
      print 'self.length:',self.length
      self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        self.unary_scores, self.output_data, self.length)
      
      self.loss = tf.reduce_mean(-self.log_likelihood)
      
      '''
      @revise time: 2017/2/19 add crf layer to predict sequence label!
      '''
      
#      output_f = tf.reshape(self.output,[-1,2*self.args.rnn_size])
#      print output_f      
#      W = tf.get_variable(
#                "W",
#                shape=[2*self.args.rnn_size, self.args.class_size],
#                initializer=tf.contrib.layers.xavier_initializer())
#      b = tf.Variable(tf.constant(0.1,shape=[self.args.class_size]),name="b")
#      prediction = tf.nn.softmax(tf.nn.xw_plus_b(output_f,W,b,name="scores"))
#      self.prediction = tf.reshape(prediction,[-1,self.args.sentence_length,self.args.class_size])
#      
#      cross_entropy = self.output_data * tf.log(self.prediction)
#      cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
#      mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
#      cross_entropy *= mask
#      cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
#      cross_entropy /= tf.cast(self.length, tf.float32)
#      self.loss = tf.reduce_mean(cross_entropy)

      