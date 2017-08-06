# -*- coding: utf-8 -*-
"""
Created on Fri May 05 14:47:15 2017

@author: wujs
@function: generate MLP,LSTM layers
"""
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse  as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell,LSTMStateTuple,GRUCell
from tensorflow.python.framework import dtypes
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder,decoder
import helpers
from tensorflow.python.layers.core import Dense

class Seq2SeqModel():
  '''
  code from: https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/model_new.py
  '''
  PAD=0; EOS=1
  def __init__(self,encoder_cell,decoder_cell,vocab_size,embedding_size,
               bidirectional=True,
               attention=False,
               debug=False,
               is_inference=False):
    self.debug = debug
    self.bidirectional = bidirectional
    self.attention = attention
    self.is_inference = is_inference
    
    
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    
    self.encoder_cell = encoder_cell
    self.decoder_cell = decoder_cell
    
    self._make_graph()
  
  def _make_graph(self):
    if self.debug:
      self._init_debug_inputs()
    else:
      self._init_placeholder()
    
    self._init_decoder_train_connectors()
    self._init_embeddings()
    
    if self.bidirectional:
      self._init_bidirectional_encoder()
    else:
      self._init_simple_encoder()
    
    self._init_decoder()
  
  @property
  def decoder_hidden_units(self):
    return self.decoder_cell.output_size
  
  def _init_debug_inputs(self):
    '''
    Time major
    '''
    x = [[5,6,7],[7,6,0],[0,7,0]]
    x1 = [3,2,1]
    self.encoder_inputs = tf.constant(x,dtype=tf.int32,name='encoder_inputs')
    self.encoder_inputs_length = tf.constant(x1,dtype=tf.int32,name='encoder_inputs_length')
    
    
    self.decoder_targets=tf.constant(x,dtype=tf.int32,name='decoder_targets')
    self.decoder_targets_length = tf.constant(x1,dtype=tf.int32,name='decoder_targets_length')
    
  def _init_placeholder(self):
    ''''time major'''
    self.encoder_inputs = tf.placeholder(
        shape=(None,None),
        dtype=tf.int32,
        name='encoder_inputs'
    )
    
    self.encoder_inputs_length = tf.placeholder(
        shape=(None,),
        dtype=tf.int32,
        name='encoder_inputs_length'
    )
    
    #require training not require for testing
    self.decoder_targets = tf.placeholder(
        shape=(None,None),
        dtype=tf.int32,
        name='decoder_targets'
    )
    
    self.decoder_targets_length = tf.placeholder(
        shape=(None,),
        dtype=tf.int32,
        name='decoder_targets_lengths'
    )
    
  def _init_decoder_train_connectors(self):
    with tf.name_scope('DecoderTrainFeeds'):
      sequence_size,batch_size = tf.unstack(tf.shape(self.decoder_targets))
      self.EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
      self.PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

      self.decoder_train_inputs = tf.concat([self.EOS_SLICE, self.decoder_targets], axis=0)
      self.decoder_train_length = self.decoder_targets_length + 1

      decoder_train_targets = tf.concat([self.decoder_targets, self.PAD_SLICE], axis=0)
      self.decoder_train_targets_seq_len,_= tf.unstack(tf.shape(decoder_train_targets))
      decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                  self.decoder_train_targets_seq_len,
                                                  on_value=self.EOS, off_value=self.PAD,
                                                  dtype=tf.int32)
      self.decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
      
      self.temp_decoder_train_targets = decoder_train_targets
      # hacky way using one_hot to put EOS symbol at the end of target sequence
      decoder_train_targets = tf.add(decoder_train_targets,
                                     self.decoder_train_targets_eos_mask)

      self.decoder_train_targets = decoder_train_targets

      self.loss_weights = tf.ones([
          batch_size,
          tf.reduce_max(self.decoder_train_length)
      ], dtype=tf.float32, name="loss_weights")
  
  def _init_embeddings(self):
    with tf.variable_scope('embedding') as scope:
      #we can also replace this by our pre-trained embedding_matrix
      self.embedding_matrix = tf.get_variable(
          name = 'embedding_matrix',
          shape=[self.vocab_size,self.embedding_size],
          initializer=tf.contrib.layers.xavier_initializer(),
          dtype=tf.float32)
      
      self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix,self.encoder_inputs)
      
      self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix,self.decoder_train_inputs)
      
      print 'decoder_train_inputs_embedded:',self.decoder_train_inputs_embedded
      
  def _init_simple_encoder(self):
    with tf.variable_scope("Encoder") as scope:
      (self.encoder_outputs, self.encoder_state) = (
          tf.nn.dynamic_rnn(cell=self.encoder_cell,
                            inputs=self.encoder_inputs_embedded,
                            sequence_length=self.encoder_inputs_length,
                            time_major=True,
                            dtype=tf.float32)
          )
     
  def _init_bidirectional_encoder(self):
    with tf.variable_scope('BidirectionalEncoder') as scope:
      ((encoder_fw_outpus,
        encoder_bw_outputs),
        (encoder_fw_state,
         encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                              cell_bw=self.encoder_cell,
                                                              inputs=self.encoder_inputs_embedded,
                                                              sequence_length=self.encoder_inputs_length,
                                                              time_major=True,
                                                              dtype=tf.float32)
        
        
      self.encoder_outputs = tf.concat((encoder_fw_outpus,encoder_bw_outputs),2)
      
      if isinstance(encoder_fw_state,LSTMStateTuple):
        encoder_state_c = tf.concat((encoder_fw_state.c,encoder_bw_state.c),1,name='bidirectional_concat_c')
        encoder_state_h = tf.concat((encoder_fw_state.h,encoder_bw_state.h),1,name='bidirectioanl_concat_h')
        
        self.encoder_state =LSTMStateTuple(c=encoder_state_c,h=encoder_state_h)
        
      elif isinstance(encoder_fw_state,tf.Tensor):
        self.encoder_state = tf.concat((encoder_fw_state,encoder_bw_state),1,name='bidirectional_concat')
        
  def _init_decoder(self):
    with tf.variable_scope('Decoder') as scope:

      self.fc_layer = Dense(self.vocab_size)
      
      if self.is_inference:
        self.start_tokens = tf.placeholder(tf.int32,shape=[None],name='start_tokens')
        self.end_token = tf.placeholder(tf.int32,name='end_token')
        
      
        self.helper = seq2seq.GreedyEmbeddingHelper(
            embedding=self.embedding_matrix,
            start_tokens=self.start_tokens,
            end_token=self.end_token
        )
      else:
        self.helper = seq2seq.TrainingHelper(
            inputs=self.decoder_train_inputs_embedded, 
            sequence_length=self.decoder_train_length,
            time_major=True
        )
      
      self.decoder = seq2seq.BasicDecoder(
          cell=self.decoder_cell,
          helper=self.helper,
          initial_state=self.encoder_state,
          output_layer=self.fc_layer
      )
      
      
      (self.decoder_outputs_train,
       self.decoder_state_train,
       self.decoder_context_state_train
       ) = (
           decoder.dynamic_decode(
               self.decoder, 
               output_time_major=True)
      )
      self.logits = self.decoder_outputs_train.rnn_output
      if not self.is_inference:
        self.decoder_prediction_inference = tf.argmax(self.logits, axis=-1, name='decoder_prediction_inference')
      
        self.decoder_prediction_train = tf.argmax(self.decoder_outputs_train.rnn_output, axis=-1, name='decoder_prediction_train')
        
        self._init_optimizer()
      else:
        self.prob = tf.nn.softmax(self.logits)
        
  def _init_optimizer(self):
    self.targets = tf.transpose(self.decoder_train_targets,[1,0])
    self.logits  = tf.nn.softmax(tf.transpose(self.logits, [1,0,2]))
    print 'targets:',self.targets
      
    print 'logits:',self.logits
    
    
    
    self.loss = seq2seq.sequence_loss(logits=self.logits, targets=self.targets,
                                          weights=self.loss_weights)
    print 'self.loss:',self.loss
    # define train op
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)

    optimizer = tf.train.AdamOptimizer(1e-3)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
  def make_train_inputs(self, input_seq, target_seq):
    inputs_, inputs_length_ = helpers.batch(input_seq)

    targets_, targets_length_ = helpers.batch(target_seq)
    return {
        self.encoder_inputs: inputs_,
        self.encoder_inputs_length: inputs_length_,
        self.decoder_targets: targets_,
        self.decoder_targets_length: targets_length_,
    }

  def make_inference_inputs(self, input_seq):
    inputs_, inputs_length_ = helpers.batch(input_seq)
    return {
        self.encoder_inputs: inputs_,
        self.encoder_inputs_length: inputs_length_,
    }
    
class CNN(object):
  '''
  CNN layer for Text
  '''
  def __init__(self,filters,word_embedding_size,num_filters):
    self.filters = filters
    self.embedding_size  = word_embedding_size
    self.num_filters = num_filters
    self.Ws = []
    self.bs = []
    for i,filter_size in enumerate(self.filters):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape =[filter_size,self.embedding_size,1,self.num_filters]
        self.Ws.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="rnn_weight"+str(i)))
        self.bs.append(tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="rnn_bias"+str(i)))
  '''
  x: [batch,sequence_length,feature_dimension,1] === >[batch,in_height,in_width,in_channels]
  filter: [filter_height,filter_widht,in_channels,out_channels]
  '''
  def __call__(self,x,sequence_length):
    self.pooled_outputs = []
    for i,filter_size in enumerate(self.filters):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        #convolution layer
        W = self.Ws[i]; b = self.bs[i]
        conv = tf.nn.conv2d(
            x,
            W,
            strides=[1,1,1,1],
            padding='VALID',
            name='conv'+str(i))
        print conv
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        print h
         # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize = [1, sequence_length-filter_size + 1, 1, 1],
            strides=[1,2,2,1],
            padding='VALID',
            name='pool')
        
    
        self.pooled_outputs.append(pooled)
    print 'self.pooled_outputs:',self.pooled_outputs
    #Combine all the pooled features
    num_filters_total = self.num_filters * len(self.filters) #?a little weird
    self.h_pool = tf.concat(self.pooled_outputs,3)
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
    
    return self.h_pool_flat
        
class LSTM(object):
  def __init__(self,cell_size,num_layers=1,name='LSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.reuse = None
    self.trainable_weights = None
    self.name = name
  def __call__(self,x,keep_prob=1.0,seq_length=None):
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      self.cell =tf.contrib.rnn.LSTMCell(self.cell_size,reuse=tf.get_variable_scope().reuse)
     
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
      
      self.output,_=tf.contrib.rnn.static_rnn(self.cell,tf.unstack(tf.transpose(x,[1,0,2])),dtype=tf.float32,sequence_length=seq_length)
      
      lstm_out = tf.transpose(tf.stack(self.output),[1,0,2])
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
    self.reuse = True
    
    return lstm_out,seq_length
      
    
class BiLSTM(object):
  '''
  LSTM layers using dynamic rnn
  '''
  def __init__(self,cell_size,num_layers=1,name='BiLSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.reuse = None
    self.trainable_weights = None
    self.name = name
  
  #x() equals to x.__call___()
  def __call__(self,x,keep_prob=1.0,seq_length=None):  #__call__ is very efficient when the state of instance changes frequently 
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      self.fw_cell =tf.contrib.rnn.LSTMCell(self.cell_size,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
      self.fw_cell1 =tf.contrib.rnn.LSTMCell(self.cell_size,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
      
      self.bw_cell =tf.contrib.rnn.LSTMCell(self.cell_size,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
      self.bw_cell1 =tf.contrib.rnn.LSTMCell(self.cell_size,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
  
  
      self.fw_cells = tf.contrib.rnn.MultiRNNCell([self.fw_cell,self.fw_cell1],state_is_tuple=True)
      self.bw_cells = tf.contrib.rnn.MultiRNNCell([self.bw_cell,self.bw_cell1],state_is_tuple=True)
      
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
      
      lstm_out,_,_ =  tf.contrib.rnn.static_bidirectional_rnn(self.fw_cells,self.bw_cells,tf.unstack(tf.transpose(x,[1,0,2])),dtype=tf.float32,sequence_length=seq_length)
      
      lstm_out = tf.transpose(tf.stack(lstm_out),[1,0,2])
      print 'lstm_out: ',lstm_out
      
      #shape(lstm_out) = (self.batch_size,sequence_length,2*cell_size)
      
      #if keep_prob < 1.:
      #  lstm_out = tf.nn.dropout(lstm_out,keep_prob)
        
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
        
    self.reuse =True
    return lstm_out,seq_length

class FullyConnection(object):
  def __init__(self,output_size,name='FullyConnection'):
    self.output_size = output_size
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    
  def __call__(self,inputs,activation_fn):
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      out = tf.contrib.layers.fully_connected(inputs,self.output_size, activation_fn=activation_fn
                                           )
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
    self.reuse =True

    return out
  
class CRF(object):
  def __init__(self,output_size,name='CRF'):
    self.output_size = output_size
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    
  def __call__(self,inputs,output_data,length):
    with tf.variable_scope(self.name,reuse =self.reuse) as vs: 
      self.log_likelihood,self.transition_params  = tf.contrib.crf.crf_log_likelihood(inputs,output_data,length)
      self.loss = tf.reduce_mean(-self.log_likelihood)
      
      if self.reuse == None:
        self.trainable_weights = vs.global_variables()
          
    self.reuse = True
    
    return self.transition_params,self.loss
  
'''
There are a lot of loss function defined in tensorflow!
'''
def classification_loss(flag,labels,logits):
  if flag == 'figer':
    loss = tf.losses.mean_pairwise_squared_error(labels,logits)
  elif flag=='sigmoid':
    loss = tf.losses.sigmoid_cross_entropy(labels,logits)
  elif flag =='hinge':
    loss = tf.losses.hinge_loss(labels,logits)
  else:
    loss = tf.losses.softmax_cross_entropy(labels,logits)  #must one-hot entropy
  
  return loss