import tensorflow as tf
import numpy as np
class CtxCNN(object):
  """
  convolution and max pooling for entity contexts information!
  """
  def __init__(self,sequence_length,vocab_size,embedding_size,filter_sizes,num_filters):
    self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
    #embedding layer
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
      W = tf.Variable(tf.random_uniform([vocab_size,embedding_size]),name='W')
      self.embedding_x = tf.nn.embedding_lookup(W,self.input_x)
      self.embedding_x_expanded = tf.expand_dims(self.embedding_x,-1) #扩展维度，在最后一行加1，表示1个通道
    
    pooled_outputs = []
    for i,filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" %filter_size):
        #Convolution Layer
        filter_shape = [filter_size,embedding_size,1,num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
        b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
        
        conv = tf.nn.conv2d(
            self.embedding_x_expanded,
            W,
            strides=[1,1,1,1]
            padding='VALID',
            name='conv')  
        #Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
        #Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1,sequence_length-filter_size+1,1,1],
            strides=[1,1,1,1],
            padding='VALID',
            name='pool')
        pooled_outputs.append(pooled)
        
    #combin all the pooled features
    num_filters_total = num_filters * len(filter_size)
    self.h_pool = tf.concat(3,pooled_outputs)
    self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])
    
    #Add dropout
    with tf.name_scope("dropout"):
      self.h_drop = tf.nn.dropout(self.h_pool_flat,self.keep_prob)
      