import tensorflow as tf
import numpy as np

class seqLSTM(object):
  '''
  args: hidden_size,sequence_length,wordsEmbed,tagsEmbed
  '''
  def __init__(self,):
    
    self.sents = tf.placeholder(tf.int32,[None,sequence_length])
    self.inputs_length = tf.placeholder(tf.int32,[self.batch_size])
    
    self.embed_sents = tf.nn.embedding_lookup(wordsEmbed,self.sents)
    
    self.tags = tf.placeholder(tf.int32,[self.batch_size,sequence_length])
    