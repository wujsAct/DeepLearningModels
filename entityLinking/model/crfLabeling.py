# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/2/17
"""

import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
  
flags = tf.app.flags
flags.DEFINE_integer("num_features",100,"num_features ")
flags.DEFINE_integer("num_tags",5,"crf tag numbers")
flags.DEFINE_integer("num_words",20,"sentence words lenght")
args = flags.FLAGS
class crfLabeling():
  '''
  args: parameters for all the entities mentions!
  '''
  def __init__(self,args):
    self.num_features = args.num_features
    self.num_tags = args.num_tags
    self.num_words = args.num_words
    
    self.x = tf.placeholder(tf.float32,[None,self.num_words,self.num_features])
    self.y = tf.placeholder(tf.int32,[None,self.num_words])
    
    self.sequence_lengths = tf.placeholder(tf.int32,[None])
    self.shape = tf.placeholder(tf.int32)
    '''
    self.bilinear_w_descrip = tf.get_variable(
                    "bilinear_w_description",
                    shape=[int(self.args.rawword_dim),2*self.args.rnn_size],
                    initializer=tf.contrib.layers.xavier_initializer())
    '''
    self.weights = tf.get_variable("crf_weights",
                                   shape=[self.num_features,self.num_tags],
                                   initializer=tf.contrib.layers.xavier_initializer())
    matricized_x_t  = tf.reshape(self.x,[-1,self.num_features])
    matricized_unary_scores = tf.matmul(matricized_x_t,self.weights)
     
    #deal with the variant shape
    self.unary_scores = tf.reshape(matricized_unary_scores,
                             [self.shape,self.num_words,self.num_tags])
    
    print 'self.unary_scores:',self.unary_scores
    
    # Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
    log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        self.unary_scores, self.y, self.sequence_lengths)
    
    self.loss = tf.reduce_mean(-log_likelihood)

def main(_):
  
  pp.pprint(args.__flags)
  
  
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  crfLabels = crfLabeling(args)
  
  num_examples  = 10
  # Random features.
  x = np.random.rand(num_examples, args.num_words, args.num_features).astype(np.float32)
  # Random tag indices representing the gold sequence.
  y = np.random.randint(args.num_tags, size=[num_examples, args.num_words]).astype(np.int32)
  print y
  train_op = tf.train.GradientDescentOptimizer(0.01).minimize(crfLabels.loss)
  sequence_lengths = np.full(num_examples,args.num_words,dtype=np.int32)
  
  sess.run(tf.global_variables_initializer())
  for i in range(3000):
    tf_unary_scores, tf_transition_params, _ = sess.run(
        [crfLabels.unary_scores, crfLabels.transition_params, train_op],
        {crfLabels.x:x,
         crfLabels.y:y,
         crfLabels.sequence_lengths:sequence_lengths,
         crfLabels.shape:num_examples})
    
    if i % 100 == 0:
      correct_labels = 0
      total_labels = 0
      for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,sequence_lengths):
        # Remove padding from the scores and tag sequence.
        tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
        y_ = y_[:sequence_length_]
        
        # Compute the highest scoring sequence.
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores_, tf_transition_params)
        #print viterbi_sequence
        # Evaluate word-level accuracy.
        correct_labels += np.sum(np.equal(viterbi_sequence, y_))
        total_labels += sequence_length_
      accuracy = 100.0 * correct_labels / float(total_labels)
      print("Accuracy: %.2f%%" % accuracy)
  
if __name__=="__main__":
  tf.app.run()    
    
    