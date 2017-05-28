# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:26:24 2017
@author: wujs
function: utilize the pre-train entity recognition deep model to recognize entity from text
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
from embedding import WordVec,MyCorpus,RandomVec,get_NER_embedding
import tensorflow as tf
import numpy as np
from model import seqLSTM_CRF
from trainAidaNER_CRF import args
import gensim
import cPickle



def getCRFRet(tf_unary_scores,tf_transition_params,y,sequence_lengths):
  predict = []
  for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,sequence_lengths):
    tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
    y_ = y_[:sequence_length_]

    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
        tf_unary_scores_, tf_transition_params)
    predict.append(viterbi_sequence)
  
  return np.array(predict)


class namedEntityRecognition():
  def __init__(self):
    self.randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
    self.word2vecModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
    config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth=True
    self.sess = tf.InteractiveSession(config=config);
    self.model = seqLSTM_CRF(args)
    if self.model.load(self.sess,args.restore,"conll2003"):
      print "[*] seqLSTM is loaded..."
    else:
      print "[*] There is no checkpoint for conll2003"

  def __call__(self,rawData):
    test_input,test_out = get_NER_embedding(self.word2vecModel,self.randomVector,rawData)
    num_examples = np.shape(test_input)[0]
    length,tf_unary_scores,tf_transition_params = self.sess.run([self.model.length,self.model.unary_scores,self.model.transition_params],
                                                                       {self.model.input_data:test_input,
                                                                        self.model.output_data:test_out,
                                                                        self.model.num_examples:num_examples,
                                                                        self.model.keep_prob:1})
    pred = getCRFRet(tf_unary_scores,tf_transition_params,test_out,length)

    return pred