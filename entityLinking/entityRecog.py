# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:26:24 2017
@author: wujs
function: utilize the pre-train entity recognition deep model to recognize entity from text
"""
import tensorflow as tf
import numpy as np
from model import seqLSTM
from utils import nerInputUtils as inputUtils
import pprint
import time
import cPickle
import argparse
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")

flags.DEFINE_integer("sentence_length",124,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",310,"hidden dimension of rnn")
flags.DEFINE_integer("ner_word_dim",300,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",90,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.005,"apply dropout during training")


args = flags.FLAGS

def f1(args, prediction, target, length):
  tp = np.array([0] * (args.class_size + 1))
  fp = np.array([0] * (args.class_size + 1))
  fn = np.array([0] * (args.class_size + 1))
  #target = np.argmax(target, 2)
  #prediction = np.argmax(prediction, 2) #crf prediction is this kind .
  for i in range(len(target)):
    for j in range(length[i]):
      if target[i][j] == prediction[i][j]:
        tp[target[i][j]] += 1
      else:
        fp[target[i][j]] += 1
        fn[prediction[i][j]] += 1
  unnamed_entity = args.class_size - 1
  for i in range(args.class_size):
    if i != unnamed_entity:
      tp[args.class_size] += tp[i]
      fp[args.class_size] += fp[i]
      fn[args.class_size] += fn[i]
  precision = []
  recall = []
  fscore = []
  for i in range(args.class_size + 1):
    precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
    recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
    fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
  return fscore

def getCRFRet(tf_unary_scores,tf_transition_params,y,sequence_lengths):
  correct_labels = 0
  total_labels = 0
  predict = []
  for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,sequence_lengths):
    # Remove padding from the scores and tag sequence.
    tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
    y_ = y_[:sequence_length_]

    # Compute the highest scoring sequence.
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
        tf_unary_scores_, tf_transition_params)
    predict.append(viterbi_sequence)
    #print viterbi_sequence
    # Evaluate word-level accuracy.
    correct_labels += np.sum(np.equal(viterbi_sequence, y_))
    total_labels += sequence_length_
  accuracy = 100.0 * correct_labels / float(total_labels)
  
  return np.array(predict),accuracy



def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
class nameEntityRecognition():
  def __init__(self,sess,dir_path,data_tag):
    self.sess = sess
    self.dir_path = dir_path
    self.data_tag = data_tag
    #optimizer = tf.train.AdamOptimizer(0.003)   #when training, we do not need to build those.
    self.model = seqLSTM(args)
    
    
    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.loss, tvars), 10)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
  
    #print 'initiliaze parameters cost time:', time.time()-start_time

    if self.model.load(self.sess,args.restore,"aida"):
      print "[*] seqLSTM is loaded..."
    else:
      print "[*] There is no checkpoint for aida"
  def getEntityRecognition(self,test_input,test_out):
    num_examples = np.shape(test_input)[0]

    loss1,length,lstm_output,tf_unary_scores,tf_transition_params = self.sess.run([self.model.loss,self.model.length,self.model.output,self.model.unary_scores,self.model.transition_params],
                                 {self.model.input_data:test_input,
                                  self.model.output_data:test_out,
                                  self.model.num_examples:num_examples,
                                  self.model.keep_prob:1})
    pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,test_out,length)
    #fscore = f1(args, pred,test_out,length)
    #print("test: loss:%.4f accuracy:%f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,accuracy,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
    
    return pred,lstm_output

if __name__=='__main__':
  '''
  testaUtils = inputUtils(args.rawword_dim,"testa")
  test_input = testaUtils.emb; test_out = testaUtils.tag
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
  parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
    
  data_args = parser.parse_args()
  
  data_tag = data_args.data_tag
  dir_path = data_args.dir_path
  
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=8,inter_op_parallelism_threads=8)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  nerClass = nameEntityRecognition(sess,dir_path,data_tag)
  
  print 'start to load data...'
  start_time = time.time()
  feature_dir_path = dir_path+'features/'
  
  utils = inputUtils()

  test_input = utils.padZeros(cPickle.load(open(feature_dir_path+data_tag+'_embed.p'+args.rawword_dim,'rb')))
  print 'load data cost time:', time.time()-start_time
  
                                         
  testShape = np.shape(test_input)
  print testShape
  if data_tag=='testa' or data_tag=='testb':
    test_out =  utils.padZeros(cPickle.load(open(feature_dir_path+data_tag+'_tag.p'+args.rawword_dim,'rb')),5)
  else:
    test_out = np.zeros([testShape[0],testShape[1],args.class_size],dtype=np.float32)
  test_out = np.argmax(test_out,2)
  
  
  final_pred= None
  batchNo =0
  allSamples = 0
  #final_lstmout =[]
  for input_batch,output_batch in utils.iterate(args.batch_size,test_input,test_out):
    print batchNo,np.shape(input_batch)[0],np.shape(output_batch)
    pred,lstm_output = nerClass.getEntityRecognition(input_batch,output_batch)
    allSamples += len(pred)
    print pred
    if batchNo==0:
      final_pred = pred
    else:
      final_pred = np.concatenate((final_pred,pred))
    batchNo += 1
    #final_lstmout.append(final_lstmout)
  print 'start to save dataset...'
  print 'all pred:',allSamples
  print np.shape(final_pred)
  cPickle.dump(final_pred,open(dir_path+'features/'+data_tag+'_NERresult.p','wb'))