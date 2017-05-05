# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:26:24 2017
@author: wujs
function: utilize the pre-train entity recognition deep model to recognize entity from text
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
import random_vec
import tensorflow as tf
import numpy as np
from model import seqLSTM_CRF
from utils import nerInputUtils as inputUtils
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec,get_input_figer_chunk
import pprint
import time
import cPickle
import argparse
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","conll2003","dataset name")
flags.DEFINE_integer("sentence_length",80,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",111,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",90,"hidden dimension of rnn")
flags.DEFINE_integer("figer_type_num",113,"figer type total numbers")
flags.DEFINE_string("rawword_dim","100","hidden dimension of rnn")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_float("learning_rate",0.005,"apply dropout during training")
args = flags.FLAGS

def getfiger(dir_path,data_tag):
  dims = 100
  print 'load figer train..'
  stime = time.time()
  model = cPickle.load(open("/home/wjs/demo/entityType/informationExtract/data/wordvec_model_100.p", 'rb'))
  #model = cPickle.load(open("data/wordvec_model_100.p", 'rb'))
  etime = time.time()
  print 'load times:',etime-stime
  input_file_obj = open(dir_path+data_tag+"Data.txt")
  sentence_length=80
  #run one by one ....
  data_batch = 0
  batch_input=[];batch_output=[]
  for test_input,test_output in get_input_figer_chunk(dir_path,model,dims,input_file_obj,sentence_length):
    batch_input.append(test_input)
    batch_output.append(test_output)
    data_batch+=1
    if data_batch %5000==0:   #total line is 2000000
      yield batch_input,np.argmax(batch_output,2)
      batch_input=[];batch_output=[]
    
    
    
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
#    correct_labels += np.sum(np.equal(viterbi_sequence, y_))
#    total_labels += sequence_length_
#  accuracy = 100.0 * correct_labels / float(total_labels)
  
  return np.array(predict)#,accuracy



def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
class nameEntityRecognition():
  def __init__(self,sess,dir_path,data_tag):
    self.sess = sess
    self.dir_path = dir_path
    self.data_tag = data_tag
    #optimizer = tf.train.AdamOptimizer(0.003)   #when training, we do not need to build those.
    self.model = seqLSTM_CRF(args)
    #print 'initiliaze parameters cost time:', time.time()-start_time

    if self.model.load(self.sess,args.restore,"conll2003"):
      print "[*] seqLSTM is loaded..."
    else:
      print "[*] There is no checkpoint for conll2003"
      
  def getEntityRecognition(self,test_input,test_out):
    num_examples = np.shape(test_input)[0]
    loss1,length,lstm_output,tf_unary_scores,tf_transition_params = self.sess.run([self.model.loss,self.model.length,self.model.output,self.model.unary_scores,self.model.transition_params],
                                 {self.model.input_data:test_input,
                                  self.model.output_data:test_out,
                                  self.model.num_examples:num_examples,
                                  self.model.keep_prob:1})
    pred = getCRFRet(tf_unary_scores,tf_transition_params,test_out,length)
    if self.data_tag == 'ace' or self.data_tag == 'msnbc':
      cPickle.dump(pred,open(self.dir_path+'features/'+self.data_tag+'_NERresult.p','wb'))

    return pred

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
  
  fout_chunk = open(dir_path+data_tag+"_ner_conll2003.txt_new",'w')
  
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config);
  nerClass = nameEntityRecognition(sess,dir_path,data_tag);
#  testi =0
#  for test_input,test_out in getfiger(dir_path,data_tag):
#    print 'batch no:',testi
#    testi += 1
#    preds = nerClass.getEntityRecognition(test_input,test_out)
#    predT = ""
#    for i in range(5000):
#      pred = preds[i]
#      predT += "\n".join(map(str,pred))
#      predT += "\n\n"
#    fout_chunk.write(predT)
#    fout_chunk.flush()
#  fout_chunk.close()
  testUtils = inputUtils(100,'data/figer_test/',flag='figer')
  test_input =  testUtils.emb;test_out = testUtils.tag
  preds = nerClass.getEntityRecognition(test_input,test_out)
  predT=""
  for i in range(len(preds)):
    pred = preds[i]
    predT += "\n".join(map(str,pred))
    predT += "\n\n"
    fout_chunk.write(predT)
    fout_chunk.flush()
    predT=""
  fout_chunk.close()
  
  
      