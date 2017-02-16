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
from sklearn.metrics import f1_score
import time
import cPickle
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",50,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","aida","dataset name")
flags.DEFINE_integer("sentence_length",124,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",114,"hidden dimension of rnn")
flags.DEFINE_integer("candidate_ent_num",30,"hidden dimension of rnn")
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
  target = np.argmax(target, 2)
  prediction = np.argmax(prediction, 2)
  for i in range(len(target)):
    for j in range(length[i]):
      if target[i, j] == prediction[i, j]:
        tp[target[i, j]] += 1
      else:
        fp[target[i, j]] += 1
        fn[prediction[i, j]] += 1
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

class nameEntityRecognition():
  def __init__(self,sess):
    self.sess = sess
    #optimizer = tf.train.AdamOptimizer(0.003)   #when training, we do not need to build those.
    self.model = seqLSTM(args)
    start_time = time.time()
    #print 'initiliaze parameters cost time:', time.time()-start_time

    if self.model.load(self.sess,args.restore,"aida"):
      print "[*] seqLSTM is loaded..."
    else:
      print "[*] There is no checkpoint for aida"
  def getEntityRecognition(self,test_input,test_out):
    loss1,pred,length,lstm_output = self.sess.run([self.model.loss,self.model.prediction,self.model.length,self.model.output],
                                 {self.model.input_data:test_input,
                                  self.model.output_data:test_out,
                                  self.model.keep_prob:1})
    fscore = f1(args, pred, test_out, length)
    #cPickle.dump(pred,open('data/ace/features/ace_NERresult.p','wb'))
    cPickle.dump(pred,open('data/msnbc/features/msnbc_NERresult.p','wb'))
    print "-----------------"
    print("test: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
    return lstm_output

if __name__=='__main__':
  '''
  testaUtils = inputUtils(args.rawword_dim,"testa")
  test_input = testaUtils.emb; test_out = testaUtils.tag
  '''
  print 'start to load data...'
  start_time = time.time()
  #dir_path = '/home/wjs/demo/entityType/informationExtract/data/ace/features'
  dir_path = '/home/wjs/demo/entityType/informationExtract/data/msnbc/features'
  test_input = cPickle.load(open(dir_path+'/msnbc_embed.p100','rb'))
  print 'load data cost time:', time.time()-start_time
  
  testShape = np.shape(test_input)
  
  print testShape
  
  test_input  = np.concatenate((test_input,np.zeros([testShape[0],max(0,124-testShape[1]),testShape[2]])),axis=1)
  
  print np.shape(test_input)
  testShape = np.shape(test_input)
  print testShape
  assert testShape[1]==124
  
  test_out = np.zeros([testShape[0],testShape[1],args.class_size],dtype=np.float32)
  
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config);
  nerClass = nameEntityRecognition(sess);
  
  '''
  testbUtils = inputUtils(args.rawword_dim,"testb")
  test_input = testbUtils.emb; test_out = testbUtils.tag;
  '''
  lstm_output = nerClass.getEntityRecognition(test_input,test_out)