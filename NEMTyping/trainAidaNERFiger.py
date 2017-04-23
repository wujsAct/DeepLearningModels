# -*- coding: utf-8 -*-
'''
@time: 2016/12/20
@editor: wujs
@function: we add entity linking module
first version: we sum description sentence for candidates entities!
@revise: 2017/1/4
'''
import sys
sys.path.append('utils')
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
import random_vec
import numpy as np
import tensorflow as tf
from model import seqLSTM
from embedding import WordVec,MyCorpus,get_input_figer,RandomVec,get_input_figer_chunk,get_input_figer_chunk_train
import cPickle
from utils import nerInputUtils as inputUtils
import pprint
import time
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","figer","dataset name")
flags.DEFINE_integer("sentence_length",80,"max sentence length")
flags.DEFINE_integer("class_size",114,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",111,"hidden dimension of rnn")
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
  prediction = np.argmax(prediction, 2) #crf prediction is this kind .
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


def main(_):
  pp.pprint(flags.FLAGS.__flags)

  '''
  @function: load the train and test datasets
  @entlinking context: 'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag
  '''
  model = seqLSTM(args)
  print 'start to load data!'
  start_time = time.time()
  word2vecModel = cPickle.load(open('data/wordvec_model_100.p'))
  
  testa_input,testa_out = get_input_figer_chunk('data/figer/',"testa",model=word2vecModel,word_dim=100,sentence_length=80) 
  testb_input,testb_out = get_input_figer_chunk('data/figer/',"testb",model=word2vecModel,word_dim=100,sentence_length=80)
  
  print 'start to build seqLSTM'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  print 'initiliaze parameters cost time:', time.time()-start_time

  optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='seqLSTM_variables')
  print 'tvars:',tvars

  start_time = time.time()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), 10)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  sess.run(tf.global_variables_initializer())
  print 'build optimizer for seqLSTM cost time:', time.time()-start_time

  if model.load(sess,args.restore,"figer"):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for aida"

  id_epoch = 0
  '''
  @train named entity recognition models
  '''
  maximum=0
  k = 0
  
  for train_input,train_out in get_input_figer_chunk_train('data/figer/',"train",model=word2vecModel,word_dim=100,sentence_length=80):
    
    id_epoch = id_epoch + 1

    loss1,length,pred,accuracy = sess.run([model.loss,model.length,model.prediction,model.accuracy],
                     {model.input_data:testa_input,
                      model.output_data:tf.SparseTensorValue(testa_out[0],testa_out[1],testa_out[2]),
                      model.num_examples:args.batch_size,
                      model.keep_prob:1})
   
      
    if accuracy > maximum:
      #model.save(sess,args.restore,"figer") #optimize in the dev file!
      maximum = accuracy
      print "------------------"
      print("test: loss:%.4f accuracy:%.2f" %(loss1,accuracy))
  
     
      loss1,length,pred,accuracy = sess.run([model.loss,model.length,model.prediction,model.accuracy],
                   {model.input_data:testb_input,
                    model.output_data:tf.SparseTensorValue(testb_out[0],testb_out[1],testb_out[2]),
                    model.num_examples:args.batch_size,
                    model.keep_prob:1})
     
      print("test: loss:%.4f accuracy :%.2f" %(loss1,accuracy))
      print "------------------"
    
    k += 1
    _,lstm_output = sess.run([train_op,model.output],
                      {model.input_data:train_input,
                       model.output_data:tf.SparseTensorValue(train_out[0],train_out[1],train_out[2]),
                       model.num_examples: args.batch_size,
                       model.keep_prob:0.5})

    loss1,length,pred,accuracy = sess.run([model.loss,model.length,model.prediction,model.accuracy],
                            {model.input_data:train_input,
                             model.output_data:tf.SparseTensorValue(train_out[0],train_out[1],train_out[2]),
                             model.num_examples: args.batch_size,
                             model.keep_prob:1})
    id_epoch += 1
    if id_epoch %10==0:
      print("train: loss:%.4f accuracy:%.2f" %(loss1,accuracy))
if __name__=='__main__':
  tf.app.run()
