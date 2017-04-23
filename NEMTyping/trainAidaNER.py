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
import numpy as np
import tensorflow as tf
from model import seqLSTM
from utils import nerInputUtils as inputUtils

import pprint
import time
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","conll2000","dataset name")
flags.DEFINE_integer("sentence_length",80,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
flags.DEFINE_integer("rnn_size",128,"hidden dimension of rnn")
flags.DEFINE_integer("word_dim",106,"hidden dimension of rnn")
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
#  trainUtils = inputUtils(args.rawword_dim,"train")
#  train_TFfileName = trainUtils.TFfileName; train_nerShapeFile = trainUtils.nerShapeFile;
#  train_batch_size = args.batch_size;
  

  testaUtils = inputUtils(args.rawword_dim,"test")
  testa_input = testaUtils.emb;testa_out =  testaUtils.tag
  testa_num_example = np.shape(testa_input)[0]
  
  
  trainUtils = inputUtils(args.rawword_dim,"train")
  train_input = trainUtils.emb;train_out = trainUtils.tag
 
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

  if model.load(sess,args.restore,"conll2000"):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for aida"

  id_epoch = 0
  '''
  @train named entity recognition models
  '''
  maximum=0
  k = 0
  
#  for train_input,train_out in ner_read_TFRecord(sess,train_TFfileName,
#                                                 train_nerShapeFile,train_batch_size,args.epoch):
  for e in range(args.epoch):
    id_epoch = 0
    print 'Epoch: %d------------' %(e)
    for ptr in xrange(0,len(train_input),args.batch_size):
      id_epoch = id_epoch + 1
    
      loss1,length,pred = sess.run([model.loss,model.length,model.prediction],
                       {model.input_data:testa_input,
                        model.output_data:testa_out,
                        model.num_examples:testa_num_example,
                        model.keep_prob:1})
     
      fscore = f1(args, pred, testa_out, length)
      
      m = fscore[args.class_size]
    
      if m > maximum:
        model.save(sess,args.restore,"conll2000") #optimize in the dev file!
        maximum = m
        print "------------------"
        print("test: loss:%.4f named :%.2f,B-NP:%.2f I-NP:%.2f B-VP:%.2f I-VP:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        print "------------------"
      
      k += 1
      num_example = min(ptr+args.batch_size,len(train_input)) - ptr  
      _,lstm_output = sess.run([train_op,model.output],
                        {model.input_data:train_input[ptr:min(ptr+args.batch_size,len(train_input))],
                         model.output_data:train_out[ptr:min(ptr+args.batch_size,len(train_input))],
                         model.num_examples: num_example,
                         model.keep_prob:0.5})
  
      
      loss1,length,pred = sess.run([model.loss,model.length,model.prediction],
                              {model.input_data:train_input[ptr:min(ptr+args.batch_size,len(train_input))],
                               model.output_data:train_out[ptr:min(ptr+args.batch_size,len(train_input))],
                               model.num_examples: num_example,
                               model.keep_prob:1})
      id_epoch += 1

      fscore = f1(args, pred,train_out[ptr:min(ptr+args.batch_size,len(train_input))],length)
      if id_epoch %10==0:
        print("train: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
#  except:
#    print 'finished train'
if __name__=='__main__':
  tf.app.run()
