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
sys.path.append('main1')
sys.path.append('main2')
import numpy as np
import tensorflow as tf
from model import seqLSTM
from TFRecordUtils import ner_read_TFRecord
from utils import nerInputUtils as inputUtils

import pprint
import time
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",25,"Epoch to train[25]")
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
  #target = np.argmax(target, 2)
  #prediction = np.argmax(prediction, 2) #crf prediction is this kind .
  for i in range(len(target)):
    print len(prediction[i]),length[i]
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
  
 
  testaUtils = inputUtils(args.rawword_dim,"testa")
  testa_input = testaUtils.emb;testa_out =  np.argmax(testaUtils.tag,2);testa_sentLent = testaUtils.sentLents
  testa_num_example = np.shape(testa_input)[0]
  print 'testa_out shape:',np.shape(testa_out)
 
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

  if model.load(sess,args.restore,"aida"):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for aida"
    
  for e in range(args.epoch):
    all_pred = []
    all_length=[]
    for ptr in xrange(0,len(testa_input),args.batch_size):
      num_examples = min(ptr+args.batch_size,len(testa_input)) - ptr
      _,loss1,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([train_op,model.loss,model.length,model.output,model.unary_scores,model.transition_params],
                       {model.input_data:testa_input[ptr:min(ptr+args.batch_size,len(testa_input))],
                        model.output_data:testa_out[ptr:min(ptr+args.batch_size,len(testa_input))],
                        model.num_examples:num_examples,
                        model.keep_prob:1})
      print length[1:10],
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testa_out[ptr:min(ptr+args.batch_size,len(testa_input))],length)
      print 'accuracy:',accuracy
      if len(all_pred)==0:
        all_pred = np.array(pred)
        all_length = np.array(length)
      else:
        all_pred = np.concatenate((all_pred,pred))
        all_length = np.concatenate((all_length,length))
    print 'predict:',np.shape(all_pred)
    print 'testa_out shape:',np.shape(testa_out)
    print 'all_length shape:',all_length
    fscore = f1(args, all_pred, testa_out, all_length)
    print "-----------------"
    print("testa: loss:%.4f accuracy:%f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,accuracy,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
    m = fscore[args.class_size]
    print m
if __name__=='__main__':
  tf.app.run()
