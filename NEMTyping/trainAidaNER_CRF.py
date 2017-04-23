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
from model import seqLSTM_CRF
from utils import nerInputUtils as inputUtils
import re
import collections
import pprint
import time
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
flags.DEFINE_string("datasets","conll2003","dataset name")
flags.DEFINE_integer("sentence_length",124,"max sentence length")
flags.DEFINE_integer("class_size",5,"number of classes")
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

#classType: r'0+',r'1+',r'2+',r'3+',r'4+'
def getChunkNER(strs):
  dicts = collections.defaultdict(set)
  for i in range(args.class_size):
    classType = r''+str(i)+r'+'
    pattern = re.compile(classType)

    match = pattern.search(strs)
    if match:
      dicts[i].add(str(match.start())+'_'+str(match.end()))
  return dicts
  
def f1_chunk(args, prediction, target, length):
  tp = np.array([0] * (args.class_size + 1))
  fp = np.array([0] * (args.class_size + 1))
  fn = np.array([0] * (args.class_size + 1))
  #0-Person,1-Location,2-Organisation,3-Misc,4-None
  #first get the target ner_chunk labels
  for i in range(len(target)):
    lents = length[i]
    reltarget = ''.join(map(str,list(target[i][:lents])))
    relpred = ''.join(map(str,list(prediction[i][:lents])))
    
    dictsTarget = getChunkNER(reltarget)
    dictsPred = getChunkNER(relpred)
    
    for key in dictsPred:
      tp[key] += len(dictsTarget[key] & dictsPred[key])
      fp[key] += len(dictsPred[key] - dictsTarget[key])  
      fn[key] += len(dictsTarget[key]-dictsPred[key])
  
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
  model = seqLSTM_CRF(args)
  print 'start to load data!'
  start_time = time.time()
#  trainUtils = inputUtils(args.rawword_dim,"train")
#  train_TFfileName = trainUtils.TFfileName; train_nerShapeFile = trainUtils.nerShapeFile;
#  train_batch_size = args.batch_size;
  
  dir_path ='data/conll2003/features/'
  testaUtils = inputUtils(args.rawword_dim,dir_path,"testa")
  testa_input = testaUtils.emb;testa_out =  np.argmax(testaUtils.tag,2)
  testa_num_example = np.shape(testa_input)[0]
  
  testbUtils = inputUtils(args.rawword_dim,dir_path,"testb")
  testb_input = testbUtils.emb;
  testb_out =  np.argmax(testbUtils.tag,2)
  testb_num_example = np.shape(testb_input)[0]
  
  
  trainUtils = inputUtils(args.rawword_dim,dir_path,"train")
  train_input = trainUtils.emb;train_out = np.argmax(trainUtils.tag,2)
 
  
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

  if model.load(sess,args.restore,"conll2003"):
    print "[*] seqLSTM is loaded..."
  else:
    print "[*] There is no checkpoint for conll2003"

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
    
      loss1,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.output,model.unary_scores,model.transition_params],
                       {model.input_data:testa_input,
                        model.output_data:testa_out,
                        model.num_examples:testa_num_example,
                        model.keep_prob:1})
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testa_out,length)
        
      fscore = f1_chunk(args, pred, testa_out, length)
      
      m = fscore[args.class_size]
    
      if m > maximum:
        maximum = m
        if maximum > 0.80:
          model.save(sess,args.restore,"conll2003") #optimize in the dev file!
        print "------------------"
        print("testa: loss:%.4f accuracy:%f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,accuracy,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        
        
        loss1,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.output,model.unary_scores,model.transition_params],
                       {model.input_data:testb_input,
                        model.output_data:testb_out,
                        model.num_examples:testb_num_example,
                        model.keep_prob:1})
        pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,testb_out,length)
        fscore = f1_chunk(args, pred, testb_out, length)
        print("testb: loss:%.4f accuracy:%f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,accuracy,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        
        print "------------------"
      k += 1
      num_example = min(ptr+args.batch_size,len(train_input)) - ptr  
      _,lstm_output = sess.run([train_op,model.output],
                        {model.input_data:train_input[ptr:min(ptr+args.batch_size,len(train_input))],
                         model.output_data:train_out[ptr:min(ptr+args.batch_size,len(train_input))],
                         model.num_examples: num_example,
                         model.keep_prob:0.5})
  
      
      loss1,length,lstm_output,tf_unary_scores,tf_transition_params = sess.run([model.loss,model.length,model.output,model.unary_scores,model.transition_params],
                              {model.input_data:train_input[ptr:min(ptr+args.batch_size,len(train_input))],
                               model.output_data:train_out[ptr:min(ptr+args.batch_size,len(train_input))],
                               model.num_examples: num_example,
                               model.keep_prob:1})
      id_epoch += 1
      pred,accuracy = getCRFRet(tf_unary_scores,tf_transition_params,train_out[ptr:min(ptr+args.batch_size,len(train_input))],length)
      fscore = f1(args, pred,train_out[ptr:min(ptr+args.batch_size,len(train_input))],length)
      if id_epoch %10==0:
        print("train: loss:%.4f accuracy:%f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,accuracy,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
#  except:
#    print 'finished train'
if __name__=='__main__':
  tf.app.run()
