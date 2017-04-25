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
from scipy.sparse import coo_matrix
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",10,"batch size of training")
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
flags.DEFINE_float("learning_rate",0.0001,"apply dropout during training")
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
'''
@sentence_final: shape:(batch_size,sequence_length,dims)
'''
def padZeros(sentence_final,max_sentence_length=80,dims=111):
  for i in range(len(sentence_final)):
    offset = max_sentence_length-len(sentence_final[i])
    sentence_final[i] += [[0]*dims]*offset
    
  return np.asarray(sentence_final)


def genEntMentMask(entment_mask_final):
  entNums = len(entment_mask_final)
  entment_masks = np.zeros((entNums,args.batch_size,args.sentence_length,args.rnn_size*2))
  for i in range(entNums):
    items = entment_mask_final[i]
    
    ids = items[0];start=items[1];end=items[2]
    for ient in range(start,end):
        entment_masks[i,ids,ient] = np.asarray([1]*args.rnn_size*2)
  return entment_masks
    
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
  print 'load word2vec model cost time:',time.time()-start_time

  start_time = time.time()                                           
  testa_entment_mask,testa_sentence_final,testa_tag_final = get_input_figer_chunk(args.batch_size,'data/figer/',"testa",model=word2vecModel,word_dim=100,sentence_length=80)
  print 'load testa data cost time:',time.time()-start_time
  
                                              
  start_time = time.time()           
  testb_entment_mask,testb_sentence_final,testb_tag_final = get_input_figer_chunk(args.batch_size,'data/figer/',"testb",model=word2vecModel,word_dim=100,sentence_length=80)
  print 'load testb data cost time:',time.time()-start_time                                  

                                              
  print 'start to build seqLSTM'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  
  print 'initiliaze parameters cost time:', time.time()-start_time

  optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
  
  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='seqLSTM_variables')
  print 'tvars:',tvars
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.01
  totalLoss = model.loss  + lossL2                
  start_time = time.time()
  grads, _ = tf.clip_by_global_norm(tf.gradients(totalLoss, tvars), 10)
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
  for epoch in range(50):
    print 'epoch:',epoch
    print '---------------------------------'
    for train_entment_mask,train_sentence_final,train_tag_final in get_input_figer_chunk_train(args.batch_size,'data/figer/',"train",model=word2vecModel,word_dim=100,sentence_length=80):
      if id_epoch % 60000 ==0 and id_epoch!=0:
        accuracy_list=[]
        for i in range(len(testa_sentence_final)):
          testa_input = padZeros(testa_sentence_final[i])
          testa_out = testa_tag_final[i]
          testa_entMentIndex = genEntMentMask(testa_entment_mask[i])
          num_examples = len(testa_entMentIndex)
          type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
          loss1,tloss,length,pred,accuracy = sess.run([model.loss,totalLoss,model.length,model.prediction,model.accuracy],
                         {model.input_data:testa_input,
                          model.output_data:tf.SparseTensorValue(testa_out[0],testa_out[1],type_shape),
                          model.num_examples:num_examples,
                          model.entMentIndex:testa_entMentIndex,
                          model.keep_prob:1})
          accuracy_list.append(accuracy)
        
        if np.average(accuracy_list) > maximum:
          maximum = np.average(accuracy_list)
          if maximum > 0.5:
            model.save(sess,args.restore,"figer") #optimize in the dev file!
          print "------------------"
          print("testa: loss:%.4f total loss:%.4f average accuracy:%.6f" %(loss1,tloss,maximum))
          accuracy_list=[]
          for i in range(len(testb_sentence_final)):
            testb_input = padZeros(testb_sentence_final[i])
            testb_entMentIndex = genEntMentMask(testb_entment_mask[i])
            testb_out = testb_tag_final[i]
            num_examples = len(testb_entMentIndex)
            type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
            loss1,tloss,length,pred,accuracy = sess.run([model.loss,totalLoss,model.length,model.prediction,model.accuracy],
                       {model.input_data:testb_input,
                        model.output_data:tf.SparseTensorValue(testb_out[0],testb_out[1],type_shape),
                        model.num_examples:args.batch_size,
                        model.entMentIndex:testb_entMentIndex,
                        model.keep_prob:1})
            accuracy_list.append(accuracy)
          print("testb: loss:%.4f total loss:%.4f average accuracy:%.6f" %(loss1,tloss,np.average(accuracy_list)))
          print "------------------"
          
      train_input = padZeros(train_sentence_final)
      train_entMentIndex = genEntMentMask(train_entment_mask)
      train_out = train_tag_final #we need to generate entity mention masks!
      num_examples = len(train_entMentIndex)
      type_shape=  np.array([num_examples,args.class_size], dtype=np.int64)
      _,loss1,tloss,length,pred,accuracy = sess.run([train_op,model.loss,totalLoss,model.length,model.prediction,model.accuracy],
                        {model.input_data:train_input,
                         model.output_data:tf.SparseTensorValue(train_out[0],train_out[1],type_shape),
                         model.num_examples: args.batch_size,
                         model.entMentIndex:train_entMentIndex,
                         model.keep_prob:0.5})
      id_epoch += 1
      if id_epoch % 100==0:
        print("ids: %d,train: loss:%.4f total loss:%.4f accuracy:%.6f" %(id_epoch,loss1,tloss,accuracy))
if __name__=='__main__':
  tf.app.run()
