# -*- coding: utf-8 -*-
'''
@time: 2016/12/20
@editor: wujs
@function: we add entity linking module
first version: we sum description sentence for candidates entities!
second version: we may utilize cnn!
'''


import os
import numpy as np
import tensorflow as tf
from model import ctxCNN,seqLSTM,ctxSum
from utils import inputUtils
import pprint
import random
import time
from sklearn.metrics import f1_score
from tqdm import tqdm
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",256,"batch size of training")
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


def getLinkingFeature(lstm_output,ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ent_linking_candprob,ptr,flag='train'):
#def getLinkingFeature(ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ptr):
  ent_mention_linking_tag_list = []
  candidate_ent_linking_feature=[]
  candidate_ent_type_feature=[]
  candidate_ent_prob_feature=[]
  candidate_ent_relcoherent_feature=[]
  ent_mention_lstm_feature = []
  allLenght = len(ent_mention_index)
  
  ent_index=[] #记录ent mention的位置
  lstm_index=[] #记录index的位置
  if flag == 'train':
    last_range = min(ptr+args.batch_size,allLenght)
  else:
    last_range = allLenght
  for ids in xrange(ptr,last_range):
    tagid = 0
    for ent_item in ent_mention_index[ids]:
      if np.sum(ent_mention_tag[ids][tagid]) != 0:
        ent_index.append([ent_item[0],ent_item[1]]) 
        lstm_index.append(ids-ptr)
        
        ent_mention_linking_tag = np.asarray(ent_mention_tag[ids][tagid],dtype=np.float32)
        ent_mention_linking_tag_list.append(ent_mention_linking_tag)
        candidate_ent_relcoherent_item = np.asarray(ent_relcoherent[ids][tagid],dtype=np.float32)
        candidate_ent_relcoherent_feature.append(candidate_ent_relcoherent_item)
        
        candidate_num = len(ent_mention_link_feature[ids][tagid])/100 #这种求解的方法貌似有问题呢！
        ent_linking_candidates =  np.concatenate((np.asarray(np.reshape(ent_mention_link_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), 
                                                          np.zeros((max(0,args.candidate_ent_num-candidate_num),100),dtype=np.float32)))
        ent_type_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_type[ids][tagid],(candidate_num,args.figer_type_num)),dtype=np.float32),
                                  np.zeros((max(0,args.candidate_ent_num-candidate_num),args.figer_type_num),dtype=np.float32)))
        ent_prob_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_candprob[ids][tagid],(candidate_num,3)),dtype=np.float32),
                                  np.zeros((max(0,args.candidate_ent_num-candidate_num),3),dtype=np.float32)))
                                        
        candidate_ent_linking_feature.append(ent_linking_candidates)
        candidate_ent_type_feature.append(ent_type_candidates)
        candidate_ent_prob_feature.append(ent_prob_candidates)
        ent_mention_lstm_feature.append(np.sum(lstm_output[ids-ptr][ent_item[0]:ent_item[1]],axis=0))
      tagid += 1
  ent_mention_linking_tag_list = np.asarray(ent_mention_linking_tag_list)
  candidate_ent_relcoherent_feature = np.asarray(candidate_ent_relcoherent_feature)
  candidate_ent_linking_feature = np.asarray(candidate_ent_linking_feature)
  candidate_ent_type_feature = np.asarray(candidate_ent_type_feature)
  candidate_ent_prob_feature = np.asarray(candidate_ent_prob_feature)
  ent_mention_lstm_feature = np.expand_dims(np.asarray(ent_mention_lstm_feature),2)
  
  return ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature
  
def shuffle_in_unison(a, b):
  assert len(a) == len(b)
  shuffled_a = np.empty(a.shape, dtype=a.dtype)
  shuffled_b = np.empty(b.shape, dtype=b.dtype)
  permutation = np.random.permutation(len(a))
  for old_index, new_index in enumerate(permutation):
    shuffled_a[new_index] = a[old_index]
    shuffled_b[new_index] = b[old_index]
  return shuffled_a, shuffled_b
    
def main(_):
  pp.pprint(flags.FLAGS.__flags)
  if not os.path.exists(args.restore):
    print("[*] creating checkpoint directory...")
    os.makedirs(args.restore)
  '''
  @function: initialize the parameters and start the interactiveSession.
  '''
  model = seqLSTM(args)
  loss_ent = model.loss
  
  modelLinking = ctxSum(args)
  loss_linking = modelLinking.linking_loss #+ 0.01*tf.nn.l2_loss(modelLinking.bilinear_w_descrip)+0.01*tf.nn.l2_loss(modelLinking.bilinear_w_type)
  
  '''
  @function: load the train and test datasets
  @entlinking context: 'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag
  '''
  print 'start to load data...'
  
  start = time.time()
  testaUtils = inputUtils(args.rawword_dim,"testa")
  testa_input = testaUtils.emb; testa_out = testaUtils.tag; testa_entliking= testaUtils.ent_linking; 
  testa_ent_mention_index = testa_entliking['ent_mention_index']; testa_ent_mention_link_feature=testa_entliking['ent_mention_link_feature'];
  testa_ent_mention_tag = testa_entliking['ent_mention_tag']; testa_ent_relcoherent = testaUtils.ent_relcoherent
  testa_ent_linking_type = testaUtils.ent_linking_type; testa_ent_linking_candprob = testaUtils.ent_linking_candprob
  
  trainUtils = inputUtils(args.rawword_dim,"train")
  train_input = trainUtils.emb; train_out = trainUtils.tag; train_entliking= trainUtils.ent_linking; 
  train_ent_mention_index = train_entliking['ent_mention_index']; train_ent_mention_link_feature=train_entliking['ent_mention_link_feature'];
  train_ent_mention_tag = train_entliking['ent_mention_tag']; train_ent_relcoherent = trainUtils.ent_relcoherent
  train_ent_linking_type = trainUtils.ent_linking_type; train_ent_linking_candprob = trainUtils.ent_linking_candprob
  
  testbUtils = inputUtils(args.rawword_dim,"testb")
  testb_input = testbUtils.emb; testb_out = testbUtils.tag; testb_entliking= testbUtils.ent_linking
  testb_ent_mention_index = testb_entliking['ent_mention_index']; testb_ent_mention_link_feature=testb_entliking['ent_mention_link_feature'];
  testb_ent_mention_tag = testb_entliking['ent_mention_tag']; testb_ent_relcoherent = testbUtils.ent_relcoherent
  testb_ent_linking_type = testbUtils.ent_linking_type; testb_ent_linking_candprob = testbUtils.ent_linking_candprob
  print 'cost:', time.time()-start,' to load data'
  
  start_time = time.time()
  print 'gradient computing...'
  optimizer1 = tf.train.RMSPropOptimizer(args.learning_rate)

  tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='seqLSTM')
  print 'tvars:',tvars
  
  grads,_ = tf.clip_by_global_norm(tf.gradients(loss_ent,tvars),10)
  train_op = optimizer1.apply_gradients(zip(grads,tvars))
  print 'gradient computing cost time:', time.time()-start_time
  
  tvars_linking = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='ctxSum')
  print 'tvars_linking:',tvars_linking
  
  #optimizer2 = tf.train.GradientDescentOptimizer(0.2)
  optimizer2 = tf.train.AdamOptimizer(0.05)
  gradsLinking,_ = tf.clip_by_global_norm(tf.gradients(loss_linking,tvars_linking),5)
  train_op_linking = optimizer2.apply_gradients(zip(gradsLinking,tvars_linking))
  
  print 'start to initialize parameters'
  start_time = time.time()
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config)
  print 'initiliaze parameters cost time:', time.time()-start
  init = tf.global_variables_initializer()
  sess.run(init)
  
  maximum = 0
  maximum_linking=0
  saver = tf.train.Saver()  #prepare to save the trained models
  start_tr = time.time()
  
  id_epoch = 0
  for e in range(args.epoch):
    id_epoch = 0
    print 'Epoch: %d------------' %(e)
    average_linking_accuracy_train = 0;average_loss_train = 0
    for ptr in xrange(0,len(train_input),args.batch_size):
      id_epoch = id_epoch + 1
      _,lstm_output = sess.run([train_op,model.output], 
                        {model.input_data:train_input[ptr:min(ptr+args.batch_size,len(train_input))],
                          model.output_data:train_out[ptr:min(ptr+args.batch_size,len(train_input))],
                          model.keep_prob:0.5})
       
      ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                  getLinkingFeature(lstm_output,train_ent_mention_index,train_ent_mention_tag,\
                                                  train_ent_relcoherent,train_ent_mention_link_feature,train_ent_linking_type,train_ent_linking_candprob,ptr,flag='train')
       
      _,loss2,accuracy,pred = sess.run([train_op_linking,loss_linking,modelLinking.accuracy,modelLinking.prediction],
                                 {modelLinking.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelLinking.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelLinking.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelLinking.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelLinking.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelLinking.ent_mention_lstm_feature:ent_mention_lstm_feature
                                 })
      f1_micro,f1_macro =f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
      #print 'total loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
      average_linking_accuracy_train += accuracy 
      average_loss_train += loss2
      if id_epoch%20==0:
        loss1,pred,length = sess.run([loss_ent,model.prediction,model.length],
                          {model.input_data:train_input[ptr:ptr+args.batch_size],
                          model.output_data:train_out[ptr:ptr+args.batch_size],
                          model.keep_prob:1})
        fscore = f1(args, pred, train_out[ptr:ptr+args.batch_size], length)
        print("train: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        
        '''
        @function: testa(dev) 
        '''
        loss1,pred,length,lstm_output = sess.run([loss_ent,model.prediction,model.length,model.output],
                             {model.input_data:testa_input,
                              model.output_data:testa_out,
                              model.keep_prob:1})
        fscore = f1(args, pred, testa_out, length)
        print "-----------------"
        print("testa: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                  getLinkingFeature(lstm_output,testa_ent_mention_index,testa_ent_mention_tag,\
                                                  testa_ent_relcoherent,testa_ent_mention_link_feature,testa_ent_linking_type,testa_ent_linking_candprob,0,flag='testa')
        loss2,accuracy,pred = sess.run([loss_linking,modelLinking.accuracy,modelLinking.prediction],
                                 {modelLinking.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelLinking.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelLinking.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelLinking.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelLinking.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelLinking.ent_mention_lstm_feature:ent_mention_lstm_feature
                                 })
        f1_micro_testa,f1_macro_testb =f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
        print 'testa total loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
        
        m = fscore[args.class_size]
        if m > maximum:
          maximum = m
          loss1,pred,length,lstm_output = sess.run([loss_ent,model.prediction,model.length,model.output],
                             {model.input_data:testb_input,
                              model.output_data:testb_out,
                              model.keep_prob:1})
          fscore = f1(args, pred, testb_out, length)
          print("testb: loss:%.4f NER:%.2f LOC:%.2f MISC:%.2f ORG:%.2f PER:%.2f" %(loss1,100*fscore[5],100*fscore[1],100*fscore[3],100*fscore[2],100*fscore[0]))
        if accuracy > maximum_linking:
          maximum_linking = accuracy
          loss1,pred,length,lstm_output = sess.run([loss_ent,model.prediction,model.length,model.output],
                             {model.input_data:testb_input,
                              model.output_data:testb_out,
                              model.keep_prob:1})

          ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                  getLinkingFeature(lstm_output,testb_ent_mention_index,testb_ent_mention_tag,\
                                                  testb_ent_relcoherent,testb_ent_mention_link_feature,testb_ent_linking_type,testb_ent_linking_candprob,0,flag='testb')
          loss2,accuracy,pred = sess.run([loss_linking,modelLinking.accuracy,modelLinking.prediction],
                                 {modelLinking.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                  modelLinking.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                  modelLinking.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                  modelLinking.candidate_ent_type_feature:candidate_ent_type_feature,
                                  modelLinking.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                  modelLinking.ent_mention_lstm_feature:ent_mention_lstm_feature
                                 })
          f1_micro_testa,f1_macro_testb =f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
          print 'testb total loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
        print "-----------------" 
    print 'average linking accuracy:',average_linking_accuracy_train/id_epoch, average_loss_train/id_epoch
if __name__ == '__main__':
  tf.app.run()      
