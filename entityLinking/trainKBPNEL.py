# -*- coding: utf-8 -*-
'''
@editor: wujs
function: we add entity linking module
revise: 2017/1/8
'''

import tensorflow as tf
import time
from model import ctxSum
from entityRecog import nameEntityRecognition,pp,flags,args #get seqLSTM features
from sklearn.metrics import f1_score
from utils import nelInputUtils as inputUtils
from utils import getLinkingFeature
import numpy  as np
import cPickle
import argparse

def getData(dataTag):
  dataUtils = inputUtils(args.rawword_dim,dataTag)
  data_input = dataUtils.emb; 
  data_entliking= dataUtils.ent_linking;data_ent_mention_index = data_entliking['ent_mention_index'];
  data_ent_mention_link_feature=data_entliking['ent_mention_link_feature'];data_ent_mention_tag = data_entliking['ent_mention_tag']; 
  data_ent_relcoherent = [dataUtils.ent_relcoherent_ngd,dataUtils.ent_relcoherent_fb]
  data_ent_linking_type = dataUtils.ent_linking_type; 
  data_ent_linking_candprob = dataUtils.ent_linking_candprob
  data_ent_surfacewordv_feature = dataUtils.ent_surfacewordv_feature
  
  data_shape = np.shape(data_input)
  if dataTag in ['train','testa','testb']:
    data_out = np.argmax(dataUtils.tag,2) 
  else:
    data_out = np.zeros([data_shape[0],data_shape[1],args.class_size],dtype=np.float32)  
    data_out = np.argmax(data_out,2)
  
  return data_input, data_out, data_ent_mention_index, data_ent_mention_link_feature, data_ent_mention_tag, data_ent_relcoherent,data_ent_linking_type,data_ent_linking_candprob,data_ent_surfacewordv_feature


def getDataTestSets():
  kbpTestList = getData("kbp_evaluation")
  
  kbpNums = np.shape(kbpTestList[1])[0]; kbpTestRange= int(kbpNums*0.9)
  print kbpNums
  valRet=[]
  testRet=[]

  for i in range(len(kbpTestList)):
    if i==5:
      valRet.append([kbpTestList[i][0][kbpTestRange:kbpNums],kbpTestList[i][1][kbpTestRange:kbpNums]])
      testRet.append([kbpTestList[i][0][0:kbpTestRange],kbpTestList[i][1][0:kbpTestRange]])
    else:
      valRet.append(kbpTestList[i][kbpTestRange:kbpNums])
      testRet.append(kbpTestList[i][0:kbpTestRange])
    
  return valRet,testRet
  
  

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--features', type=str, help='0,1,2,3', required=True)
  
  data_args = parser.parse_args()

  features = data_args.features
  
  train_input, train_out, train_ent_mention_index, train_ent_mention_link_feature, \
  train_ent_mention_tag, train_ent_relcoherent,train_ent_linking_type,\
  train_ent_linking_candprob,train_ent_surfacewordv_feature = getData("kbp_training")
  print 'train_input shape:',np.shape(train_input)
  
  
  valRet,testRet = getDataTestSets()
  
  test_input, test_out, test_ent_mention_index, test_ent_mention_link_feature, \
  test_ent_mention_tag, test_ent_relcoherent,test_ent_linking_type,\
  test_ent_linking_candprob,test_ent_surfacewordv_feature =  testRet
  print 'test_input shape:',np.shape(test_input)
  
  val_input, val_out, val_ent_mention_index, val_ent_mention_link_feature, \
  val_ent_mention_tag, val_ent_relcoherent,val_ent_linking_type,\
  val_ent_linking_candprob,val_ent_surfacewordv_feature =  valRet
  print 'val_input shape:',np.shape(val_input)
  

  #function: lstm_output from seqLSTM
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=6,inter_op_parallelism_threads=6)
  config.gpu_options.allow_growth=True
  sess_ner = tf.InteractiveSession(config=config)
  
  nerInstance = nameEntityRecognition(sess_ner,'checkpoint','aida')
  
  batch_size = 64
  test_batch_size = 1000
  
  
  
#  lstm_output_test_list = [] ; lstm_output_train_list = []
#  for ptr_test in xrange(0,len(test_input),test_batch_size):
#    _,lstm_output_test = nerInstance.getEntityRecognition(test_input[ptr_test:min(ptr_test+test_batch_size,len(test_input))],test_out[ptr_test:min(ptr_test+test_batch_size,len(test_input))])
#    lstm_output_test_list.append(lstm_output_test)
    
    
#  for ptr in xrange(0,len(train_input),batch_size):
#    _,lstm_output = nerInstance.getEntityRecognition(train_input[ptr:min(ptr+batch_size,len(train_input))],train_out[ptr:min(ptr+batch_size,len(train_input))])
#    lstm_output_train_list.append(lstm_output)
    
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  testflag = 'ace'
  with tf.Session(config=config) as sess:
    modelNEL = ctxSum(args,features)  #build named entity linking models
    
    optimizer = tf.train.AdamOptimizer(0.001)
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.005
    loss_linking = modelNEL.linking_loss  + lossL2
    print 'tvars_linking:',tvars
    grads,_ = tf.clip_by_global_norm(tf.gradients(loss_linking,tvars),10)
    train_op_linking = optimizer.apply_gradients(zip(grads,tvars))
    sess.run(tf.global_variables_initializer())
    
    if modelNEL.load(sess,args.restore,"KBP_"+features):
      print "[*] ctxSum is loaded..."
    else:
      print "[*] There is no checkpoint for ctxSum"
    max_accracy_test=0
   
    for e in range(200):
      print 'Epoch: %d------------' %(e)
      id_epoch=-1
      for ptr in xrange(0,len(train_input),batch_size):
        accuracy_test_list=[];ent_mention_linking_tag_lists=[]; pred_lists = []
        
        id_epoch += 1
        _,lstm_output_val = nerInstance.getEntityRecognition(val_input,val_out,'val')
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
            ent_mention_lstm_feature,candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,ent_surfacewordv_feature = \
               getLinkingFeature(args,lstm_output_val,val_ent_mention_index,val_ent_mention_tag,
                                 val_ent_relcoherent,val_ent_mention_link_feature,val_ent_linking_type,
                                 val_ent_linking_candprob,val_ent_surfacewordv_feature,0,0,flag='kbp_val')
        loss2,lossl2,accuracy,pred,w1,w2,w3,w4,w5 = sess.run([loss_linking,lossL2,modelNEL.accuracy,modelNEL.prediction,modelNEL.w1,modelNEL.w2,modelNEL.w3,modelNEL.w4,modelNEL.w5],
                                   {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                     modelNEL.candidate_ent_coherent_feature_ngd:candidate_ent_relcoherent_feature_ngd,
                                     modelNEL.candidate_ent_coherent_feature_fb:candidate_ent_relcoherent_feature_fb,
                                    modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                    modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                    modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                    modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature,
                                    modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                    modelNEL.keep_prob:1.0
                                   })
        
        if accuracy > max_accracy_test:
          max_accracy_test = accuracy
          f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
          print '---------------'
          print 'w1:%f, w2:%f, w3:%f, w4:%f, w5:%f' %(w1,w2,w3,w4,w5)
          print 'val loss:',loss2,lossl2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
              
          id_test = 0
          for ptr_test in xrange(0,len(test_input),test_batch_size):
            #lstm_output_test =lstm_output_test_list[id_test]
            _,lstm_output_test = nerInstance.getEntityRecognition(test_input[ptr_test:min(ptr_test+test_batch_size,len(test_input))],test_out[ptr_test:min(ptr_test+test_batch_size,len(test_input))],'test')
            
            id_test += 1
            ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
              ent_mention_lstm_feature,candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,ent_surfacewordv_feature = \
                 getLinkingFeature(args,lstm_output_test,test_ent_mention_index,test_ent_mention_tag,
                                   test_ent_relcoherent,test_ent_mention_link_feature,test_ent_linking_type,
                                   test_ent_linking_candprob,test_ent_surfacewordv_feature,test_batch_size,ptr_test,flag='kbp_test')
            if len(ent_mention_lstm_feature)==0:
              continue
            
            loss2,lossl2,accuracy,pred,w1,w2,w3,w4,w5 = sess.run([loss_linking,lossL2,modelNEL.accuracy,modelNEL.prediction,modelNEL.w1,modelNEL.w2,modelNEL.w3,modelNEL.w4,modelNEL.w5],
                                     {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                       modelNEL.candidate_ent_coherent_feature_ngd:candidate_ent_relcoherent_feature_ngd,
                                       modelNEL.candidate_ent_coherent_feature_fb:candidate_ent_relcoherent_feature_fb,
                                      modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                      modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                      modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                      modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature,
                                      modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                      modelNEL.keep_prob:1.0
                                     })
            if ptr_test==0:
              ent_mention_linking_tag_lists=ent_mention_linking_tag_list;
              pred_lists = pred
            else:
              ent_mention_linking_tag_lists = np.concatenate((ent_mention_linking_tag_lists,ent_mention_linking_tag_list))
              pred_lists = np.concatenate((pred_lists,pred))
            accuracy_test_list.append(accuracy)
          accuracy = np.average(accuracy_test_list)
          cPickle.dump(pred_lists,open('data/kbp/LDC2017EDL/data/2014/evaluation/features/'+str(args.candidate_ent_num)+'/entityLinkingResult.p'+features,'wb'))
          f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_lists,1),np.argmax(pred_lists,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_lists,1),np.argmax(pred_lists,1),average='macro')
  
          print 'test loss:',loss2,lossl2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
          print '---------------'
          
        #lstm_output = lstm_output_train_list[id_epoch]
        
        _,lstm_output = nerInstance.getEntityRecognition(train_input[ptr:min(ptr+batch_size,len(train_input))],train_out[ptr:min(ptr+batch_size,len(train_input))],'train')
        #lstm_output_train_list.append(lstm_output)
        #lstm_output = lstm_output_train[ptr:min(ptr+10,len(train_input))];
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
         ent_mention_lstm_feature,candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,ent_surfacewordv_feature= \
                                                getLinkingFeature(args,lstm_output,train_ent_mention_index,train_ent_mention_tag,
                                                train_ent_relcoherent,train_ent_mention_link_feature,train_ent_linking_type,
                                                train_ent_linking_candprob,train_ent_surfacewordv_feature,batch_size,ptr,flag='kbp_train')
        if len(ent_mention_lstm_feature)==0:
          continue
        _,loss2,accuracy,pred = sess.run([train_op_linking,loss_linking,modelNEL.accuracy,modelNEL.prediction],
                               {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                 modelNEL.candidate_ent_coherent_feature_ngd:candidate_ent_relcoherent_feature_ngd,
                                 modelNEL.candidate_ent_coherent_feature_fb:candidate_ent_relcoherent_feature_fb,
                                modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature, 
                                modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                modelNEL.keep_prob:0.5
                               })
        if id_epoch % 5==0:
          f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
          print 'train loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
if __name__=="__main__":
  tf.app.run()

