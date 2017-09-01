# -*- coding: utf-8 -*-
'''
@editor: wujs
function: we add entity linking module
revise: 2017/1/8
'''

import tensorflow as tf
import time
from model import ctxSum
from sklearn.metrics import f1_score
from utils import nelInputUtils as inputUtils
from utils import getLinkingFeature,genEntMentMask
import numpy  as np
import pprint
import time
import cPickle
import argparse
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch",100,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",10,"batch size of training")
flags.DEFINE_string("datasets","ace","hidden dimension of rnn")
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
  aceFeatureList = getData("ace")
  aceNums = np.shape(aceFeatureList[1])[0]; aceTrainRange= int(aceNums*0.7)
  print aceNums
  msnbcFeatureList = getData("msnbc")
  msnbcNums = np.shape(msnbcFeatureList[1])[0]; msnbcTrainRange= int(msnbcNums*0.7)
  print msnbcNums
  aceTestRet=[]
  msnbcTestRet=[]
  TrainRet=[]
  #for i in range(len(aceFeatureList)):
    #aceTestRet.append(aceFeatureList[i][aceTrainRange:aceNums])
    #msnbcTestRet.append(msnbcFeatureList[i][msnbcTrainRange:msnbcNums])
    #TrainRet.append(np.concatenate((aceFeatureList[i][0:aceTrainRange],msnbcFeatureList[i][0:msnbcTrainRange])))
    #TrainRet.append(aceFeatureList[i])
  return aceFeatureList,msnbcFeatureList,msnbcFeatureList
  
  
'''
@convert ace and msnbc split into train and test()
'''
def getDataSets():
  
  aceFeatureList = getData("ace")
  aceNums = np.shape(aceFeatureList[1])[0]; aceTrainRange= int(aceNums*0.7)
  print aceNums
  msnbcFeatureList = getData("msnbc")
  msnbcNums = np.shape(msnbcFeatureList[1])[0]; msnbcTrainRange= int(msnbcNums*0.7)
  print msnbcNums
  trainFeatureList = getData("train")
  print len(trainFeatureList[1])
#  trainRet = []
#  aceTestRet=[]
#  msnbcTestRet=[]
#  for i in range(len(trainFeatureList)):
#    trainRet.append(np.concatenate((trainFeatureList[i],aceFeatureList[i][0:aceTrainRange],msnbcFeatureList[i][0:msnbcTrainRange])))
#    aceTestRet.append(aceFeatureList[i][aceTrainRange:aceNums])
#    msnbcTestRet.append(msnbcFeatureList[i][msnbcTrainRange:msnbcNums])
  return trainFeatureList,aceFeatureList,msnbcFeatureList 
  #return trainRet,aceTestRet,msnbcTestRet

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--features', type=str, help='0,1,2,3', required=True)
  
  data_args = parser.parse_args()

  features = data_args.features
  
  
  aceTestList,msnbcTestList,trainList = getDataTestSets()
  
  ace_input, ace_out, ace_ent_mention_index, ace_ent_mention_link_feature, \
  ace_ent_mention_tag, ace_ent_relcoherent,ace_ent_linking_type,\
  ace_ent_linking_candprob,ace_ent_surfacewordv_feature =  aceTestList
  ace_example_nums = np.shape(ace_input)[0]
  ace_entMentIndex,ace_entMent_length = genEntMentMask(args,ace_example_nums,ace_ent_mention_index)
  
  
  train_input, train_out, train_ent_mention_index, train_ent_mention_link_feature, \
  train_ent_mention_tag, train_ent_relcoherent,train_ent_linking_type,\
  train_ent_linking_candprob,train_ent_surfacewordv_feature = trainList
  
  
  
  
  msnbc_input, msnbc_out, msnbc_ent_mention_index, msnbc_ent_mention_link_feature, \
  msnbc_ent_mention_tag, msnbc_ent_relcoherent,msnbc_ent_linking_type,\
  msnbc_ent_linking_candprob,msnbc_ent_surfacewordv_feature =  msnbcTestList
  msnbc_example_nums = np.shape(msnbc_input)[0]
  msnbc_entMentIndex,msnbc_entMent_length = genEntMentMask(args,msnbc_example_nums,msnbc_ent_mention_index)
  
#  #function: lstm_output from seqLSTM
#  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=6,inter_op_parallelism_threads=6)
#  config.gpu_options.allow_growth=True
#  sess_ner = tf.InteractiveSession(config=config)
#  
#  nerInstance = nameEntityRecognition(sess_ner,'checkpoint','aida')
#  lstm_output_train = nerInstance.getEntityRecognition(train_input,train_out)
#  lstm_output_ace = nerInstance.getEntityRecognition(ace_input,ace_out)
#  lstm_output_msnbc = nerInstance.getEntityRecognition(msnbc_input,msnbc_out)
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  testflag = 'ace'
  with tf.Session(config=config) as sess:
    modelNEL = ctxSum(args,features)  #build named entity linking models
    
    optimizer = tf.train.AdamOptimizer(0.001)
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='ctxSum')
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]) * 0.01
    loss_linking = modelNEL.linking_loss  #+ lossL2
    print 'tvars_linking:',tvars
    grads,_ = tf.clip_by_global_norm(tf.gradients(loss_linking,tvars),10)
    train_op_linking = optimizer.apply_gradients(zip(grads,tvars))
    sess.run(tf.global_variables_initializer())
    
    if modelNEL.load(sess,args.restore,"aida_"+features):
      print "[*] ctxSum is loaded..."
    else:
      print "[*] There is no checkpoint for ctxSum"
    max_accracy_msnbc = 0
    max_accracy_ace=0
    for e in range(200):
      print 'Epoch: %d------------' %(e)
      for ptr in xrange(0,len(train_input),args.batch_size):
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
        candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,ent_surfacewordv_feature = \
               getLinkingFeature(args,ace_ent_mention_index,ace_ent_mention_tag,
                                 ace_ent_relcoherent,ace_ent_mention_link_feature,ace_ent_linking_type,
                                 ace_ent_linking_candprob,ace_ent_surfacewordv_feature,ace_example_nums,0,flag='ace')
        loss2,lossl2,accuracy,pred,w1,w2,w3,w4,w5 = sess.run([loss_linking,lossL2,modelNEL.accuracy,modelNEL.prediction,modelNEL.w1,modelNEL.w2,modelNEL.w3,modelNEL.w4,modelNEL.w5],
                                   {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                     modelNEL.candidate_ent_coherent_feature_ngd:candidate_ent_relcoherent_feature_ngd,
                                     modelNEL.candidate_ent_coherent_feature_fb:candidate_ent_relcoherent_feature_fb,
                                    modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                    modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                    modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                    modelNEL.input_data:ace_input,
                                    modelNEL.entMentIndex:ace_entMentIndex,
                                    #modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature,
                                    modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                    modelNEL.keep_prob:1.0
                                   })
        if accuracy > max_accracy_ace:
          if testflag=='ace':
            cPickle.dump(pred,open('data/ace/features/'+str(args.candidate_ent_num)+'/entityLinkingResult.p'+features,'wb'))
          max_accracy_ace = accuracy
          f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
          print '---------------'
          print 'w1:%f, w2:%f, w3:%f, w4:%f, w5:%f' %(w1,w2,w3,w4,w5)
          print 'ace loss:',loss2,lossl2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
              
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,ent_surfacewordv_feature = \
    getLinkingFeature(args,msnbc_ent_mention_index,msnbc_ent_mention_tag,
                       msnbc_ent_relcoherent,msnbc_ent_mention_link_feature,msnbc_ent_linking_type,
                       msnbc_ent_linking_candprob,msnbc_ent_surfacewordv_feature,msnbc_example_nums,0,flag='msnbc')
        loss2,lossl2,accuracy,pred = sess.run([loss_linking,lossL2,modelNEL.accuracy,modelNEL.prediction],
                         {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                          #modelNEL.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                          modelNEL.candidate_ent_coherent_feature_ngd:candidate_ent_relcoherent_feature_ngd,
                          modelNEL.candidate_ent_coherent_feature_fb:candidate_ent_relcoherent_feature_fb,
                          modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                          modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                          modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                          modelNEL.input_data:msnbc_input,
                          modelNEL.entMentIndex:msnbc_entMentIndex,
                          #modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature,
                          modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                          modelNEL.keep_prob:1.0
                         })
        
        if accuracy > max_accracy_msnbc:
          if testflag=='msnbc':
            cPickle.dump(pred,open('data/msnbc/features/'+str(args.candidate_ent_num)+'/entityLinkingResult.p'+str(features),'wb'))   
          max_accracy_msnbc = accuracy
          f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
          print '------------'
          print 'msnbc loss:',loss2,lossl2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
          #if max_accracy_msnbc > 0.75:
          #  modelNEL.save(sess,args.restore,"aida_"+features)
        batch_size = min(np.shape(train_input)[0],ptr+args.batch_size) -ptr
        ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
         candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,ent_surfacewordv_feature= \
                                                getLinkingFeature(args,train_ent_mention_index,train_ent_mention_tag,
                                                train_ent_relcoherent,train_ent_mention_link_feature,train_ent_linking_type,
                                                train_ent_linking_candprob,train_ent_surfacewordv_feature,batch_size,ptr,flag='trainmsnbc')
        if len(ent_mention_linking_tag_list)==0:
          continue
        train_entMentIndex,train_entMent_length = genEntMentMask(args,np.shape(train_input[ptr:min(np.shape(train_input)[0],ptr+args.batch_size)])[0],train_ent_mention_index[ptr:min(np.shape(train_input)[0],ptr+args.batch_size)]) 
        
        _,loss2,accuracy,pred = sess.run([train_op_linking,loss_linking,modelNEL.accuracy,modelNEL.prediction],
                               {modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                 modelNEL.candidate_ent_coherent_feature_ngd:candidate_ent_relcoherent_feature_ngd,
                                 modelNEL.candidate_ent_coherent_feature_fb:candidate_ent_relcoherent_feature_fb,
                                modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                modelNEL.input_data:train_input[ptr:min(np.shape(train_input)[0],ptr+args.batch_size)],
                                modelNEL.entMentIndex:train_entMentIndex,
                                #modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature, 
                                modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                modelNEL.keep_prob:0.5
                               })
        f1_micro,f1_macro=f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='micro'),f1_score(np.argmax(ent_mention_linking_tag_list,1),np.argmax(pred,1),average='macro')
        #print 'train loss:',loss2,' accuracy:',accuracy,' f1_micro:',f1_micro,' f1_macro:',f1_macro
if __name__=="__main__":
  tf.app.run()
