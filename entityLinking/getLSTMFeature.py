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
def getData(dataTag):
 
  dataUtils = inputUtils(args.rawword_dim,dataTag)
  data_input = dataUtils.emb; 
  data_entliking= dataUtils.ent_linking;data_ent_mention_index = data_entliking['ent_mention_index'];
  data_ent_mention_link_feature=data_entliking['ent_mention_link_feature'];data_ent_mention_tag = data_entliking['ent_mention_tag']; 
  data_ent_relcoherent = dataUtils.ent_relcoherent
  data_ent_linking_type = dataUtils.ent_linking_type; 
  data_ent_linking_candprob = dataUtils.ent_linking_candprob
  #data_ent_surfacewordv_feature = dataUtils.ent_surfacewordv_feature
  
  data_shape = np.shape(data_input)
  if dataTag in ['train','testa','testb']:
    data_out = np.argmax(dataUtils.tag,2) 
  else:
    data_out = np.zeros([data_shape[0],data_shape[1],args.class_size],dtype=np.float32)  
    data_out = np.argmax(data_out,2)
  
  return data_input, data_out, data_ent_mention_index, data_ent_mention_link_feature, data_ent_mention_tag, data_ent_relcoherent,data_ent_linking_type,data_ent_linking_candprob#,data_ent_surfacewordv_feature


'''
@convert ace and msnbc split into train and test()
'''
def getDataSets():
  
  aceFeatureList = getData("ace")
  aceNums = np.shape(aceFeatureList[1])[0]; 
  print aceNums
  msnbcFeatureList = getData("msnbc")
  msnbcNums = np.shape(msnbcFeatureList[1])[0]; 
  print msnbcNums
  trainFeatureList = getData("train")
  print len(trainFeatureList[1])

  return trainFeatureList,aceFeatureList,msnbcFeatureList
  #return trainRet,aceTestRet,msnbcTestRet
  
def main(_):
  pp.pprint(flags.FLAGS.__flags)

  print 'start to load data...'
  start_time = time.time()
  
  testa_input, testa_out, testa_ent_mention_index, testa_ent_mention_link_feature, \
  testa_ent_mention_tag, testa_ent_relcoherent,testa_ent_linking_type,\
  testa_ent_linking_candprob =  getData("testa") 
  
  trainList,aceTestList,msnbcTestList  = getDataSets()
  
  train_input, train_out, train_ent_mention_index, train_ent_mention_link_feature, \
  train_ent_mention_tag, train_ent_relcoherent,train_ent_linking_type,\
  train_ent_linking_candprob = trainList
  
  ace_input, ace_out, ace_ent_mention_index, ace_ent_mention_link_feature, \
  ace_ent_mention_tag, ace_ent_relcoherent,ace_ent_linking_type,\
  ace_ent_linking_candprob =  aceTestList
  
  msnbc_input, msnbc_out, msnbc_ent_mention_index, msnbc_ent_mention_link_feature, \
  msnbc_ent_mention_tag, msnbc_ent_relcoherent,msnbc_ent_linking_type,\
  msnbc_ent_linking_candprob =  msnbcTestList
  

  
  
  testb_input, testb_out, testb_ent_mention_index, testb_ent_mention_link_feature, \
  testb_ent_mention_tag, testb_ent_relcoherent,testb_ent_linking_type,\
  testb_ent_linking_candprob =  getData("testb")
  print 'cost:', time.time()-start_time,' to load data'
  
  #function: lstm_output from seqLSTM
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess_ner = tf.InteractiveSession(config=config)
  
  nerInstance = nameEntityRecognition(sess_ner,'data/aida/','aida')
  
  lstm_output_ace = nerInstance.getEntityRecognition(ace_input,ace_out)
  lstm_output_msnbc = nerInstance.getEntityRecognition(msnbc_input,msnbc_out)
  lstm_output_testa = nerInstance.getEntityRecognition(testa_input,testa_out)
  lstm_output_testb = nerInstance.getEntityRecognition(testb_input,testb_out)
  lstm_output_train = nerInstance.getEntityRecognition(train_input,train_out)
  
  cPickle.dump(lstm_output_ace,open('data/lstmout/lstm_output_ace.p','wb'))
  cPickle.dump(lstm_output_msnbc,open('data/lstmout/lstm_output_msnbc.p','wb'))
  cPickle.dump(lstm_output_testa,open('data/lstmout/lstm_output_testa.p','wb'))
  cPickle.dump(lstm_output_testb,open('data/lstmout/lstm_output_testb.p','wb'))
  cPickle.dump(lstm_output_train,open('data/lstmout/lstm_output_train.p','wb'))
  
  sess_ner.close()
  

if __name__=="__main__":
  tf.app.run()
