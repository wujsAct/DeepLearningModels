# -*- coding: utf-8 -*-
'''
@editor: wujs
function: we add entity linking module
revise: 2017/1/19
'''
import sys
sys.path.append('utils')

import tensorflow as tf
import argparse
from model import ctxSum
from entityRecog import nameEntityRecognition,pp,flags,args #get seqLSTM features
from sklearn.metrics import f1_score
from utils import nelInputUtils as inputUtils
from utils import getLinkingFeature
import numpy as np
import cPickle

class namedEntityLinking(object):
  def __init__(self,sess,dir_path,data_tag):
    self.sess = sess
    self.modelNEL = ctxSum(args)  #build named entity linking models
    self.loss_linking = self.modelNEL.linking_loss
    self.dir_path = dir_path
    self.data_tag = data_tag
    #utilize the entity linking model trained on aida to get the linking results for ace datasets
    if self.modelNEL.load(self.sess,args.restore,"aida"):
      print "[*] ctxSum is loaded..."
    else:
      print "[*] There is no checkpoint for aida ctxSum model..."
      
  def getEntityLinking(self,features):
    test_ent_mention_link_feature= features['test_ent_mention_link_feature']
    test_ent_mention_tag = features['test_ent_mention_tag']
    test_ent_mention_index = features['test_ent_mention_index']
    test_ent_relcoherent = features['test_ent_relcoherent']
    test_ent_linking_type = features['test_ent_linking_type']
    test_ent_linking_candprob = features['test_ent_linking_candprob']
    #test_ent_surfacewordv_feature = features['test_ent_surfacewordv_feature']
    
    lstm_output_test=features['lstm_output_test']
    ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,ent_mention_lstm_feature,candidate_ent_relcoherent_feature = \
                                                getLinkingFeature(args,lstm_output_test,test_ent_mention_index,test_ent_mention_tag,\
                                                test_ent_relcoherent,test_ent_mention_link_feature,test_ent_linking_type,test_ent_linking_candprob,0,flag='traintotal')
    print 'ent_mention_linking_tag_list:',np.shape(ent_mention_linking_tag_list)
    print 'candidate_ent_type_feature shape:',np.shape(candidate_ent_type_feature)
    loss2,accuracy,pred = self.sess.run([self.loss_linking,self.modelNEL.accuracy,self.modelNEL.prediction],
                               {self.modelNEL.ent_mention_linking_tag:ent_mention_linking_tag_list,
                                self.modelNEL.candidate_ent_coherent_feature:candidate_ent_relcoherent_feature,
                                self.modelNEL.candidate_ent_linking_feature:candidate_ent_linking_feature,
                                self.modelNEL.candidate_ent_type_feature:candidate_ent_type_feature,
                                self.modelNEL.candidate_ent_prob_feature:candidate_ent_prob_feature,
                                self.modelNEL.ent_mention_lstm_feature:ent_mention_lstm_feature,
                                #self.modelNEL.ent_surfacewordv_feature:ent_surfacewordv_feature,
                                self.modelNEL.keep_prob:1
                               })
    print accuracy 
    cPickle.dump(pred,open(self.dir_path+'entityLinkingResult.p','wb'))                                           
    
    #print pred
    
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
  parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
  data_args = parser.parse_args()
  
  data_tag=data_args.data_tag
  dir_path=data_args.dir_path
  
  testUtils = inputUtils(args.rawword_dim,data_tag)
  test_input = testUtils.emb;
  print np.shape(test_input)
  testShape = np.shape(test_input)
  test_input  = np.concatenate((test_input,np.zeros([testShape[0],max(0,124-testShape[1]),testShape[2]])),axis=1)
  testShape = np.shape(test_input)
  print testShape
  assert testShape[1]==124
  
  test_out = np.argmax(np.zeros([testShape[0],testShape[1],args.class_size],dtype=np.float32),2)
  #test_out = testaUtils.tag;
  test_entliking= testUtils.ent_linking;
  test_ent_mention_index = test_entliking['ent_mention_index'];
  test_ent_mention_link_feature=test_entliking['ent_mention_link_feature'];
  test_ent_mention_tag = test_entliking['ent_mention_tag']; 
  test_ent_relcoherent = testUtils.ent_relcoherent
  test_ent_linking_type = testUtils.ent_linking_type
  test_ent_linking_candprob = testUtils.ent_linking_candprob
  #test_ent_surfacewordv_feature = testUtils.ent_surfacewordv_feature
  
  
  features = {}
  features['test_ent_mention_index']=test_ent_mention_index
  features['test_ent_mention_link_feature'] = test_ent_mention_link_feature
  features['test_ent_mention_tag'] = test_ent_mention_tag
  features['test_ent_relcoherent']=test_ent_relcoherent
  features['test_ent_linking_type']=test_ent_linking_type
  features['test_ent_linking_candprob']=test_ent_linking_candprob
  #features['test_ent_surfacewordv_feature'] = test_ent_surfacewordv_feature
  
  '''function: lstm_output from seqLSTM'''
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess_ner = tf.InteractiveSession(config=config)
  nerInstance = nameEntityRecognition(sess_ner,dir_path,data_tag)

  lstm_output_test = nerInstance.getEntityRecognition(test_input,test_out)
  cPickle.dump(lstm_output_test,open(dir_path+data_tag+'lstm_output.p','wb'))
  sess_ner.close() 
  exit(0)
  features['lstm_output_test']=lstm_output_test  
  
  config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
  config.gpu_options.allow_growth=True
  sess = tf.InteractiveSession(config=config);
  nelClass = namedEntityLinking(sess,dir_path,data_tag);
  nelClass.getEntityLinking(features)
  sess.close()
    
  
  