# -*- coding: utf-8 -*-

import sys
import time
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
from PhraseRecord import EntRecord
from sklearn import preprocessing
import cPickle as pkl
import numpy as np
import time
class nelInputUtils(object):
  def __init__(self,dims,flag='train'):
    dir_path ='/home/wjs/demo/entityType/informationExtract/data/aida/features'
    
    if flag=='train':
      print 'load train'
      stime = time.time()
      self.emb = pkl.load(open(dir_path+'/train_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/train_tag.p'+dims,'rb'))
      self.ent_linking  = pkl.load(open(dir_path+'/train_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open(dir_path+'/train_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open(dir_path+'/train_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open(dir_path+'/train_ent_linking_candprob.p','rb'))
      #self.ent_surfacewordv_feature = pkl.load(open(dir_path+'/train_ent_mentwordv.p','rb'))
      print 'load train cost time:',time.time()-stime
      '''
      if flag=='train':
        self.TFfileName = dir_path+'/trainNEL.tfrecord'
        self.nelShapeFile = dir_path +'/trainNEL.shape'
      '''
    elif flag=='testa':
      print 'load testa'
      stime = time.time()
      self.ent_linking  = pkl.load(open(dir_path+'/testa_ent_linking.p','rb'))
      self.ent_relcoherent  = pkl.load(open(dir_path+'/testa_ent_relcoherent.p','rb'))
      self.emb = pkl.load(open(dir_path+'/testa_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/testa_tag.p'+dims,'rb'))
      self.ent_linking_type  = pkl.load(open(dir_path+'/testa_ent_linking_type.p','rb'))
      self.ent_linking_candprob = pkl.load(open(dir_path+'/testa_ent_linking_candprob.p','rb'))
      
      #self.ent_surfacewordv_feature = pkl.load(open(dir_path+'/testa_ent_mentwordv.p','rb'))
      print 'load testa cost time:',time.time()-stime
    elif flag=='testb':
      print 'load testb'
      stime = time.time()
      self.emb = pkl.load(open(dir_path+'/testb_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/testb_tag.p'+dims,'rb'))
      self.ent_linking  = pkl.load(open(dir_path+'/testb_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open(dir_path+'/testb_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open(dir_path+'/testb_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open(dir_path+'/testb_ent_linking_candprob.p','rb'))
      #self.ent_surfacewordv_feature = pkl.load(open(dir_path+'/testb_ent_mentwordv.p','rb'))
      print 'load testb cost time:',time.time()-stime
    elif flag=='ace':
      print 'load ace'
      stime = time.time()
      self.emb = pkl.load(open('data/ace/features/ace_embed.p100','rb'))
      self.ent_linking  = pkl.load(open('data/ace/features/ace_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open('data/ace/features/ace_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open('data/ace/features/ace_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open('data/ace/features/ace_ent_linking_candprob.p','rb')) 
      #self.ent_surfacewordv_feature = pkl.load(open('data/ace/features/ace_ent_mentwordv.p','rb'))
      print 'load ace cost time:',time.time()-stime
    elif flag=='msnbc':
      print 'load msnbc'
      stime = time.time()
      self.emb = pkl.load(open('data/msnbc/features/msnbc_embed.p100','rb'))
      self.ent_linking  = pkl.load(open('data/msnbc/features/msnbc_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open('data/msnbc/features/msnbc_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open('data/msnbc/features/msnbc_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open('data/msnbc/features/msnbc_ent_linking_candprob.p','rb')) 
      #self.ent_surfacewordv_feature = pkl.load(open('data/msnbc/features/msnbc_ent_mentwordv.p','rb'))
      print 'load msnbc cost time:',time.time()-stime
    else:
      print 'can not load enttiy linking features'

def getLinkingFeature(args,lstm_output,ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ent_linking_candprob,ptr,flag='train'):
#def getLinkingFeature(ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ptr):
  ent_mention_linking_tag_list = []
  candidate_ent_linking_feature=[]
  #candidate_ent_surfacewordv_feature=[]
  candidate_ent_type_feature=[]
  candidate_ent_prob_feature=[]
  candidate_ent_relcoherent_feature=[]
  ent_mention_lstm_feature = []
  allLenght = len(ent_mention_index)
  sequence_length = []
  ent_index=[]
  lstm_index=[]
  dim3 = np.shape(lstm_output)[2]
  if 'trainmsnbc' == flag:
    last_range = min(ptr+10,allLenght)
  elif 'train' == flag:
    last_range = min(ptr+args.batch_size,allLenght)
#  if 'train' in flag:
#    last_range = min(ptr+args.batch_size,allLenght)
  else:
    last_range = allLenght
  #last_range = allLenght
  for ids in xrange(ptr,last_range):
    tagid = -1
    
    for ent_item in ent_mention_index[ids]:
      tagid += 1
      #if np.sum(ent_mention_tag[ids][tagid]) != 0:
      ent_index.append([ent_item[0],ent_item[1]]) 
      lstm_index.append(ids-ptr)
      
      if flag in ['train','testa','testb','traintotal']:
        ent_mention_linking_tag = np.zeros((args.candidate_ent_num,))
        if len(ent_mention_tag[ids][tagid])!=0:
          tag_index = ent_mention_tag[ids][tagid][0]
          ent_mention_linking_tag[tag_index] = 1
      else:  
        ent_mention_linking_tag = np.asarray(ent_mention_tag[ids][tagid],dtype=np.float32)
      ent_mention_linking_tag_list.append(ent_mention_linking_tag)
      candidate_ent_relcoherent_item = np.asarray(ent_relcoherent[ids][tagid],dtype=np.float32)
      candidate_ent_relcoherent_feature.append(candidate_ent_relcoherent_item)
      
      candidate_num = len(ent_mention_link_feature[ids][tagid])/100 
      ent_linking_candidates =  np.concatenate((np.asarray(np.reshape(ent_mention_link_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), np.zeros((max(0,args.candidate_ent_num-candidate_num),100),dtype=np.float32)))
      #ent_mentWordV_candidates =  np.concatenate((np.asarray(np.reshape(ent_surfacewordv_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), np.zeros((max(0,args.candidate_ent_num-candidate_num),100),dtype=np.float32)))
      
      ent_type_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_type[ids][tagid],(candidate_num,args.figer_type_num)),dtype=np.float32),np.zeros((max(0,args.candidate_ent_num-candidate_num),args.figer_type_num),dtype=np.float32)))
      ent_prob_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_candprob[ids][tagid],(candidate_num,3)),dtype=np.float32),np.zeros((max(0,args.candidate_ent_num-candidate_num),3),dtype=np.float32)))
                                      
      candidate_ent_linking_feature.append(ent_linking_candidates)
      #candidate_ent_surfacewordv_feature.append(ent_mentWordV_candidates)
      candidate_ent_type_feature.append(ent_type_candidates)
      '''
      @time:2017/3/21 we need to l2 normalize the vectors!
      '''
      ent_prob_candidates = preprocessing.normalize(ent_prob_candidates,norm='l2',axis=0)
      candidate_ent_prob_feature.append(ent_prob_candidates)
      if ent_item[1]-ent_item[0] >5:
        ent_mention_lstm_feature.append(lstm_output[ids-ptr][ent_item[0]:ent_item[0]+5])
      else:
        ent_mention_lstm_feature.append(np.concatenate([lstm_output[ids-ptr][ent_item[0]:ent_item[1]],np.zeros([5-(ent_item[1]-ent_item[0]),dim3])]))
        
      #ent_mention_lstm_feature.append(np.sum(lstm_output[ids-ptr][ent_item[0]:ent_item[1]],axis=0))
      #ent_mention_lstm_feature.append(lstm_output[ids-ptr][ent_item[0]:ent_item[1]])
      sequence_length.append(ent_item[1]-ent_item[0])
  
  ent_mention_linking_tag_list = np.asarray(ent_mention_linking_tag_list,dtype=np.int64)
  candidate_ent_relcoherent_feature = np.asarray(candidate_ent_relcoherent_feature,dtype=np.float)
  candidate_ent_linking_feature = np.asarray(candidate_ent_linking_feature,dtype=np.float)
  #candidate_ent_surfacewordv_feature = np.asarray(candidate_ent_surfacewordv_feature,dtype=np.float)
  candidate_ent_type_feature = np.asarray(candidate_ent_type_feature,dtype=np.float)
  candidate_ent_prob_feature = np.asarray(candidate_ent_prob_feature,dtype=np.float)
  ent_mention_lstm_feature = np.asarray(ent_mention_lstm_feature,dtype=np.float)
  #print np.shape(ent_mention_lstm_feature)
  sequence_length = np.asarray(sequence_length)
  
  #print 'ent_mention_linking_tag_list:',np.shape(ent_mention_linking_tag_list)
  return ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
         ent_mention_lstm_feature,candidate_ent_relcoherent_feature#,candidate_ent_surfacewordv_feature#,sequence_length
  
  
