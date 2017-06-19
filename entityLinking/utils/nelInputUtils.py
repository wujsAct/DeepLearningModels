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
  
  def padZeros(self,sentence_final,dims=110,max_sentence_length=250):
    for i in range(len(sentence_final)):
      offset = max_sentence_length-len(sentence_final[i])
      sentence_final[i] =np.concatenate((sentence_final[i], [[0]*dims]*offset))
    return np.asarray(sentence_final)
  
  def __init__(self,dims,flag='train'):
    if flag in ['train','testa','testb']:
      candidateEntNum=90
      dir_path ='/home/wjs/demo/entityType/informationExtract/data/aida/features/'
      feature_path = dir_path +str(candidateEntNum) +'/'
      print 'load ',flag,' datasets.......'
      stime = time.time()
      self.emb = pkl.load(open(dir_path+flag+'_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+flag+'_tag.p'+dims,'rb'))
      self.ent_linking  = pkl.load(open(feature_path+flag+'_ent_linking.p'+str(candidateEntNum),'rb'))
      self.ent_linking_type  = pkl.load(open(feature_path+flag+'_ent_linking_type.p'+str(candidateEntNum),'rb'))
      self.ent_relcoherent_ngd  = pkl.load(open(feature_path+flag+'_ent_relcoherent.p'+str(candidateEntNum),'rb'))
      self.ent_relcoherent_fb = pkl.load(open(feature_path+flag+'_ent_fbrelcoherent.p'+str(candidateEntNum),'rb'))
      self.ent_linking_candprob = pkl.load(open(feature_path+flag+'_ent_linking_candprob.p'+str(candidateEntNum),'rb'))
      self.ent_surfacewordv_feature = pkl.load(open(feature_path+flag+'_ent_mentwordv.p'+str(candidateEntNum),'rb'))
      print 'load '+flag+' cost time:',time.time()-stime
    elif flag=='msnbc' or flag=='ace':
      print 'load '+flag
      stime = time.time()
      candidateEntNum=90
      dir_path = 'data/msnbc/features/'
      feature_path = dir_path +str(candidateEntNum) +'/'+flag
      self.emb = pkl.load(open(dir_path+flag+'_embed.p100','rb'))
      self.ent_linking  = pkl.load(open(feature_path+'_ent_linking.p'+str(candidateEntNum),'rb'))
      self.ent_linking_type  = pkl.load(open(feature_path+'_ent_linking_type.p'+str(candidateEntNum),'rb'))
      self.ent_relcoherent_ngd  = pkl.load(open(feature_path+'_ent_relcoherent.p'+str(candidateEntNum),'rb'))
      self.ent_relcoherent_fb  = pkl.load(open(feature_path+'_ent_fbrelcoherent.p'+str(candidateEntNum),'rb'))
      self.ent_linking_candprob = pkl.load(open(feature_path+'_ent_linking_candprob.p'+str(candidateEntNum),'rb')) 
      self.ent_surfacewordv_feature = pkl.load(open(feature_path+'_ent_mentwordv.p'+str(candidateEntNum),'rb'))
      print 'load msnbc cost time:',time.time()-stime
    elif flag=='kbp_training' or flag=='kbp_evaluation':
      print 'start to load data '+flag 
      dir_path = "data/kbp/LDC2017EDL/data/2014/"
      dataset = flag.split('_')[0]
      tag = flag.split('_')[1]
      dir_path = dir_path + tag +"/features/"
      stime = time.time()
      candidateEntNum=90
      
      feature_path = dir_path +str(candidateEntNum) +'/'
      self.emb = self.padZeros(pkl.load(open(dir_path+dataset+'_embed.p100','rb')))
      print 'self.emb shape:',np.shape(self.emb)
      self.ent_linking  = pkl.load(open(feature_path+dataset+'_ent_linking.p'+str(candidateEntNum),'rb'))
      self.ent_linking_type  = pkl.load(open(feature_path+dataset+'_ent_linking_type.p'+str(candidateEntNum),'rb'))
      self.ent_relcoherent_ngd  = pkl.load(open(feature_path+dataset+'_ent_relcoherent.p'+str(candidateEntNum),'rb'))
      self.ent_relcoherent_fb  = pkl.load(open(feature_path+dataset+'_ent_fbrelcoherent.p'+str(candidateEntNum),'rb'))
      self.ent_linking_candprob = pkl.load(open(feature_path+dataset+'_ent_linking_candprob.p'+str(candidateEntNum),'rb')) 
      self.ent_surfacewordv_feature = pkl.load(open(feature_path+dataset+'_ent_mentwordv.p'+str(candidateEntNum),'rb'))
      
      

#def getLinkingFeature(args,candidateEntNum,lstm_output,ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ent_linking_candprob,ptr,flag='train'):
#  allLenght = len(ent_mention_index)
#  ent_index=[]
#  lstm_index=[]
#  if 'trainmsnbc' == flag:
#    last_range = min(ptr+1,allLenght)
#  elif 'train' == flag:
#    last_range = min(ptr+args.batch_size,allLenght)
##  if 'train' in flag:
##    last_range = min(ptr+args.batch_size,allLenght)
#  else:
#    last_range = allLenght
#  #last_range = allLenght
#  for ids in xrange(ptr,last_range):
#    tagid = -1
#    
#    for ent_item in ent_mention_index[ids]:
#      tagid += 1
#      #if np.sum(ent_mention_tag[ids][tagid]) != 0:
#      ent_index.append([ent_item[0],ent_item[1]]) 
#      lstm_index.append(ids-ptr)
#      
#      candidate_num = len(ent_mention_link_feature[ids][tagid])/100   #get the candidate entity numbers.
#      
#      if candidate_num == 0:
#        continue
#      
#      if flag in ['train','testa','testb','traintotal']:
#        ent_mention_linking_tag = np.zeros((candidate_num,))
#        if len(ent_mention_tag[ids][tagid])!=0:
#          tag_index = ent_mention_tag[ids][tagid][0]
#          ent_mention_linking_tag[tag_index] = 1
#      else:  
#        ent_mention_linking_tag = np.asarray(ent_mention_tag[ids][tagid],dtype=np.float32)[0:candidate_num]
#
#      candidate_ent_relcoherent_feature = np.asarray(ent_relcoherent[ids][tagid],dtype=np.float32)[0:candidate_num]
#      #print 'candidate_ent_relcoherent_feature:',candidate_ent_relcoherent_feature
#      #print ent_relcoherent[ids][tagid]
#      candidate_ent_linking_feature = np.reshape(ent_mention_link_feature[ids][tagid],(candidate_num,100))
#      
#      candidate_ent_type_feature =  np.reshape(ent_linking_type[ids][tagid],(candidate_num,args.figer_type_num))
#      ent_prob_candidates =np.reshape(ent_linking_candprob[ids][tagid],(candidate_num,3))
#      candidate_ent_prob_feature = preprocessing.normalize(ent_prob_candidates,norm='l2',axis=0)
#      #candidate_ent_surfacewordv_feature.append(ent_mentWordV_candidates)
#      ent_mention_lstm_feature = np.sum(lstm_output[ids-ptr][ent_item[0]:ent_item[1]],axis=0)
#  
#      yield ent_mention_linking_tag,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
#         ent_mention_lstm_feature,candidate_ent_relcoherent_feature
def getLinkingFeature(args,lstm_output,ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ent_linking_candprob,ent_surfacewordv_feature,batch_size,ptr,flag='train'):
#def getLinkingFeature(ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ptr):
  ent_relcoherent_ngd = ent_relcoherent[0]
  ent_relcoherent_fb =  ent_relcoherent[1]
    
  candidateEntNum = args.candidate_ent_num
  ent_mention_linking_tag_list = []
  candidate_ent_linking_feature=[]
  candidate_ent_surfacewordv_feature=[]
  candidate_ent_type_feature=[]
  candidate_ent_prob_feature=[]
  candidate_ent_relcoherent_feature_ngd=[]
  candidate_ent_relcoherent_feature_fb=[]
  ent_mention_lstm_feature = []
  allLenght = len(ent_mention_index)
  sequence_length = []
  ent_index=[]
  lstm_index=[]
  #print 'lstm shape:',np.shape(lstm_output)
  dim3 = np.shape(lstm_output)[2]
  if 'trainmsnbc' == flag or flag == 'kbp_train' or flag == 'kbp_test':
    last_range = min(ptr+batch_size,allLenght)
  elif 'train' == flag:
    last_range = min(ptr+batch_size,allLenght)
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
        ent_mention_linking_tag = np.zeros((candidateEntNum,))
        if len(ent_mention_tag[ids][tagid])!=0:
          tag_index = ent_mention_tag[ids][tagid][0]
          ent_mention_linking_tag[tag_index] = 1
      else:  
        ent_mention_linking_tag = np.asarray(ent_mention_tag[ids][tagid],dtype=np.float32)
      candidate_num = len(ent_mention_link_feature[ids][tagid])/100 
      
      ent_mention_linking_tag_list.append(ent_mention_linking_tag)
      
      candidate_ent_relcoherent_item =  np.concatenate((np.asarray(np.reshape(ent_relcoherent_ngd[ids][tagid],(candidate_num,)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),),dtype=np.float32)))
      candidate_ent_relcoherent_feature_ngd.append(candidate_ent_relcoherent_item)
      
      
      items =np.zeros((candidate_num,))
      for icnum in range(candidate_num):
        if ent_relcoherent_fb[ids][tagid] !=0:
          items[icnum]=1
        else:
          items[icnum]=0
      candidate_ent_relcoherent_item1 =  np.concatenate((np.asarray(np.reshape(items,(candidate_num,)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),),dtype=np.float32)))
      candidate_ent_relcoherent_feature_fb.append(candidate_ent_relcoherent_item1)
      
      
      ent_linking_candidates =  np.concatenate((np.asarray(np.reshape(ent_mention_link_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),100),dtype=np.float32)))
      ent_mentWordV_candidates =  np.concatenate((np.asarray(np.reshape(ent_surfacewordv_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),100),dtype=np.float32)))
      
      ent_type_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_type[ids][tagid],(candidate_num,args.figer_type_num)),dtype=np.float32),np.zeros((max(0,candidateEntNum-candidate_num),args.figer_type_num),dtype=np.float32)))
      ent_prob_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_candprob[ids][tagid],(candidate_num,3)),dtype=np.float32),np.zeros((max(0,candidateEntNum-candidate_num),3),dtype=np.float32)))
                                      
      candidate_ent_linking_feature.append(ent_linking_candidates)
      candidate_ent_surfacewordv_feature.append(ent_mentWordV_candidates)
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
  candidate_ent_relcoherent_feature_fb = np.asarray(candidate_ent_relcoherent_feature_fb,dtype=np.float)
  candidate_ent_relcoherent_feature_ngd = np.asarray(candidate_ent_relcoherent_feature_ngd,dtype=np.float)
  candidate_ent_linking_feature = np.asarray(candidate_ent_linking_feature,dtype=np.float)
  #candidate_ent_surfacewordv_feature = np.asarray(candidate_ent_surfacewordv_feature,dtype=np.float)
  candidate_ent_type_feature = np.asarray(candidate_ent_type_feature,dtype=np.float)
  candidate_ent_prob_feature = np.asarray(candidate_ent_prob_feature,dtype=np.float)
  ent_mention_lstm_feature = np.asarray(ent_mention_lstm_feature,dtype=np.float)
  #print np.shape(ent_mention_lstm_feature)
  sequence_length = np.asarray(sequence_length)
  
  #print 'ent_mention_linking_tag_list:',np.shape(ent_mention_linking_tag_list)
  
  return ent_mention_linking_tag_list,candidate_ent_linking_feature,candidate_ent_type_feature,candidate_ent_prob_feature,\
         ent_mention_lstm_feature,candidate_ent_relcoherent_feature_ngd,candidate_ent_relcoherent_feature_fb,candidate_ent_surfacewordv_feature#,sequence_length

'''
def getLinkingFeature(args,lstm_output,ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ent_linking_candprob,ent_surfacewordv_feature,ptr,flag='train'):
#def getLinkingFeature(ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ptr):
  if len(ent_relcoherent)==2:
    
  candidateEntNum = args.candidate_ent_num
  ent_mention_linking_tag_list = []
  candidate_ent_linking_feature=[]
  candidate_ent_surfacewordv_feature=[]
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
        ent_mention_linking_tag = np.zeros((candidateEntNum,))
        if len(ent_mention_tag[ids][tagid])!=0:
          tag_index = ent_mention_tag[ids][tagid][0]
          ent_mention_linking_tag[tag_index] = 1
      else:  
        ent_mention_linking_tag = np.asarray(ent_mention_tag[ids][tagid],dtype=np.float32)
      candidate_num = len(ent_mention_link_feature[ids][tagid])/100 
      
      ent_mention_linking_tag_list.append(ent_mention_linking_tag)
      
      candidate_ent_relcoherent_item =  np.concatenate((np.asarray(np.reshape(ent_relcoherent[ids][tagid],(candidate_num,)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),),dtype=np.float32)))
      candidate_ent_relcoherent_feature.append(candidate_ent_relcoherent_item)
      
      ent_linking_candidates =  np.concatenate((np.asarray(np.reshape(ent_mention_link_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),100),dtype=np.float32)))
      ent_mentWordV_candidates =  np.concatenate((np.asarray(np.reshape(ent_surfacewordv_feature[ids][tagid],(candidate_num,100)),dtype=np.float32), np.zeros((max(0,candidateEntNum-candidate_num),100),dtype=np.float32)))
      
      ent_type_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_type[ids][tagid],(candidate_num,args.figer_type_num)),dtype=np.float32),np.zeros((max(0,candidateEntNum-candidate_num),args.figer_type_num),dtype=np.float32)))
      ent_prob_candidates = np.concatenate((np.asarray(np.reshape(ent_linking_candprob[ids][tagid],(candidate_num,3)),dtype=np.float32),np.zeros((max(0,candidateEntNum-candidate_num),3),dtype=np.float32)))
                                      
      candidate_ent_linking_feature.append(ent_linking_candidates)
      candidate_ent_surfacewordv_feature.append(ent_mentWordV_candidates)
      candidate_ent_type_feature.append(ent_type_candidates)
#      
#      @time:2017/3/21 we need to l2 normalize the vectors!
#      
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
         ent_mention_lstm_feature,candidate_ent_relcoherent_feature,candidate_ent_surfacewordv_feature#,sequence_length
'''         
 
  
