# -*- coding: utf-8 -*-

import cPickle as pkl
import numpy as np
class nelInputUtils(object):
  def __init__(self,dims,flag='train'):
    dir_path ='/home/wjs/demo/entityType/informationExtract/data/aida/features'
    if flag=='train':
      self.emb = pkl.load(open(dir_path+'/train_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/train_tag.p'+dims,'rb'))
      self.ent_linking  = pkl.load(open(dir_path+'/train_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open(dir_path+'/train_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open(dir_path+'/train_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open(dir_path+'/train_ent_linking_candprob.p','rb'))
    elif flag=='testa':
      self.emb = pkl.load(open(dir_path+'/test_a_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/test_a_tag.p'+dims,'rb'))
      self.ent_linking  = pkl.load(open(dir_path+'/testa_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open(dir_path+'/testa_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open(dir_path+'/testa_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open(dir_path+'/testa_ent_linking_candprob.p','rb'))
    else:
      self.emb = pkl.load(open(dir_path+'/test_b_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/test_b_tag.p'+dims,'rb'))
      self.ent_linking  = pkl.load(open(dir_path+'/testb_ent_linking.p','rb'))
      self.ent_linking_type  = pkl.load(open(dir_path+'/testb_ent_linking_type.p','rb'))
      self.ent_relcoherent  = pkl.load(open(dir_path+'/testb_ent_relcoherent.p','rb'))
      self.ent_linking_candprob = pkl.load(open(dir_path+'/testb_ent_linking_candprob.p','rb'))


def getLinkingFeature(args,lstm_output,ent_mention_index,ent_mention_tag,ent_relcoherent,ent_mention_link_feature,ent_linking_type,ent_linking_candprob,ptr,flag='train'):
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