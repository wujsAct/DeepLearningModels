# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/1/23
"""
import codecs
import collections
from tqdm import tqdm
import cPickle
from utils import nelInputUtils as inputUtils
from entityRecog import nameEntityRecognition,pp,flags,args
import numpy as np
import argparse
import time

def printfeatuesThreshold(ent_id):
  all_candidate_mids = []
  allEnts=0
  passEnts=0
  rightEnts=0
  wrongEnts=0
  
  entids=0
  relentid = -1
  k  = -1
  rightWiki = 0
  rightFb=0
  rightWord2vec = 0
  disambiguation = []
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    tagid=-1
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      #key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      
      linkTag = entMentsTags[ent_id][1]   #get entity mention linking mid!
      #print linkTag
      enti_name = ents[j].content.lower()
      
      if linkTag=='NIL':
        passEnts+=1
        ent_id +=1
        continue
      
      k+=1
      print 'k:',k
      ent_id += 1
    
      tagid +=1
     
      relentid += 1
      rel_cand_mid_dict={}
      oreder_cand_mid_dict={}
      candMids = NERentCands[k].keys()
      wikicands = wiki_candidate_mids[k]
      FBcands = freebase_candidate_mids[k]
    
  
#      if 'surrey' in enti_name:
#        print 'candMids:',len(candMids)
#        
      '''
      @2017/3/23 need to reserve the top 10 wikipedia, revise tomorrow!
      '''
      for key in NERentCands[k]:
        if NERentCands[k][key][2]==0 and NERentCands[k][key][1]==0 and key not in wikicands:
          candMids.remove(key)
#          print 'candMids:',len(candMids)
#          print linkTag,NERentCands[k][linkTag]
      #print len(candMids)
      for key in candMids:
        if key in wikicands:
          rel_cand_mid_dict[key] = NERentCands[k][key]
        else:
          oreder_cand_mid_dict[key] = NERentCands[k][key]
          
      sorted_cand_mid_dict= sorted(oreder_cand_mid_dict.items(), key=lambda oreder_cand_mid_dict:oreder_cand_mid_dict[1][2])[0:max(0,40-len(rel_cand_mid_dict))]
            
      sorted_cand_mid_dict = dict(sorted_cand_mid_dict)
      
      dictMerged2=dict(rel_cand_mid_dict, **sorted_cand_mid_dict)
      
      disambiguation.append(len(dictMerged2))
      
      all_candidate_mids.append(dict(dictMerged2))  
      if linkTag in dictMerged2:
        rightEnts+=1
        if linkTag in wikicands:
          rightWiki += 1
        if linkTag in FBcands:
          rightFb+= 1
#        if linkTag not in FBcands and linkTag not in wikicands:
#          rightWord2vec += 1
        if NERentCands[k][linkTag][1] !=0:
          rightWord2vec += 1
      else:
        wrongEnts += 1
  
  cPickle.dump(all_candidate_mids,open(dir_path+'features/'+data_tag+'_ent_cand_mid_new.p','wb'))
      #print len(set(candMids))
  #只剩下85.7的正确率了呢！ 
  #明天一般写论文，还需要加上很多信息呢！ 
  print '---------------------------'
  print 'disambiguation:',len(disambiguation)
  print 'all_candidate_mids:',len(all_candidate_mids)
  print 'datatag',data_tag
  print 'right:',rightEnts
  print 'wrong:',wrongEnts
  print 'pass:',passEnts
  print 'all ents:',allEnts
  print 'rightWiki:',rightWiki, rightWiki*1.0/(rightEnts+wrongEnts)
  print 'rightFB:',rightFb, rightFb*1.0/(rightEnts+wrongEnts)
  print 'rightWord2vec:',rightWord2vec, rightWord2vec*1.0/(rightEnts+wrongEnts)
  print rightEnts*1.0/(rightEnts+wrongEnts)
  
  print 'average disambiguation:',np.average(disambiguation)
  print '---------------------------------------'
        

def printfeatues(ent_id):
  allEnts=0
  passEnts=0
  rightEnts=0
  wrongEnts=0
  
  entids=0
  relentid = -1
  k  = -1
  rightWiki = 0
  rightFb=0
  rightWord2vec = 0
  disambiguation = []
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      #key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      
      linkTag = entMentsTags[ent_id][1]   #get entity mention linking mid!
      #print linkTag
      enti_name = ents[j].content.lower()
      
      if linkTag=='NIL':
        passEnts+=1
        ent_id +=1
        continue      
      k+=1
      '''
      @revise: 很神奇，应该是18578的呢，为啥没有达到呢？
      '''
      print 'k:',k
      ent_id += 1

     
      relentid += 1
      
      candMids = list(NERentCands[k].keys())
      wikicands = wiki_candidate_mids[k]
      FBcands = freebase_candidate_mids[k]
    
  
#      if 'surrey' in enti_name:
#        print 'candMids:',len(candMids)
#        
      '''
      @2017/3/23 need to reserve the top 10 wikipedia, revise tomorrow!
      '''
      for key in NERentCands[k]:
        if NERentCands[k][key][2]==0 and NERentCands[k][key][1]==0 and key not in wikicands:
          candMids.remove(key)
#          print 'candMids:',len(candMids)
#          print linkTag,NERentCands[k][linkTag]
      #print len(candMids)
      disambiguation.append(len(candMids))
      
        
      if linkTag in candMids:
        rightEnts+=1
        if linkTag in wikicands:
          rightWiki += 1
        if linkTag in FBcands:
          rightFb+= 1
#        if linkTag not in FBcands and linkTag not in wikicands:
#          rightWord2vec += 1
        if NERentCands[k][linkTag][1] !=0:
          rightWord2vec += 1
      else:
        wrongEnts += 1
      #print len(set(candMids))
  #只剩下85.7的正确率了呢！ 
  #明天一般写论文，还需要加上很多信息呢！ 
  print '---------------------------' 
  print 'datatag',data_tag
  print 'right:',rightEnts
  print 'wrong:',wrongEnts
  print 'pass:',passEnts
  print 'all ents:',allEnts
  print 'rightWiki:',rightWiki, rightWiki*1.0/(rightEnts+wrongEnts)
  print 'rightFB:',rightFb, rightFb*1.0/(rightEnts+wrongEnts)
  print 'rightWord2vec:',rightWord2vec, rightWord2vec*1.0/(rightEnts+wrongEnts)
  print rightEnts*1.0/(rightEnts+wrongEnts)
  
  print 'average disambiguation:',np.average(disambiguation)
  print '---------------------------------------'
        
      
'''
def getAccuracy(ent_id):
  rightPred = 0
  recall = 0
  wrongEnts =0
  entids = -1
  lesisNum = 0
  
  print 'NERentCands:',len(NERentCands)
  print 'NERets shape:',np.shape(NERrets)
  ent_mention_index_right ={}
  NILEntKeys = {}
  relentid = -1
  passEnts = 0
  wrongEnts = 0
  notrecall = 0
  rightEnts =0
  RecallEnts = 0
  allEnts = 0
  k  = -1
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    tagid=-1
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      
      linkTag = entMentsTags[ent_id][1]   #get entity mention linking mid!
      #print linkTag
      enti_name = ents[j].content.lower()
      
      if linkTag=='NIL':
        passEnts+=1
        ent_id +=1
        NILEntKeys[key] =1
        continue
      k+=1
      ent_id += 1
    
      tagid +=1
     
      relentid += 1
      
      candMids = NERentCands[k].keys()
      
        
      ret = NERrets[relentid]
      if len(candMids) !=0:  
        ret = ret[0:len(candMids)]
      right_id = np.argmax(ret,0)
      
      if right_id < len(candMids):
        predMid = candMids[right_id]
      
      if 'surrey' in enti_name:
        print enti_name
        print predMid,right_id,candMids[right_id],NERentCands[k][candMids[right_id]]
        print linkTag,NERentCands[k][linkTag]
        print candMids
        print np.concatenate((np.asarray(np.reshape(linkcandsProb[i][tagid],(len(candMids),3)),dtype=np.float32),np.zeros((max(0,30-len(candMids)),3),dtype=np.float32)))
        
        print '------------------'
        
      if predMid in linkTag: #or entMentsTags[key]=='NIL':
        rightPred += 1
        ent_mention_index_right[key] = 1  
      else:
        wrongEnts += 1
  #      print predMid,right_id
  #      print linkTag
  #      print candMids
  #      print '--------------------'
        
      
  print 'rightPred:',rightPred
  print 'pass ents:',passEnts
  print 'wrong ents:',wrongEnts
  print 'total:',wrongEnts+rightPred
  print 'precision:',rightPred*1.0/(wrongEnts+rightPred)
  print 'Nil ents:',len(NILEntKeys)
  print 'all ents:',allEnts
  print 'not recall:', notrecall
  print 'rel ent id:',relentid
  
  #cPickle.dump(ent_mention_index_right,open(dir_path+'rightPred.p','wb'))
  
  total = wrongEnts+rightPred
  
  print 'len corefEnts:',len(corefEnts)
  for key in corefEnts:
    if key not in NILEntKeys and key not in ent_mention_index_right:
      val = corefEnts[key]
      items = val.split('\t')
      key_coref = '\t'.join(items[0:3])
      
      if key_coref in ent_mention_index_right:
        rightPred += 1
        wrongEnts-=1
      else:
        pass
        #print key,val
  print 'right:',rightPred
  print 'wrong ents:',wrongEnts
  print 'total:',total
  print 'pass ents:',passEnts
  print 'precison:',rightPred*1.0/(total)
'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
data_args = parser.parse_args()
data_tag = data_args.data_tag
dir_path = data_args.dir_path

corefEnts = cPickle.load(open(dir_path+'process/'+data_tag+'_entMent2repMent.p'))


entMentsTags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
ent_id=0
if data_tag=='train':
  ent_id = 0
if data_tag=='testa':
  ent_id = 23396
if data_tag=='testb':
  ent_id = 29313

print 'start to load datasets....'
stime = time.time()
if data_tag=='train':
  f_input_ent_ments = dir_path + '/features/' + data_tag+'_entms.p100_new'
else:
  f_input_ent_ments = dir_path + '/features/' + data_tag+'_entms.p100'
dataEnts = cPickle.load(open(f_input_ent_ments,'rb'));
ent_Mentions = dataEnts['ent_Mentions']
data = cPickle.load(open(dir_path+'process/'+ data_tag+'.p','r'))
aNosNo2id = data['aNosNo2id']
id2aNosNo = {val:key for key,val in aNosNo2id.items()}

#linkcandsProb = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_linking_candprob.p','rb'))
#ent_linking  = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_linking.p','rb'))
#print 'load dataset cost time:',time.time()-time.time()
#print 'ent_linking:',len(ent_linking['ent_mention_tag'])
#
#NERrets = cPickle.load(open(dir_path+'entityLinkingResult.p'))
#print dir_path+data_tag+'_entityLinkingResult.p'
#print np.shape(NERrets)

#data_param={'all_candidate_mids':all_candidate_mids,'wiki_candidate_mids':wiki_candidate_mids,'freebase_candidate_mids':freebase_candidate_mids}

data_param = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_cand_mid.p'))  #[300,candsNums]
NERentCands = data_param['all_candidate_mids']
print 'NERentCands:',len(NERentCands)

wiki_candidate_mids = data_param['wiki_candidate_mids']
freebase_candidate_mids = data_param['freebase_candidate_mids']


#getAccuracy(ent_id)
#printfeatues(ent_id)
printfeatuesThreshold(ent_id)