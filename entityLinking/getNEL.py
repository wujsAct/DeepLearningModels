# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/1/23
"""
import sys
sys.path.append('/home/wjs/demo/entityType/informationExtract')
sys.path.append('/home/wjs/demo/entityType/informationExtract/utils')
sys.path.append('/home/wjs/demo/entityType/informationExtract/embeddings')
import codecs
import collections
from tqdm import tqdm
import cPickle
import nelInputUtils as inputUtils
from entityRecog import nameEntityRecognition,pp,flags,args
import numpy as np
import argparse
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
parser.add_argument('--features', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
parser.add_argument('--candidateEntNum', type=int, help='candidateEntNum', required=True)
parser.add_argument('--ngdType', type=str, help='ngdType', required=True)
  
data_args = parser.parse_args()

data_tag = data_args.data_tag
dir_path = data_args.dir_path
features = data_args.features
ngdType = data_args.ngdType
candidateEntNum = data_args.candidateEntNum

corefEnts = cPickle.load(open(dir_path+'entMent2repMent.p'))
print data_tag,' coref:',len(corefEnts)

#for key in corefEnts:
#  print key, corefEnts[key]


ent_Mentions = cPickle.load(open(dir_path+'features/ent_mention_index.p'))
entMentsTags = cPickle.load(open(dir_path+data_tag+'_entMentsTags.p','rb'))
print len(entMentsTags)

'''read sents2aNosNo'''
sentid = 0
sentId2aNosNo = {}
with codecs.open(dir_path+'sentid2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    sentId2aNosNo[sentid] = line
    sentid += 1

testents = 0
allEnts = 0
passEnts = 0
for i in tqdm(range(len(ent_Mentions))):
  ents = ent_Mentions[i]
  aNosNo = sentId2aNosNo[i]
  if i < 0.95*len(ent_Mentions):
    testents += len(ents)
  #tagid=-1
  for j in range(len(ents)):
    allEnts += 1
    
    sindex = ents[j][0];eindex = ents[j][1]
    
    key = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)
    key1 = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)+'\t'+ents[j][2]
   
    linkTag = entMentsTags[key]   #get entity mention linking mid!
    
    if 'NIL' in linkTag:
      passEnts+=1
      continue
linkable_all = allEnts-passEnts
print 'total ents:',allEnts
print 'pass ents:',passEnts
print 'need to predict entities:',allEnts-passEnts  
print 'testEnts:', testents
print '----------------------'
#exit(0)

#'''read mid2name'''
#fname = 'data/mid2name.tsv'
#wikititle2fb = collections.defaultdict(list)
#fb2wikititle=collections.defaultdict(list)
#with codecs.open(fname,'r','utf-8') as file:
#  for line in tqdm(file):
#    line = line.strip()
#    items = line.split('\t')
#    if len(items)==2:
#      fbId = items[0]; title = items[1]  
#      fb2wikititle[fbId].append(title)
#      wikititle2fb[title].append(fbId)
#fb2wikititle['NIL'] = 'NIL'

'''read ER result entmentions'''
#if data_tag == 'kbp':
#  testUtils = inputUtils(args.rawword_dim,data_tag+'_evaluation')
#else:
#  testUtils = inputUtils(args.rawword_dim,data_tag)
feature_path =dir_path + 'features/'+str(candidateEntNum)+'/'
test_entliking= cPickle.load(open(feature_path+data_tag+'_ent_linking.p'+str(candidateEntNum),'rb'))
ent_mention_index_real = test_entliking['ent_mention_index'];
ent_mention_tag = test_entliking['ent_mention_tag']; 

NERrets = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+'entityLinkingResult.p'+features+"_"+ngdType))  #[300,30]
#NERrets = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+features+'_aida_'+data_tag+'_entityLinkingResult.p'))  #[300,30]


NERentCands = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_ent_cand_mid_new.p'+str(candidateEntNum)))  #[300,candsNums]
rightPred = 0
recall = 0
wrongEnts =0
allLenght = len(ent_mention_index_real)

lesisNum = 0
print 'NERentCands:',len(NERentCands)
print 'allLenght:',allLenght
print 'NERets shape:',np.shape(NERrets)
ent_mention_index_right ={}
NILEntKeys = {}
entids = -1

menKey2predEnt={}
menKey2target={}

entid = -1
k  = -1
'''
we must add a threshold 
'''
TN=0.0;FP=0.0
FN=0.0;TP=0.0
passEnts = 0
linkableEntisWrong = {}
ent_mention_index={};ent_mention_index_right ={};ent_mention_index_wrong_nil={};ent_mention_index_wrong_oent={}
NILEntKeys={};NILEnt_right_Keys = {};NILEnt_wrong_Keys={}
gold = []; predict = []
linkable_right=0.0; candi_include_gold=0.0
linkable_all=0
for i in range(len(ent_Mentions)):
  #if i > 0.95*len(ent_Mentions):
  #  break
  aNosNo = sentId2aNosNo[i]
  
  ents = ent_Mentions[i]
  
  tagid=-1
  for j in range(len(ents)):
    
    allEnts += 1
    
    sindex = ents[j][0];eindex = ents[j][1]
    
    key = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)
   
    linkTag = entMentsTags[key]   #get entity mention linking mid!
    menKey2target[key] = linkTag
    
    tagid +=1
    entids += 1
    candMids = NERentCands[entids].keys()
    
    if linkTag=='NIL':
      continue
      if linkTag=='NIL':
        passEnts+=1
        NILEntKeys[key]='NIL'
        
        ret = NERrets[k]
        
        right_id = np.argmax(ret,0)
        
        print 'NIL', ents[j], 'predict to:',candMids[right_id],ret[right_id]
        if right_id >= len(candMids) or ret[right_id] < 0.025:
          #print ents[j],' is wrong NIL!'
          TN += 1
          NILEnt_right_Keys[key]='NIL'
        else:
          FN+= 1
          NILEnt_wrong_Keys[key] = candMids[right_id]
    else:
      linkTag = linkTag[0]

   
    k+=1
    linkable_all += 1
    ent_mention_index[key]= linkTag
    if linkTag in candMids:
      candi_include_gold+= 1
    predMid='NIL'
    if len(candMids)!=0:
      ret = NERrets[k]
      #if len(candMids) !=0:  
      ret = ret[0:len(candMids)]
      
      right_id = np.argmax(ret,0)
      predict.append(right_id)
      
      
      if right_id < len(candMids):# and ret[right_id]>=0.025:
        predMid = candMids[right_id]
    
    else:
      passEnts += 1
        
    if predMid == linkTag: #or entMentsTags[key]=='NIL':
      for candi in range(len(candMids)):
        if candMids[candi] == linkTag:
          gold.append(candi)
          break
      TP += 1
      linkable_right += 1
      rightPred += 1
      ent_mention_index_right[key] = predMid
      #print 'right ent score:',ret[right_id]
    else:
      gold.append(len(candMids))
      linkableEntisWrong[key] =[linkTag,predMid,candMids,ret]
      wrongEnts += 1
      ent_mention_index_wrong_oent[key] =1
      FP += 1 
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print '--------------------------------------'
print 'passEnts:',passEnts,' candi_include_gold:',candi_include_gold
print 'accuracy:',linkable_right,linkable_all,linkable_right/linkable_all




not_in_training_ents =0
for key in corefEnts:
  print key,corefEnts[key]
  items = key.split('\t')
  new_key =  '\t'.join(items[0:3])
  
  
  coref_linkingTag = entMentsTags[new_key]
  if coref_linkingTag!='NIL':
    coref_linkingTag = coref_linkingTag[0]
    linkable_all += 1
  
    ent_name = items[3]
    
    #if key not in NILEntKeys and key not in ent_mention_index_right:
    val = corefEnts[key]
    coref_items = val.split('\t')
    coref_key = '\t'.join(coref_items[0:3])
    coref_name = coref_items[3]
    #we need to consider the coref
    if ent_name != coref_name:
      if new_key in ent_mention_index_wrong_nil or new_key in ent_mention_index_wrong_oent:
        if coref_key in ent_mention_index:
          if ent_mention_index[new_key] == ent_mention_index[coref_key]:
            print 'wrong 2 right:',ent_mention_index[new_key], ent_mention_index[coref_key]
            linkable_right += 1
      elif new_key in ent_mention_index_right:
          
          if coref_key in ent_mention_index:
            if ent_mention_index[new_key] != ent_mention_index[coref_key]:
              print 'right 2 wrong:',ent_mention_index[new_key], ent_mention_index[coref_key]
              linkable_right -=1
            if coref_key in NILEntKeys:
              print 'right 2 wrong:',ent_mention_index[new_key], ent_mention_index[coref_key]
              linkable_right -= 1
      else:
        if coref_key not in ent_mention_index:
          continue
        else:
          if coref_linkingTag in ent_mention_index[coref_key]:
            linkable_right += 1
            
print 'accuracy:',linkable_right,linkable_all,linkable_right/linkable_all        