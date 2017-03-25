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

parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
data_args = parser.parse_args()

data_tag = data_args.data_tag
dir_path = data_args.dir_path

corefEnts = cPickle.load(open(dir_path+'entMent2repMent.p'))

ent_Mentions = cPickle.load(open(dir_path+'features/ent_mention_index.p'))

entMentsTags = cPickle.load(open(dir_path+data_tag+'_entMentsTags.p','rb'))
#print entMentsTags
    

#'''read mid2name'''
#fname = 'data/mid2name.tsv'
#wikititle2fb = collections.defaultdict(list)
#fb2wikititle={}
#with codecs.open(fname,'r','utf-8') as file:
#  for line in tqdm(file):
#    line = line.strip()
#    items = line.split('\t')
#    if len(items)==2:
#      fbId = items[0]; title = items[1]  
#      fb2wikititle[fbId] = title
#      wikititle2fb[title].append(fbId)
#fb2wikititle['NIL'] = 'NIL'


'''read sents2aNosNo'''
sentid = 0
sentId2aNosNo = {}
with codecs.open(dir_path+'sentid2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    sentId2aNosNo[sentid] = line
    sentid += 1

'''read ER result entmentions'''

testUtils = inputUtils(args.rawword_dim,data_tag)
test_entliking= testUtils.ent_linking;
ent_mention_index = test_entliking['ent_mention_index'];
ent_mention_tag = test_entliking['ent_mention_tag']; 
#NERrets = cPickle.load(open(dir_path+data_tag+'_entityLinkingResult.p'))  #[300,30]
NERrets = cPickle.load(open(dir_path+'entityLinkingResult.p'))  #[300,30]
NERentCands = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_cand_mid.p'))  #[300,candsNums]
rightPred = 0
recall = 0
wrongEnts =0
allLenght = len(ent_mention_index)
entids = -1
lesisNum = 0
print 'NERentCands:',len(NERentCands)
print 'allLenght:',allLenght
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

for i in tqdm(range(len(ent_Mentions))):
  aNosNo = sentId2aNosNo[i]
  
  ents = ent_Mentions[i]
  
  tagid=-1
  for j in range(len(ents)):
    allEnts += 1
    entids +=1
    sindex = ents[j][0];eindex = ents[j][1]
    
    key = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)
    
    
    linkTag = entMentsTags[key]   #get entity mention linking mid!
    k+=1
    cand_mid_dict = NERentCands[k]
   
    if linkTag=='NIL':
      passEnts+=1
      NILEntKeys[key] = 1
      continue
    
    tagid +=1
#    if np.sum(ent_mention_tag[i][tagid]) == 0:
#      notrecall += 1
#      continue
    relentid += 1
    '''
    @2017/3/21 we need to return the top 1 candiate 
    '''
    candMids = NERentCands[entids].keys()
    ret = NERrets[relentid]
    if len(candMids) !=0:  
      ret = ret[0:len(candMids)]
      
    right_id = np.argmax(ret,0)
    
   
    if right_id < len(candMids):
      predMid = candMids[right_id]
 
    if predMid in linkTag: #or entMentsTags[key]=='NIL':
      rightPred += 1
      ent_mention_index_right[key] = 1  
    else:
      wrongEnts += 1
#      print predMid
#      print linkTag
#      print candMids
#      print '---------------------'
    
print 'rightPred:',rightPred
print 'pass ents:',passEnts
print 'wrong ents:',wrongEnts
print 'total:',wrongEnts+notrecall+rightPred
print 'precision:',rightPred*1.0/(wrongEnts+rightPred)
print 'allEnts:',allEnts
#cPickle.dump(ent_mention_index_right,open(dir_path+'rightPred.p','wb'))

total = wrongEnts+rightPred


if data_tag=='msnbc':
  for key in corefEnts:
    if key not in ent_mention_index and key not in NILEntKeys:
      val = corefEnts[key]
      items = val.split('\t')
      key_coref = '\t'.join(items[0:3])
      if entMentsTags[key_coref]!='NIL':
        if key_coref in ent_mention_index_right:
          rightPred += 1
        else:
          wrongEnts += 1
          pass
      else:
        passEnts+=1
    else:
      passEnts += 1
else:
  for key in corefEnts:
    if key not in ent_mention_index and key not in NILEntKeys:
      val = corefEnts[key]
      items = val.split('\t')
      key_coref = '\t'.join(items[0:3])
      if entMentsTags[key_coref]!='NIL':
        if key_coref in ent_mention_index_right:
          rightPred += 1
          wrongEnts -= 1
      else:
        passEnts+=1
      #print key,val
print 'rightPred:',rightPred
print 'pass ents:',passEnts
print 'wrong ents:',wrongEnts
print 'total:',wrongEnts+rightPred
print 'precision:',rightPred*1.0/(wrongEnts+rightPred)