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


'''read mid2name'''
fname = 'data/mid2name.tsv'
wikititle2fb = collections.defaultdict(list)
fb2wikititle={}
with codecs.open(fname,'r','utf-8') as file:
  for line in tqdm(file):
    line = line.strip()
    items = line.split('\t')
    if len(items)==2:
      fbId = items[0]; title = items[1]  
      fb2wikititle[fbId] = title
      wikititle2fb[title].append(fbId)
fb2wikititle['NIL'] = 'NIL'

dir_path = 'data/ace/'    
'''read entmention 2 aNosNoid'''
entsFile = dir_path+'entMen2aNosNoid.txt'
hasMid = 0
entMentsTags={}
entMents2surfaceName={}
with codecs.open(entsFile,'r','utf-8') as file:
  for line in file:
    line = line.strip()
    items = line.split('\t')
    entMent = items[0]; linkingEnt = items[1]; aNosNo = items[2]; start = items[3]; end = items[4]
    key = aNosNo + '\t' + start+'\t'+end
    if linkingEnt == 'NIL':
      hasMid += 1
      entMentsTags[key]='NIL'

    if linkingEnt in wikititle2fb:
      print wikititle2fb[linkingEnt]
      hasMid +=1 
      entMentsTags[key] =wikititle2fb[linkingEnt]
      entMents2surfaceName[key] = entMent
print 'entMentsTags nums:',len(entMentsTags)

'''read sents2aNosNo'''
sentid = 0
sentId2aNosNo = {}
with codecs.open(dir_path+'sentid2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    sentId2aNosNo[sentid] = line
    sentid += 1

'''read ER result entmentions'''
testUtils = inputUtils(args.rawword_dim,"ace")
test_entliking= testUtils.ent_linking;
ent_mention_index = test_entliking['ent_mention_index'];
ent_mention_tag = test_entliking['ent_mention_tag']; 
NERrets = cPickle.load(open(dir_path+'entityLinkingResult.p'))  #[300,30]
NERentCands = cPickle.load(open(dir_path+'features/ace_ent_cand_mid.p'))  #[300,candsNums]
rightPred = 0
recall = 0
allLenght = len(ent_mention_index)
entids = 0
notinentmenttags = 0
for ids in range(allLenght):
  aNosNo = sentId2aNosNo[ids]
  tagid = 0
  for ent_item in ent_mention_index[ids]:
    if len(ent_mention_tag[ids])!=0:
      predMid = 'NIL';
      ret = NERrets[entids]      
      right_id = np.argmax(ret,0)
      
      candMids = NERentCands[entids].keys()
      
      if right_id < len(candMids):
        predMid = candMids[right_id]
      key = aNosNo + '\t' + str(ent_item[0])+'\t'+str(ent_item[1])
      print key
      flag=False
      if key in entMentsTags:
        for midi in candMids:
          if midi in entMentsTags[key] or entMentsTags[key]=='NIL':
            flag = True
        if flag:
          recall += 1
        if predMid in entMentsTags[key] or entMentsTags[key]=='NIL':
          rightPred += 1
        else:
          print ret
          candmidlist = []
          for item in candMids:
            if item in fb2wikititle:
              candmidlist.append(fb2wikititle[item])
            else:
              candmidlist.append(item)
          print candmidlist
          print entMents2surfaceName[key]
          print 'right tag:',entMentsTags[key]
          print fb2wikititle[entMentsTags[key][0]]
          if predMid in fb2wikititle:
            print 'wrong tag:',predMid,fb2wikititle[predMid]
          else:
            print predMid
      else:
        notinentmenttags += 1
      entids += 1
print 'rightPred:',rightPred
print 'recall:',recall
print 'notinentmenttags:',notinentmenttags