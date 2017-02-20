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

entMentsTags = cPickle.load(open(dir_path+data_tag+'_entMentsTags.p','rb'))

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
NERrets = cPickle.load(open(dir_path+'entityLinkingResult.p'))  #[300,30]
NERentCands = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_cand_mid.p'))  #[300,candsNums]
rightPred = 0
recall = 0
allLenght = len(ent_mention_index)
entids = 0
notinentmenttags = 0
allEnts = 0
for ids in range(allLenght):
  aNosNo = sentId2aNosNo[ids]
  tagid = 0
  for ent_item in ent_mention_index[ids]:
    allEnts += 1
    
    predMid = 'NIL'
    ret = NERrets[entids]      
    right_id = np.argmax(ret,0)
    
    candMids = NERentCands[entids].keys()
    
    if right_id < len(candMids):
      predMid = candMids[right_id]
    key = aNosNo + '\t' + str(ent_item[0])+'\t'+str(ent_item[1])

    flag=False
    linkTag = entMentsTags[key]
    if entMentsTags[key] =='NIL':
      notinentmenttags += 1
    for midi in candMids:
      if midi in linkTag: #or entMentsTags[key]=='NIL':
        flag = True
    if flag:
      recall += 1
    if predMid in linkTag: #or entMentsTags[key]=='NIL':
      rightPred += 1
  
    entids += 1

#          candmidlist = []
#          for item in candMids:
#            if item in fb2wikititle:
#              candmidlist.append(fb2wikititle[item])
#            else:
#              candmidlist.append(item)
#          print candmidlist
#          print entMents2surfaceName[key]
          #print 'right tag:',entMentsTags[key]
#          if predMid in fb2wikititle:
#            print 'wrong tag:',predMid,fb2wikititle[predMid]
#          else:
#            print predMid
print 'all ents:',allEnts,entids
print 'rightPred:',rightPred
print 'recall:',recall
print 'notinentmenttags:',notinentmenttags