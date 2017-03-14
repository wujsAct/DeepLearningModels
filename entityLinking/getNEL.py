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


entMentsTags = cPickle.load(open(dir_path+data_tag+'_entMentsTags.p','rb'))
#print entMentsTags
    

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
entids = -1
notinentmenttags = 0
allEnts = 0
testRange = int(allLenght *0.8)
lesisNum = 0
print allLenght
print np.shape(NERrets)
#exit(0)
ent_mention_index_right ={}
relentid = -1
for ids in range(allLenght):
  
  aNosNo = sentId2aNosNo[ids]
  tagid = -1
  for ent_item in ent_mention_index[ids]:
    tagid+=1
    if np.sum(ent_mention_tag[ids][tagid]) == 0:
      #print tagid
      entids +=1
      continue

    #if ids<testRange:
      #entids += 1
    #  continue
    relentid += 1
    entids +=1
    allEnts += 1
    
    predMid = 'NIL'
    ret = NERrets[relentid]      
    right_id = np.argmax(ret,0)
    
    candMids = NERentCands[entids].keys()
    
    if right_id < len(candMids):
      predMid = candMids[right_id]
      
    key = aNosNo + '\t' + str(ent_item[0])+'\t'+str(ent_item[1])
    
    flag=False
    linkTag = entMentsTags[key]
    if '/m/0c_md_' in linkTag:
      lesisNum += 1
    if entMentsTags[key] =='NIL':
      notinentmenttags += 1
    midi_index = -1
    right_index = 0
    for midi in candMids:
      midi_index += 1
      if midi in linkTag: #or entMentsTags[key]=='NIL':
        flag = True
        right_index = midi_index
  
    if flag:
      recall += 1
    if predMid in linkTag: #or entMentsTags[key]=='NIL':
      rightPred += 1
      ent_mention_index_right[key] = 1
    else:
      candmidlist = []
      if entMentsTags[key]!='NIL':
        for item in candMids:
          if item in fb2wikititle:
            candmidlist.append(fb2wikititle[item])
          else:
            candmidlist.append(item)
        print  'wrong mentions',entMentsTags[key],'right_index:',right_index
        for key in linkTag:
          print 'title:',fb2wikititle[key] 
        print 'predict rets:', predMid,'index:',right_id
        if predMid in fb2wikititle:
          print 'predict entity mention:',fb2wikititle[predMid] 
        print candmidlist
        print ret
        print '--------------'

#cPickle.dump(ent_mention_index_right,open(dir_path+'rightPred.p','wb'))


print 'all ents:',allEnts,entids
print 'rightPred:',rightPred
print 'recall:',recall
print 'notinentmenttags:',notinentmenttags
print 'lesisNum:',lesisNum



for key in corefEnts:
  val = corefEnts[key]
  items = val.split('\t')
  key_coref = '\t'.join(items[0:3])
  if entMentsTags[key_coref]!='NIL':
    allEnts += 1
  #print key_coref
  if key_coref in ent_mention_index_right:
    rightPred += 1
  else:
    print key,val

print 'all ents:',allEnts,entids
print 'rightPred:',rightPred
print 'recall:',recall
print 'notinentmenttags:',notinentmenttags
print 'lesisNum:',lesisNum