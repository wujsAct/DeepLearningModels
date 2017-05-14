# -*- coding: utf-8 -*-
"""
Created on Mon May 08 09:15:08 2017

@author: DELL
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/evals/")
import numpy as np
import cPickle
from tqdm import tqdm
from evalChunk import openSentid2aNosNoid,getaNosNo2entMenOntos

'''
@generate the type dictionary!
'''
def typeFiger():
  figer2id = {}
  typeId = 0
  fname = 'data/OntoNotes/features/train_gold_entMen2aNosNoid.txt'
  print fname
  for line in open(fname):
    line = line.strip()
    items = line.split('\t')
    
    for i in range(4,len(items)):
       fbType = items[i]
       if fbType not in figer2id:
         figer2id[fbType] = typeId
         typeId += 1
         print typeId
         
  print typeId
  cPickle.dump(figer2id,open('data/OntoNotes/type2id.p','wb'))

'''
@generate the ents
'''  
def getEnts(dir_path,tag):
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,tag)
  #print len(sid2aNosNo)
  aNosNo2entMen = getaNosNo2entMenOntos(dir_path,tag)
  
 # print aNosNo2entMen
  
  entMents =[]
  for sid in tqdm(range(len(sid2aNosNo))):
    aNosNo = sid2aNosNo[sid]
    
    entList = aNosNo2entMen[aNosNo]
    entMents.append(entList)
  cPickle.dump(entMents,open(dir_path+'features/'+tag+'_entMents.p','wb'))

'''
@generate validation tests!
'''
dir_path = 'data/OntoNotes/'
tag = 'train'
#typeFiger()
getEnts(dir_path,tag)