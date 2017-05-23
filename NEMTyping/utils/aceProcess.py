# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:45:05 2017

@author: DELL
@function: process web quesion
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/evals/")
import cPickle
from tqdm import tqdm
from evalChunk import getaNosNo2entACE

'''
@generate the ents
'''  
def getEnts(dir_path,tag):
  sid2aNosNo = {}
  aNosNo2sid = {}
  
  sNo = 0
  
  for line in open(dir_path+'features/'+tag+'_sentid2aNosNoid.txt'):
    line = line.strip()
    sid2aNosNo[sNo] = line
    aNosNo2sid[line] = sNo
    sNo += 1
  entNums = 0
  aNosNo2entMen = getaNosNo2entACE(dir_path,tag)
  #print aNosNo2entMen
  
 # print aNosNo2entMen
  entMents =[]
  for sid in range(len(sid2aNosNo)):
    aNosNo = sid2aNosNo[sid]
    
    entList = aNosNo2entMen[aNosNo]
    entNums += len(entList)
    entMents.append(entList)
  print entNums
  #cPickle.dump(entMents,open(dir_path+'features/'+tag+'_entMents.p','wb'))
  
dir_path = 'data/ace/'
tag = 'ace'
getEnts(dir_path,tag)