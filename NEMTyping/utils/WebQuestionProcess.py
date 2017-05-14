# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:45:05 2017

@author: DELL
@function: process web quesion
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/evals/")
import numpy as np
import cPickle
from tqdm import tqdm
from evalChunk import openSentid2aNosNoid,getaNosNo2entWebQuestion

'''
@generate the ents
'''  
def getEnts(dir_path,tag):
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,tag)
  
  print len(sid2aNosNo)
  aNosNo2entMen = getaNosNo2entWebQuestion(dir_path,tag)
  #print aNosNo2entMen
  
 # print aNosNo2entMen
  
  entMents =[]
  for sid in tqdm(range(len(sid2aNosNo))):
    aNosNo = sid2aNosNo[sid]
    
    entList = aNosNo2entMen[aNosNo]
    print entList
    entMents.append(entList)
    
  cPickle.dump(entMents,open(dir_path+'features/'+tag+'_entMents.p','wb'))
  
  

dir_path = 'data/WebQuestion/'
tag = 'train'
getEnts(dir_path,tag)
