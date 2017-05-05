# -*- coding: utf-8 -*-
"""
Created on Thu May 04 10:10:13 2017

@author: DELL
"""

import cPickle
import numpy as np

figer2id = cPickle.load(open('data/figer/figer2id.p','rb'))
id2figer = {figer2id[key]:key for key in figer2id}


#get the first level
row = 0
first2Level = {}
secondLevel = []
for figer in figer2id:
  if len(figer.split('/')) ==2:
    first2Level[figer] = [row]
    row += 2
  else:
    secondLevel.append(figer)

second2Level={}        
for figer in secondLevel:
  hasHier=False
  for key in first2Level:
    if key in figer:
      hasHier=True
      rowId = first2Level[key][0]
      second2Level[figer] =[rowId,rowId+1]
  if hasHier==False:
    second2Level[figer]=[row]
    row += 1
      
figerHier = np.zeros((row,len(figer2id)))
figer2level = dict(first2Level,**second2Level)

for ids in id2figer:
  for row in figer2level[id2figer[ids]]:
    print row,ids
    figerHier[row,ids]=1
             
             
cPickle.dump(figerHier,open('data/figer/figerhierarchical.p','wb'))

    
  