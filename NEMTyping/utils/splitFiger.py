# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:57:29 2017

@author: DELL
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/evals/")
import random
import numpy as np
import collections
from tqdm import tqdm
import cPickle
from evalChunk import openSentid2aNosNoid,getaNosNo2entMen
from scipy import spatial
from sklearn import preprocessing
#split figer into train_data and testa_data and testb_data
'''
@66.5 percent of annotation entities should has the figer ner types
'''
def typeFiger():
  figer2id = {}
  tid = 0
  fname = 'data/freebasetype2figertype.map'
  fb2figer = {}
  for line in open(fname):
    line = line.strip()
    items = line.split('\t')
    if items[1] not in figer2id:
      figer2id[items[1]]=tid
      tid += 1
      
    fb2figer[items[0]] = figer2id[items[1]]
  for key in figer2id:
    print key,figer2id[key]
  print 'figer2id:',len(figer2id)
  print 'fb2figer:',len(fb2figer)
  cPickle.dump(figer2id,open('data/figer/figer2id.p','wb'))
  cPickle.dump(fb2figer,open('data/figer/fb2figer.p','wb'))
  

  fname = 'data/figer/gold_entMen2aNosNoid.txt'
  allEnts = 0
  rightEnts = 0
  for line in open(fname):
    allEnts += 1
    line = line.strip()
    
    items = line.split('\t')
    flag = False
    nums = 0
    for i in range(4,len(items)):
       fbType = items[i]
       if fbType in fb2figer:
         flag = True
         nums += 1
    if flag:
      rightEnts += 1
      #print nums
   
  print rightEnts,allEnts,rightEnts*1.0/allEnts

def get_figerChunk(dir_path):
  fName = dir_path+'figer_chunk.txt'
  tag=[]
  
  for line in tqdm(open(fName)):
    if line not in ['\n']:
      tag.append(line.strip())
  return tag  

def getFullTrainData(dir_path):
  print 'get figer chunk...'
  chunks = get_figerChunk(dir_path)
  print 'get sid2aNosNo...'
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,"train")
  print 'get aNosNo2entMents ...'
  aNosNo2entMen = getaNosNo2entMen(dir_path,"train")
  
  word = []
  chunk_id = 0
  sid = 0
  train_entMents=[]
  input_file_obj = open(dir_path+'figerData.txt')
  train_outfile = open(dir_path+'features/figerData_train.txt','w')
  
  for line in tqdm(input_file_obj):
    if line in ['\n', '\r\n']:
      aNosNo = sid2aNosNo[sid]
      entList = aNosNo2entMen[aNosNo]
      datas = '\n'.join(word) + '\n\n'

      train_entMents.append(entList)
      train_outfile.write(datas) 
      
      word = []
      sid += 1
    else:
      line = line.strip()
      line += '\t'+chunks[chunk_id]
      word.append(line)
      chunk_id += 1
  train_outfile.close()
  cPickle.dump(train_entMents,open(dir_path+'features/train_entMents.p','wb'))
  
def getSplitData():
  allEntsList = cPickle.load(open('data/figer_test/features/figer_entMents.p'))
  for key in allEntsList:
    print key
  print len(allEntsList)
  docSet = range(len(allEntsList))
  print docSet
  
  input_file_obj = open('data/figer_test/features/figerData.txt')
  nums = int(0.1 * len(allEntsList))
  print len(allEntsList),nums
  validation = random.sample(docSet, nums)
  
  #trainType = np.zeros((113,))
  #valType = np.zeros((113,))
  testa_entMents=[];
  testb_entMents=[];
  allEnts = 0
  testaEnts=0
  word = []
  maxLents = 0
  sid = 0
  
  testa_outfile = open(dir_path+'features/figerData_testa.txt','w')
  testb_outfile = open(dir_path+'features/figerData_testb.txt','w')
  for line in tqdm(input_file_obj):
    if line in ['\n', '\r\n']:
      entList = allEntsList[sid]
#      newEntList =[]
#      for ent in entList:
#
#        if ent[3].lower() not in pronominal_words:
#          newEntList.append(ent)
      allEnts += len(entList)
      datas = '\n'.join(word) + '\n\n'
      maxLents = max(maxLents,len(word))
      if sid in validation:
        testaEnts += len(entList)
        testa_entMents.append(entList)
        testa_outfile.write(datas)
#        for enti in entList:
#          for typei in sorted(list(set(enti[2]))):
#            valType[typei] += 1
      else:
#        for enti in entList:
#          for typei in sorted(list(set(enti[2]))):
#            trainType[typei] += 1
        testb_entMents.append(entList)
        testb_outfile.write(datas) 
      word = []
      sid += 1
    else:
      line = line.strip()
      word.append(line)
  print 'testa ent numbers:',testaEnts,allEnts-testaEnts
  print 'all test ent numbers:',allEnts
  print abs(testaEnts*1.0/allEnts - 0.1)
  if abs( testaEnts*1.0/allEnts) - 0.1 <= 0.001:
    cPickle.dump(testa_entMents,open(dir_path+'features/testa_entMents.p','wb'))
    cPickle.dump(testb_entMents,open(dir_path+'features/testb_entMents.p','wb'))
  testa_outfile.close();
  testb_outfile.close()
  return testaEnts*1.0/allEnts - 0.1


if __name__ == "__main__":
  #split
  #typeFiger()
  dir_path = 'data/figer/'
  #getFullTrainData(dir_path)
  
  
  
  tag = True
  while(tag):  
    sim = getSplitData()
    print sim
    if abs(sim) <=0.001:
      tag=False