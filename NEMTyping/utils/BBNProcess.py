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
import random
from evalChunk import openSentid2aNosNoid,getaNosNo2entMenOntos
import collections
from scipy import spatial
from sklearn import preprocessing
'''
@generate the type dictionary!
'''
def typeFiger():
  figer2id = {}
  typeId = 0
  fname = 'data/BBN/features/train_gold_entMen2aNosNoid.txt'
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
         print fbType
         
  print typeId
  cPickle.dump(figer2id,open('data/BBN/type2id.p','wb'))

def splitData():
  fname = 'data/BBN/features/test_sentid2aNosNoid.txt'
  docSet=set()
  aNo2sNo= collections.defaultdict(int)
  for line in open(fname):
    line = line.strip()
    items = line.split('\t')
    
    #print items
    
    if len(items[1].split('-'))!=2:
      print items
    aNo = items[1].split('-')[0]
    
    aNo2sNo[aNo] += 1
    docSet.add(aNo)
    
  print 'all docs:',len(docSet)
  
  '''
  random extract doc as train, validation, test
  '''
  nums = int(0.1 * len(docSet))
  print nums
  
  validation = random.sample(list(docSet), nums)  #从list中随机获取5个元素，作为一个片断返回  
  #print validation
  valNums = 0
  for key in validation:
    valNums += aNo2sNo[key]
  allNums = 0
  for key in docSet:
    allNums += aNo2sNo[key]
  print 'allNums:',allNums

  print 'all val nums:',valNums, allNums-valNums
  
  return validation

def getSplitData(dir_path='data/BBN/'):
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,"test")
  aNosNo2entMen = getaNosNo2entMenOntos(dir_path,"test")
  #testa = cPickle.load(open(dir_path+"figer.testa",'rb'))
  testa = splitData()
  print 'testa nums :',len(testa)
  
  testa_dict={testa[i]:i for i in range(len(testa))}
  #testb_dict={testb[i]:i for i in range(len(testb))}
  
  print 'testa nums:',len(testa_dict)
  
  #print len(testb_dict)
  trainType = np.zeros((113,))
  valType = np.zeros((113,))
  testa_entMents=[];
  testb_entMents=[];
  allEnts = 0
  testaEnts=0
  word = []
  maxLents = 0
  sid = 0
  input_file_obj = open(dir_path+'features/BBNData_test.txt')
  testa_outfile = open(dir_path+'features/BBNData_testa.txt','w')
  testb_outfile = open(dir_path+'features/BBNData_testb.txt','w')
  for line in tqdm(input_file_obj):
    if line in ['\n', '\r\n']:
      aNosNo = sid2aNosNo[sid]
      #aNo = aNosNo.split('_')[0]
      aNo = aNosNo.split('-')[0]   
      entList = aNosNo2entMen[aNosNo]
#      newEntList =[]
#      for ent in entList:
#
#        if ent[3].lower() not in pronominal_words:
#          newEntList.append(ent)
      allEnts += len(entList)
      datas = '\n'.join(word) + '\n\n'
      maxLents = max(maxLents,len(word))
      if aNo in testa_dict:
        testaEnts += len(entList)
        testa_entMents.append(entList)
        testa_outfile.write(datas)
        for enti in entList:
          for typei in sorted(list(set(enti[2]))):
            valType[typei] += 1
      else:
        for enti in entList:
          for typei in sorted(list(set(enti[2]))):
            trainType[typei] += 1
        testb_entMents.append(entList)
        testb_outfile.write(datas) 
      word = []
      sid += 1
    else:
      line = line.strip()
      word.append(line)
  print 'testa ent numbers:',testaEnts
  print 'all test ent numbers:',allEnts
  print abs(testaEnts*1.0/allEnts - 0.1)
  if abs( testaEnts*1.0/allEnts) - 0.1 <= 0.001:
    cPickle.dump(testa_entMents,open(dir_path+'features/testa_entMents.p','wb'))
    cPickle.dump(testb_entMents,open(dir_path+'features/testb_entMents.p','wb'))
  testa_outfile.close();
  testb_outfile.close()
  return testaEnts*1.0/allEnts - 0.1
  
  

'''
@generate the ents
'''  
def getEnts(dir_path,tag,pronominal_words):
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,tag)
  print len(sid2aNosNo)
  aNosNo2entMen = getaNosNo2entMenOntos(dir_path,tag)
  
 # print aNosNo2entMen
  
  entMents =[]
  entMentNums = 0
  for sid in tqdm(range(len(sid2aNosNo))):
    aNosNo = sid2aNosNo[sid]
    #newEntList =[]
    entList = aNosNo2entMen[aNosNo]
    entMentNums += len(entList)
    '''
    @we need to delete the pronominal mentions
    '''
    
    #for ent in entList:

      #if ent[3].lower() not in pronominal_words:
        #newEntList.append(ent)
       # entMentNums+=1
      #print ent
    entMents.append(list(entList))
  print 'entMents:',entMentNums
  cPickle.dump(entMents,open(dir_path+'features/'+tag+'_entMents.p','wb'))
  
  

'''
@generate same test as the Sonse Shimaoka 2017
'''
#def getTestEnts(dir_path,tag,pronominal_words):
#  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,tag)
#  #print len(sid2aNosNo)
#  #aNosNo2entMen = getaNosNo2entMenOntos(dir_path,tag)
#  
#  sent2id={}
#  hasfindSent={}
#  sid = 0
#  for line in open(dir_path+'features/OntoNotes_test_sent.txt'):
#    line = line.strip()
#    sent2id[line] = sid
#    sid += 1
#  rightsent = set()
#  for line in open(dir_path+'features/'+'SonseTest.txt'):
#    line = line.strip()
#    items = line.split('\t')
#    start = items[0]; end = items[1]; sent = items[2].strip(); types = items[3]
#    
#    if sent in sent2id:
#      rightsent.add(sent2id[sent])
#    else:
#      print sent
#  print len(sent2id),len(rightsent)

'''
@generate validation tests!
'''
words = cPickle.load(open('data/stopwords.p'))
pronominal_words={}
for key in words:
  pronominal_words[key]=1
#pronominal_words={}
#for line in open('data/pronominal_words.txt'):
#  line = line.strip()
#  pronominal_words[line.lower()]=1
                   
dir_path = 'data/BBN/'
tag = 'train'
#typeFiger()
#getEnts(dir_path,tag,pronominal_words)
#getTestEnts(dir_path,tag,pronominal_words)


tag = True
while tag:    
  print '----------------------'
  sim = getSplitData()
  if abs(sim) <=0.001:
    tag=False
  print '----------------------'
