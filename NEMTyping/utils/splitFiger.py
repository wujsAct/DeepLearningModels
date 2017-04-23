# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:57:29 2017

@author: DELL
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/evals/")
import random
import collections
from tqdm import tqdm
import cPickle
from evalChunk import openSentid2aNosNoid,getaNosNo2entMen

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
         
      
  
  
def splitData():
  fname = 'data/figer/sentid2aNosNoid.txt'
  docSet=set()
  
  for line in open(fname):
    line = line.strip()
    items = line.split('\t')
    
   
    aNo,sNo = items[1].split('_')
    docSet.add(aNo)
    
  print len(docSet)
  
  '''
  random extract doc as train, validation, test
  '''
  nums = int(0.1 *len(docSet))
  print nums
  validation = random.sample(list(docSet), nums)  #从list中随机获取5个元素，作为一个片断返回  
  #print validation
  
  train = docSet - set(validation)
  
  test = random.sample(list(train),nums)
  
  print set(validation) & set(test)
  return validation,test

def get_figerChunk(dir_path):
  fName = dir_path+'figer_chunk.txt'
  tag=[]
  
  for line in open(fName):
    if line not in ['\n']:
      tag.append(line.strip())
  return tag  

def getSplitData(dir_path='data/figer/'):
  chunks = get_figerChunk(dir_path)
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path)
  aNosNo2entMen = getaNosNo2entMen(dir_path)
  testa = cPickle.load(open(dir_path+"figer.testa",'rb'))
  testb = cPickle.load(open(dir_path+"figer.testb",'rb'))
  
  testa_dict={testa[i]:i for i in range(len(testa))}
  testb_dict={testb[i]:i for i in range(len(testb))}
  
  print len(testa_dict)
  print len(testb_dict)
  
  testa_entMents=[];testb_entMents=[];train_entMents=[]
  
  word = []
  chunk_id = 0
  sid = 0
  input_file_obj = open(dir_path+'figerData.txt')
  testa_outfile = open(dir_path+'features/figerData_testa.txt','w')
  testb_outfile = open(dir_path+'features/figerData_testb.txt','w')
  train_outfile = open(dir_path+'features/figerData_train.txt','w')
  for line in tqdm(input_file_obj):
    if line in ['\n', '\r\n']:
      aNosNo = sid2aNosNo[sid]
      aNo = aNosNo.split('_')[0]
      entList = aNosNo2entMen[aNosNo]
      datas = '\n'.join(word) + '\n\n'
      if aNo in testa_dict:
        testa_entMents.append(entList)
        testa_outfile.write(datas)
      elif aNo in testb_dict:
        testb_entMents.append(entList)
        testb_outfile.write(datas)
      else:
        train_entMents.append(entList)
        train_outfile.write(datas) 
      word = []
      sid += 1
    else:
      line = line.strip()
      line += '\t'+chunks[chunk_id]
      word.append(line)
      chunk_id += 1
  cPickle.dump(testa_entMents,open(dir_path+'features/testa_entMents.p','wb'))
  cPickle.dump(testb_entMents,open(dir_path+'features/testb_entMents.p','wb'))
  cPickle.dump(train_entMents,open(dir_path+'features/train_entMents.p','wb'))
  testa_outfile.close();testb_outfile.close();train_outfile.close()

if __name__ == "__main__":
  #split
  #validation,test = splitData()
  #cPickle.dump(validation,open('data/figer/figer.testa','wb'))
  #cPickle.dump(test,open('data/figer/figer.testb','wb'))
  #typeFiger()
  getSplitData()
  