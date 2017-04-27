# -*- coding: utf-8 -*-
"""
Created on Wed April 19 15:26:24 2017
@author: wujs
function: chunk evaluation
"""
from tqdm import tqdm
import re
import collections
import cPickle


dir_path = 'data/figer/'

#first chunk + Freebase alias
def openSentid2aNosNoid(dir_path):
  sNo = 0
  sid2aNosNo = {}
  aNosNo2sid = {}
  fName = dir_path +'sentid2aNosNoid.txt'
  for line in open(fName):
    line = line.strip()
    
    sid,aNosNo = line.split('\t')
    sid2aNosNo[sNo] = aNosNo
    aNosNo2sid[aNosNo] = sNo
    sNo += 1
  return sid2aNosNo,aNosNo2sid



def getaNosNo2entMen(dir_path):
  fb2figer = cPickle.load(open(dir_path+'fb2figer.p','rb'))
  figer2id = cPickle.load(open(dir_path+'figer2id.p','rb'))
  totalEnts = 0
  figerTypeEnts = 0
  #lists: [[entSatrt,entEnd],...]
  #aNosNo2entMen= collections.defaultdict(set)
  aNosNo2entMen = collections.defaultdict(list)
  fName = dir_path +'gold_entMen2aNosNoid.txt'
  for line in open(fName):
    line = line.strip()
    items = line.split('\t')
    
    aNosNo = items[1]
    ents = items[2];ente = items[3]
    typeList=[]
    flag = False
    totalEnts += 1
    for i in range(4,len(items)):
      fbType = items[i]
      if fbType in fb2figer:
        typeList.append(fb2figer[fbType])
        flag = True
        
    if flag==False:
      typeList.append(len(figer2id)) 
    else:
      figerTypeEnts += 1
    #aNosNo2entMen[aNosNo].add(ents+'_'+ente)
    aNosNo2entMen[aNosNo].append([ents,ente,typeList,items[0]])
  print totalEnts,figerTypeEnts
  return aNosNo2entMen

def isFullEnts(tag,indexs):
  ent_s = int(indexs[0]); ent_e = int(indexs[1])
  #print tag[max(0,ent_s-1):ent_e]
  if tag[ent_s]=='B-NP' or (tag[ent_s]=='I-NP' and (ent_s-1==0 or tag[ent_s-1] not in ['B-NP','I-NP'])):
    if ent_e > ent_s+1:
      for i in range(ent_s+1,ent_e):
        if tag[i] != 'I-NP':
          return False
    #print list(set(indexs[2]))
#    if ent_s-1 ==0:
#      print 'ents=0:',tag[0:min(ent_e+1,len(tag))],indexs[3]
#    else:
#      print 'ents>0:',tag[ent_s-1:min(ent_e+1,len(tag))],indexs[3]
    return True
  return False

def allLents():
  fName = dir_path+'figerData.txt'
  allLents = []
  tag=[]
  for line in open(fName):
    if line in ['\n']:
      allLents.append(len(tag))
      tag=[]
    else:
      tag.append(line)
  return allLents

#evaluate the chunk

def evalFiger():
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path)
  aNosNo2entMen = getaNosNo2entMen(dir_path)
  fName = dir_path+'figer_chunk.txt'
  right = 0
  alls = 0
  sid=0
  tag = []
  for line in tqdm(open(fName)):
    if line in ['\n']:
      if sid %50000==0:
        print sid
      #evluations
      #print 'sid:',sid,allLents[sid]
      aNosNo = sid2aNosNo[sid]
      ents = aNosNo2entMen[aNosNo]
      for enti in ents:
        if enti[2][0]!=113:
          alls += 1
          if isFullEnts(tag,enti):
            right += 1
          
      tag=[]
      sid += 1
    else:
      line = line.strip()
      tag.append(line)
  return alls,right
  
#alls,right = evalFiger()
#print alls,right,right * 1.0/alls
'''

def getChunkNER(strs):
  nerSet = set()
  for i in range(4):
    classType = r''+str(i)+r'+'
    pattern = re.compile(classType)

    match = pattern.search(strs)
    if match:
      nerSet.add(str(match.start())+'_'+str(match.end()))
  return nerSet
  

def evalFigerNER():
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path)
  aNosNo2entMen = getaNosNo2entMen(dir_path,)
  fName = dir_path+'figer_ner_conll2003.txt'
  
  right = 0
  alls = 0
  sid=0
  tag = []
  for line in open(fName):
    if line in ['\n']:
      if sid %50000==0:
        print sid
      aNosNo = sid2aNosNo[sid]
      ents = aNosNo2entMen[aNosNo]  #targets ent lists
      nerSet = getChunkNER(''.join(tag))
      right += len(ents & nerSet)
      alls += len(ents)
      tag = []
      sid += 1
    else:
      tag.append(line.strip())
  return alls,right

#alls,right = evalFigerNER()
#print alls,right,right * 1.0/alls
'''
  