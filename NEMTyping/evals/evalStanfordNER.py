# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:54:43 2017

@author: DELL
@function: evaluate the stanford ner for figer and webquestion
"""
import re
import codecs
import cPickle
import numpy as np

def getChunkNER(strs):
  entMents = set()
  for i in range(4):
    classType = str(i) + r'+'   #greed matching, find the longest substring.
    #print classType
    pattern = re.compile(classType)

    matchList = re.finditer(pattern,strs)  #very efficient layers!
    for match in matchList:
      entMents.add(str(match.start())+'_'+str(match.end()))
    
  return entMents

def getChunkNER1(strs):
  entMents = set()
  classType = r'01*'   #greed matching, find the longest substring.
  pattern = re.compile(classType)

  matchList = re.finditer(pattern,strs)  #very efficient layers!
  for match in matchList:
    entMents.add(str(match.start())+'_'+str(match.end()))
  return entMents
    
def getConll2003EntMent():
  data_tag = cPickle.load(open('data/conll2003/nerFeatures/testb_tag.p300'))
  tag = np.argmax(data_tag,2)
  entMentList = []

  for i in range(len(tag)):
    strs = ''.join(map(str,list(tag[i])))
    entMentList.append(getChunkNER1(strs))
  return entMentList
    
def getStanfordNER():
  if dir_path=='data/WebQuestion/':
    dataFile = open(dir_path+'features/train_stanfordNER.txt')
  else:
    dataFile = open(dir_path+'stanfordNER.txt')
  stanfordNERSets =[]
  
  strs = []
  type2id = {'PERSON':'0','LOCATION':'1','ORGANIZATION':'2','MISC':'3'}
  for line in dataFile:
    if line in ['\n']:
      #print strs
      stanfordNERSets.append( getChunkNER(''.join(strs)))
      strs=[]
    else:
      line = line.strip()
      if line in type2id:
        strs.append(type2id[line])
      else:
        strs.append('4')
  return stanfordNERSets
  
def getFigerNER(sentenceNums,figerData):

  ansi_escape = re.compile(r'\x1b[^m]*m')
  #figerData = codecs.open(dir_path+"features/figer_test.ret",'r','utf-8')
  figerNERSets = [None]*sentenceNums
  
  for line in figerData:
    print line
    line = line.strip()
    if '[s0]mention' in line:
      lineNo = line.split(']')[1].strip().replace('[l','')
      
      lineNo = int(ansi_escape.sub('', lineNo))
      print lineNo
      if figerNERSets[lineNo] == None:
        figerNERSets[lineNo]= set()
        
      
      ids = line.find('[s0]mention')+10
      mentionIndex = line[ids:].split('=')[0].split('(')[1].replace(')','').replace(",",'_').strip()
      figerNERSets[lineNo].add(mentionIndex)
  for i in range(sentenceNums):
    if figerNERSets[i] == None:
      figerNERSets[i]=set()
  return figerNERSets


#dir_path = 'data/conll2003/'
#dir_path = 'data/figer_test/'

      
'''
@dir_path = 'data/OntoNotes'
'''
dir_path = 'data/OntoNotes/'
targetNER= cPickle.load(open(dir_path+'features/test_entMents.p'))

targetNERSets=[]
for i in range(len(targetNER)):
  temps = set()
  for j in range(len(targetNER[i])):
    temps.add(targetNER[i][j][0]+"_"+targetNER[i][j][1])
  targetNERSets.append(temps)
  
sentence_num = len(targetNERSets)    
figerNERSets = getFigerNER(sentence_num,open(dir_path+'features/OntoNotes_testRet'))
right = 0.0
allPred = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i],figerNERSets[i]
  right += len(set(targetNERSets[i]) & figerNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(figerNERSets[i])

recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'figer:', precision,recall,f1

stanfordNERSets = getStanfordNER()

allPred=0.0
right = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i]
  right += len(set(targetNERSets[i]) & stanfordNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(stanfordNERSets[i])

recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'stanford:', precision,recall,f1


'''
@dir_path = 'data/WebQuestion/'
targetNER= cPickle.load(open(dir_path+'features/test_entMents.p'))

targetNERSets=[]
for i in range(len(targetNER)):
  temps = set()
  for j in range(len(targetNER[i])):
    temps.add(targetNER[i][j][0]+"_"+targetNER[i][j][1])
  targetNERSets.append(temps)

sentence_num = len(targetNERSets)    
figerNERSets = getFigerNER(sentence_num,open(dir_path+'test.figerRet'))
right = 0.0
allPred = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i],figerNERSets[i]
  right += len(set(targetNERSets[i]) & figerNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(figerNERSets[i])
  
recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'figer:', precision,recall,f1

stanfordNERSets = getStanfordNER()

allPred=0.0
right = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i]
  right += len(set(targetNERSets[i]) & stanfordNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(stanfordNERSets[i])

recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'stanford:', precision,recall,f1
'''


'''
targetNER= cPickle.load(open(dir_path+'features/figer_entMents.p'))
targetNERSets=[]
for i in range(len(targetNER)):
  temps = set()
  for j in range(len(targetNER[i])):
    temps.add(targetNER[i][j][0]+"_"+targetNER[i][j][1])
  targetNERSets.append(temps)
stanfordNERSets = getStanfordNER()
allPred=0.0
right = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i]
  right += len(set(targetNERSets[i]) & stanfordNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(stanfordNERSets[i])

recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'stanford:', precision,recall,f1


figerNERSets = getFigerNER()
right = 0.0
allPred = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i],figerNERSets[i]
  right += len(set(targetNERSets[i]) & figerNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(figerNERSets[i])
  
recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'figer:', precision,recall,f1
'''

'''
targetNERSets=getConll2003EntMent()
stanfordNERSets = getStanfordNER()
print len(stanfordNERSets)
allPred=0.0
right = 0.0
allEnts = 0.0
for i in range(len(targetNERSets)):
  #print targetNERSets[i]
  right += len(set(targetNERSets[i]) & stanfordNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(stanfordNERSets[i])

recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'stanford:', precision,recall,f1

figerData = codecs.open(dir_path+"testb.figerRet",'r','utf-8')
figerNERSets = getFigerNER(len(targetNERSets),figerData)
right = 0.0
allPred = 0.0
allEnts = 0.0
for i in range(len(targetNERSets)):
  #print targetNERSets[i],figerNERSets[i]
  right += len(set(targetNERSets[i]) & figerNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(figerNERSets[i])
  
recall = right/allEnts
precision = right/allPred
f1 = recall*precision*2/(recall+precision)
print 'figer:', precision,recall,f1
'''