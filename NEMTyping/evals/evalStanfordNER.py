# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:54:43 2017

@author: DELL
@function: evaluate the stanford ner for figer and webquestion
"""
import re
import codecs
import cPickle
dir_path = 'data/figer_test/'

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

def getStanfordNER():
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
  
def getFigerNER():
  rawData = open(dir_path+"features/figerData.txt")
  sentence2LineNo={}
  lineNo = 0
  sentence=[]
  for line in rawData:
    if line in ['\n']:
      #print ' '.join(sentence)
      sentence2LineNo[' '.join(sentence)]=lineNo
      
      sentence=[]
      lineNo += 1
    else:
      line = line.strip()
      sentence.append(line.split('\t')[0])      
     
  ansi_escape = re.compile(r'\x1b[^m]*m')
  figerData = codecs.open(dir_path+"features/figer_test.ret",'r','utf-8')
  figerNERSets = [None]*len(sentence2LineNo)
  
  for line in figerData:
    line = line.strip()
    if '[s0]mention' in line:
      lineNo = line.split(']')[1].strip().replace('[l','')
      
      lineNo = int(ansi_escape.sub('', lineNo))

      if figerNERSets[lineNo] == None:
        figerNERSets[lineNo]= set()
        
      
      ids = line.find('[s0]mention')+10
      mentionIndex = line[ids:].split('=')[0].split('(')[1].replace(')','').replace(",",'_').strip()
      figerNERSets[lineNo].add(mentionIndex)
  for i in range(len(sentence2LineNo)):
    if figerNERSets[i] == None:
      figerNERSets[i]=set()
  return figerNERSets
        
      
  
  

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

print 'stanford:', right/allEnts,right/allPred, right, allEnts
  

figerNERSets = getFigerNER()
print 'figer:',len(figerNERSets)
right = 0.0
allPred = 0.0
allEnts = 0.0
for i in range(len(targetNER)):
  #print targetNERSets[i],figerNERSets[i]
  right += len(set(targetNERSets[i]) & figerNERSets[i])
  allEnts += len(targetNERSets[i])
  allPred += len(figerNERSets[i])

print 'figer:', right/allEnts, right/allPred,right, allEnts

