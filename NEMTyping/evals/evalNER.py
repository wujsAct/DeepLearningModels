# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:20:38 2017
@author: DELL
"""
import re
import numpy as np
import collections

#@function: evaluate the conll2003(B-E,I-E,O)
def getChunkNER(strs):
  entMents = set()
  classType = r'0?1*'
  pattern = re.compile(classType)

  match = pattern.search(strs)
  if match:
    entMents.add(str(match.start())+'_'+str(match.end()))
    
  return entMents
    

def f1_chunk(flag,args,prediction, target, length):
  totalPred=0;totalTarget=0;totalRight = 0  #strict F1
  
  if flag =='nonCRF':
    target = np.argmax(target,2)
    prediction = np.argmax(prediction,2)
  
  for i in range(len(target)):
    lents = length[i]
    targetRel = ''.join(map(str,list(target[i][:lents])));predRel= ''.join(map(str,list(prediction[i][:lents])))
    targetNER= getChunkNER(targetRel); predNER= getChunkNER(predRel)
    
    totalRight += len(targetNER & predNER)
    totalTarget += len(targetNER); totalPred += len(predNER)
  
  #strict F1
  precision = totalRight*1.0/totalPred
  recall = totalRight*1.0/totalTarget
    
  return 2*precision*recall/(precision+recall)


def f1(args, prediction, target, length):
  tp = np.array([0] * (args.class_size + 1))
  fp = np.array([0] * (args.class_size + 1))
  fn = np.array([0] * (args.class_size + 1))
  #target = np.argmax(target, 2)
  #prediction = np.argmax(prediction, 2) #crf prediction is this kind .
  for i in range(len(target)):
    for j in range(length[i]):
      if target[i][j] == prediction[i][j]:
        tp[target[i][j]] += 1
      else:
        fp[target[i][j]] += 1
        fn[prediction[i][j]] += 1
  unnamed_entity = args.class_size - 1
  for i in range(args.class_size):
    if i != unnamed_entity:
      tp[args.class_size] += tp[i]
      fp[args.class_size] += fp[i]
      fn[args.class_size] += fn[i]
  precision = []
  recall = []
  fscore = []
  for i in range(args.class_size + 1):
    precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
    recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
    fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
  return fscore

#classType: r'0+',r'1+',r'2+',r'3+',r'4+'
def getChunkNER_type(args,strs):
  dicts = collections.defaultdict(set)
  for i in range(args.class_size):
    classType = r''+str(i)+r'+'
    pattern = re.compile(classType)

    match = pattern.search(strs)
    if match:
      dicts[i].add(str(match.start())+'_'+str(match.end()))
  return dicts


  
def f1_chunk_type(args, prediction, target, length):
  tp = np.array([0] * (args.class_size + 1))
  fp = np.array([0] * (args.class_size + 1))
  fn = np.array([0] * (args.class_size + 1))
  #0-Person,1-Location,2-Organisation,3-Misc,4-None
  #first get the target ner_chunk labels
  for i in range(len(target)):
    lents = length[i]
    reltarget = ''.join(map(str,list(target[i][:lents])))
    relpred = ''.join(map(str,list(prediction[i][:lents])))
    
    dictsTarget = getChunkNER_type(args,reltarget)
    dictsPred = getChunkNER_type(args,relpred)
    
    for key in dictsPred:
      tp[key] += len(dictsTarget[key] & dictsPred[key])
      fp[key] += len(dictsPred[key] - dictsTarget[key])  
      fn[key] += len(dictsTarget[key]-dictsPred[key])
  
  unnamed_entity = args.class_size - 1
  for i in range(args.class_size):
    if i != unnamed_entity:
      tp[args.class_size] += tp[i]
      fp[args.class_size] += fp[i]
      fn[args.class_size] += fn[i]
  
  precision = []
  recall = []
  fscore = []
  for i in range(args.class_size + 1):
    precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
    recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
    fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
  return fscore