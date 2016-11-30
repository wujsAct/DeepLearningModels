# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 16:03:18 2016

@author: DELL
"""
'''
@Input: txt file
@output: tag information
@function: 获取文件的tag信息
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('../utils')

from tags2id import PosTag2id
from spacyUtils import spacyUtils

import codecs
from spacy.en import English
    
def pro_sen(line,nlp):
  lineNo,sentence = line.split(u'\t')  
  retValue = []
  retValue.append(lineNo)
  if sentence!='':
    spUtils = spacyUtils(sentence,nlp)
    tags = spUtils.getPosTags()
    if tags!=None:
      retValue.append(tags)
    if len(retValue)==2:
        return retValue
    else:
        return None
    
def completedCallback1(data):
  try:
    if data!=None:
      lineNo = data[0]
      tags = data[1]
    if tags!=None:
      sys.stdout.write(lineNo+'\n')
      for k in tags:
        sys.stdout.write(str(k[0])+'\t'+k[1]+'\n')
        sys.stdout.write('\n')
  except:
    exit(-1)

if __name__ == '__main__':
  if len(sys.argv) !=3:
    print 'usage: python pyfile dir_path input_name'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path+'/'+sys.argv[2]
    
  nlp = English()
  iters = 1
  try:
    with codecs.open(f_input,'r','utf-8') as file:
      for line in file:
        data = pro_sen(line,nlp)
        completedCallback1(data)             
    except KeyboardInterrupt:
        print 'control-c presd butan'


        