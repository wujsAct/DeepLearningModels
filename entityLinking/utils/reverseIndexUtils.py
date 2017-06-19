# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 16:03:18 2016
revise on 2016/12/14
@author: DELL
"""
'''
@Input: doc_a={'id':'a','words':['word_w','word_x','word_y']}
@function: build reverse index for docments
'''

from collections import defaultdict
from tqdm import tqdm
import string
from string import maketrans
from wiki2fb import getWikititle2Freebase


def getreverseIndex():
  intab = string.punctuation
  outtab=''
  for key in intab:
    outtab+=' '
  trantab = maketrans(intab, outtab)
  
  wikititle2fb = getWikititle2Freebase()
  docs = wikititle2fb.keys()
  print len(docs),type(docs)
  indices = defaultdict(dict)
  print 'start to process datas'
  
  for doc in tqdm(docs):
    doc_new = str(doc).translate(trantab)   #translate do not support unicode!
    
    words_org = set(doc.split(' '))
    words_new = set(doc_new.split(' '))
    words = words_org|words_new
    if ' ' in words:
      words.remove(' ')
    if '' in words:
      words.remove('')
    for word in words:
      indices[word][doc] = 1
    indices[doc][doc] = 1
    '''
    @delete the bracket 
    '''
    rawdoc = doc.split('(')[0]
    indices[doc][rawdoc] = 1
  return indices