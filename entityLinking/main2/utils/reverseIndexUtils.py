# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 16:03:18 2016
revise on 2016/12/14
@author: DELL
"""
'''
@Input: doc_a={'id':'a','words':['word_w','word_x','word_y']}
@function: build reverse index for docments
在这个实验，建立freebase name reverse index,方便进行检索
'''

from collections import defaultdict
import cPickle
from tqdm import tqdm
import string
from string import maketrans

intab = unicode(string.punctuation)
trantab={}
for key in intab:
  trantab[ord(key)]=u' '  #attention to unicode 的问题啦!

wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
docs = wikititle2fb.keys()
indices = defaultdict(dict)
for doc in tqdm(docs):
  doc_new = unicode(doc).translate(trantab)   #translate do not support unicode!
  words_org = set(doc.split(u' '))
  words_new = set(doc_new.split(u' '))
  words = words_org|words_new
  if u' ' in words:
    words.remove(u' ')
  if u'' in words:
    words.remove(u'')
  for word in words:
    indices[word][doc] = 1
  #print indices

cPickle.dump(indices,open('/home/wjs/demo/entityType/informationExtract/data/wtitleReverseIndex.p','wb'))