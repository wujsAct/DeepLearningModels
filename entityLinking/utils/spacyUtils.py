# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 16:03:18 2016
revise on 2016/11/25
@author: DELL
"""
'''
@Input: sentence, nlp handler
@function: obtain spacy pos tag and dependecy tree
'''

from spacy.en import English
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class spacyUtils():
  def __init__(self,sentence,nlp):
    self.sentence = sentence
    self.nlp = nlp
    self.doc = nlp(self.sentence)
    
  def getPosTags(self):
    self.tags = []
    for token in self.doc:
      self.tags.append(token.tag_)
    if len(self.tags)>0:
      return self.tags
    else:
      return None
      
  def getDepTree(self):
    t = {token.idx:i for i,token in enumerate(self.doc)}
    self.dep_triple = []
    for token in self.doc:
      temp=[]
      temp.append([token.orth_,t[token.idx]])
      temp.append(self.nlp.vocab.strings[token.dep])
      temp.append([token.head.orth_,t[token.head.idx]])
      self.dep_triple.append(temp)
    if len(self.dep_triple)>0:
      return self.dep_triple
    else:
      return None
      