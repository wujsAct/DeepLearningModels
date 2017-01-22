# -*- coding: utf-8 -*-
'''
@time: 2016/12/30
@editor: wujs
@function: to generate the final candidate
'''

import os
import sys
import math
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import cPickle
import Levenshtein
from PhraseRecord import EntRecord
import codecs
import gensim
import string
from tqdm import tqdm
import collections

sentence_list=[u'Swiss',u'Grand',u'Prix',u'World',u'Cup',u'cycling']
mentag = [u'I-MISC',u'B-MISC',u'I-MISC',u'B-MISC',u'I-MISC',u'O']

lent = len(mentag)
entMen = []
entType= []
p = 0;q = 0
while(q<lent and p < lent):
  if mentag[p]==U'O':
    p = p + 1
    q = p
  else:
    if mentag[q]==U'O':
      entName = u' '.join(sentence_list[p:q])
      if entName.lower() == u'county':
        print sentence_list
        exit()
      temp = EntRecord(p,q)
      temp.setContent(sentence_list)
      entMen.append(temp)
      entType.append(mentag[p])
      print u' '.join(sentence_list[p:q]),'\t',mentag[p]
      p=q+1
      q=p
    else:
      if (mentag[q] == mentag[p] and (p==q or mentag[q].split('-')[0]!=u'B')) or (mentag[q].split('-')[1] == mentag[p].split('-')[1] and mentag[q].split('-')[0]!=u'B'): #这个地方会有一点问题呢!
        print 'p:',p,' q:',q
        q = q + 1
        if q == lent:
          print u' '.join(sentence_list[p:q]),'\t',mentag[p]
          entName = u' '.join(sentence_list[p:q])
          if entName.lower() == 'county':
            print sentence_list
            exit()
          temp = EntRecord(p,q)
          temp.setContent(sentence_list)
          entMen.append(temp)
          entType.append(mentag[p])
          break
      else:
        entName = u' '.join(sentence_list[p:q])
        if entName.lower() == 'county':
          print sentence_list
          exit()
        temp = EntRecord(p,q)
        temp.setContent(sentence_list)
        entMen.append(temp)
        entType.append(mentag[p])
        print u' '.join(sentence_list[p:q]),'\t',mentag[p]
        p = q
        print 'p:',p
#for ent in entMen:
#  print ent.getContent()