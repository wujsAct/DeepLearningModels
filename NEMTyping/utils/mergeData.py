# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:21:33 2017

@author: DELL, transfer figer test data into conll formats
"""
ftoken = open('data/figer_test/exp.tokens')
tpos = open('data/figer_test/exp.pos')



tokens_list = ftoken.readlines()
pos_list = tpos.readlines()

lents = len(tokens_list)

fout = open('data/figer_test/figerData.txt','w')
for i in range(lents):
  
  tokens = tokens_list[i].strip().split(' ')
  pos = pos_list[i].strip().split(' ')
  temps = [tokens[j]+'\t'+pos[j] for j in range(len(tokens))]
  fout.write('\n'.join(temps))
  fout.flush()
  fout.write('\n\n')
  fout.flush()
fout.close()  