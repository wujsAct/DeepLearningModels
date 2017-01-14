# -*- coding: utf-8 -*-

'''
@time:2017/1/14
'''

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import multiprocessing
import cPickle
from getCandiates import funcs

dir_path ='data/ace'

f_input = 'data/ace/features/ent_mention_index.p'
f_output = 'data/ace/features/ace_candEnts.p'
ents = cPickle.load(open(f_input,'r'))

entstr2id = {}
id2entstr = {}

entId = 0
for entlist in ents:
  for entitem in entlist:
    entstr = entitem[2]
    if entstr not in entstr2id:
      entstr2id[entstr] = entId
      id2entstr[entId] = entstr
      entId = entId + 1
print id2entstr[22]
lent = len(id2entstr)

candiate_ent=[None]*lent
candiate_coCurrEnts=[None]*lent
result = []

for ptr in xrange(0,lent,30):
  pool = multiprocessing.Pool(processes=6)
  for ids in xrange(ptr,min(ptr+30,lent)):  #数组越界问题！
    #print '----------------------'
    #print ids,entstr
    result.append(pool.apply_async(funcs, (ids,id2entstr,lent)))
  pool.close()
  pool.join()
  
  for ret in result:
    retget = ret.get()
    ids = retget[0];candidate_ent_i=retget[1];co_occurence_ent_i=retget[2]
    candiate_ent[ids] = candidate_ent_i
    candiate_coCurrEnts[ids] = co_occurence_ent_i

para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
cPickle.dump(para_dict,open(f_output,'wb'))