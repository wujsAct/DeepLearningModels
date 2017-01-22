#get aritcle_no sentence_no sentence entity_mentions

import sys
import os
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import multiprocessing
import urllib2
import codecs
import cPickle
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
from urllibUtils import urllibUtils
from getCandiates import funcs

  
if __name__=='__main__':
  if len(sys.argv) !=4:
    print 'usage: python pyfile dir_path inputfile outputfile'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  + sys.argv[2]
  f_input_canents = dir_path  + sys.argv[3]
  
  #we also need to add the candidates ents description context!
  if os.path.isfile(f_input_canents+'new'):
    data = cPickle.load(open(f_input_canents+'new','r'))
    entstr2id = data['entstr2id'];
    id2entstr = {value:key for key,value in entstr2id.items()}
    candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
  else:
    data = cPickle.load(open(f_input_canents,'r'))
    entstr2id = data['entstr2id'];
    id2entstr = {value:key for key,value in entstr2id.items()}
    lent = len(id2entstr)
    candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
    for enti in entstr2id:
      ids = entstr2id[enti]
      if len(candiate_ent[ids])==0:
        ids,candidate_enti,co_occurence_enti = funcs(ids,id2entstr,lent)
        candiate_ent[ids] = candidate_enti
        candiate_coCurrEnts[ids] = co_occurence_enti
        print '-------------------------------'
    param = {'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
    data = cPickle.dump(param,open(f_input_canents+'new','wb'))
  
  
  
