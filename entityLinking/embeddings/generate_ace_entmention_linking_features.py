# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/1/14
"""

import sys
sys.path.append('/home/wjs/demo/entityType/informationExtract')
import cPickle
import gensim

from getctxCnnData import get_cantent_mid,get_ent_word2vec_cands,get_freebase_ent_cands


def get_final_ent_cands():
  all_candidate_mids=[]
  allentmention_numbers = 0
  
  for i in range(len(ent_Mentions)):
    ents = ent_Mentions[i]
    
    
    for j in range(len(ents)):
      allentmention_numbers += 1
      
      enti = ents[j][2]; 
      
      enti_name = enti.content.lower();entids = entstr2id[enti_name]   
      
      listentcs = []
      for entid in entids:
        listentcs += (candiate_ent[entid])
      cantent_mid1 = get_cantent_mid(listentcs,w2fb,wikititle2fb)   #get wikidata&dbpedia search candidates
      
      if enti_name in wikititle2fb:
        for wmid in wikititle2fb[enti_name]:
          #cantent_mid1[wmid] = enti_name
          cantent_mid1[wmid] = [1,0,0]
  
      cantent_mid2 = get_ent_word2vec_cands(enti,w2fb,wikititle2fb,w2vModel,entstr2id,candiate_ent,cantent_mid1) #get word2vec coherent candidates
      
      freebaseNum = max(0,30 - len(cantent_mid2))
      
      cantent_mid3 = get_freebase_ent_cands(cantent_mid2,enti,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum) #search by freebase matching 这部分将花费很多的时间！
      final_mid = cantent_mid3
      
      #print cantent_mid1,cantent_mid2,cantent_mid3
      #exit()
        
      all_candidate_mids.append(final_mid)
      
  print len(all_candidate_mids), allentmention_numbers
  

ent_Mentions = cPickle.load(open('data/ace/features/ent_mention_index.p'))
print ent_Mentions

f_input = "data/ace/features/ace_candEnts.p"
f_output = "data/ace/features/ace_ent_cand_mid"
data = cPickle.load(open(f_input,'r'))
entstr2id = data['entstr2id']

print 'entstr2id',len(entstr2id)
print entstr2id
id2entstr = {value:key for key,value in entstr2id.items()}

candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']

w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
wikititle_reverse_index  = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitleReverseIndex.p','rb'))
#print wikititle_reverse_index
  
print 'start to solve problems...'
w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
all_candidate_mids = get_final_ent_cands()

cPickle.dump(all_candidate_mids,open(f_output,'wb'))