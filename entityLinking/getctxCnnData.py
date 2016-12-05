# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
@editor: wujs
@function: to generate the entity linking features
'''

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import cPickle
from PhraseRecord import EntRecord

def semantic_relateness()


if __name__=='__main__':
  if len(sys.argv) !=4:
    print 'usage: python pyfile dir_path inputfile train_entms.p100(test)'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  +'/process/'+ sys.argv[2]
  f_input_entMents = dir_path  +'/features/'+ sys.argv[3]
  
  #data context:  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
  data = cPickle.load(open(f_input,'r'))
  entstr2id = data['entstr2id'];
  id2entstr = {value:key for key,value in entstr2id.items()}
  candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
  
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_entMents,'r'))
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents'];ent_ctxs=dataEnts['ent_ctxs']
  for i in range(len(ent_Mentions)):
    ents = ent_Mentions[i]
    ent_ctx = ent_ctxs[i]  #ent_ctx.append([aNo,ctx])
    for j in range(len(ents)):
      enti = ents[j]
      aNo = ent_ctx[j][0]
      entlists = list(aNo_has_ents[aNo])  #不要随意把一个对象赋值给另一个对象，否则可能会出错呢！
      entid = entstr2id[enti.content]
      entlists.remove(enti.content.lower())
      print '----------------------------------'
      print 'entlists:',entlists
      print candiate_ent[entid][0]
      print candiate_coCurrEnts[entid][0]
      print '----------------------------------'
      
  
  