# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
@editor: wujs
@function: to generate the entity linking features
'''
import os
import sys
import math
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import cPickle
from PhraseRecord import EntRecord
import codecs

def getAllEnts(ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb):
  allents = set()
  for i in range(len(ent_Mentions)):
    if i%100==0:
      print i
    ents = ent_Mentions[i]
    ent_ctx = ent_ctxs[i]  #ent_ctx.append([aNo,ctx])
      
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content
      aNo = ent_ctx[j][0]
      seta = set(aNo_has_ents[aNo])  #不要随意把一个对象赋值给另一个对象，否则可能会出错呢！
      entid = entstr2id[enti.content]
      seta.remove(enti.content.lower())
      allents = allents|seta
      listb = candiate_coCurrEnts[entid]
      listentcs = candiate_ent[entid]
      if len(listb)==0:
        print enti.content
      else:
        for coentid in range(len(listb)):
          setb = set()
          for item in listb[coentid]:
            setb.add(item.split(u'\t')[1].lower())
          
          ids = listentcs[coentid][u'ids']
          titles = listentcs[coentid][u'title']
          if ids in w2fb:
            allents = allents|setb
          else:
            if titles in wikititle2fb:
              allents = allents|setb
  return len(allents)

def semantic_relateness(seta,setb,W):
  maxab = math.log(1.0*max(len(seta),len(setb)),2)
  joinab = math.log(len(seta&setb)+1e-4,2)
  minab = math.log(min(len(seta),len(setb))+1e-4,2)
  return 1-(maxab-joinab)/(W-minab)

if __name__=='__main__':
  if len(sys.argv) !=5:
    print 'usage: python pyfile dir_path inputfile train_entms.p100(test) train_totalEntNum.p'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  +'/process/'+ sys.argv[2]
  f_input_entMents = dir_path  +'/features/'+ sys.argv[3]
  f_output1 = dir_path  +'/features/'+ sys.argv[4] #record the total entity nums.
  
  #data context:  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
  data = cPickle.load(open(f_input,'r'))
  entstr2id = data['entstr2id'];
  id2entstr = {value:key for key,value in entstr2id.items()}
  candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
  
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_entMents,'r'))
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents'];ent_ctxs=dataEnts['ent_ctxs']
  all_ents = set()
  w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
  wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
  print 'start to solve problems...'
  
  if os.path.isfile(f_output1):
    allents = cPickle.load(open(f_output1,'rb'))
  else:
    alllents = getAllEnts(ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb)
    cPickle.dump(alllents,open(f_output1,'wb'))
    
  print allents
  '''
  print alllents
  alllents=37000
  noncandidates=0
  havacandidates=0
  ent_candidateEnts=[]
  
  ents_description=[]
  for i in range(len(ent_Mentions)):
    if i%100==0:
      print i
    ents = ent_Mentions[i]
    ent_ctx = ent_ctxs[i]  #ent_ctx.append([aNo,ctx])
    temp_ent_candidateEnts=[]
    for j in range(len(ents)):
      enti = ents[j]
      aNo = ent_ctx[j][0]
      seta = set(aNo_has_ents[aNo])  #不要随意把一个对象赋值给另一个对象，否则可能会出错呢！
      entid = entstr2id[enti.content]
      seta.remove(enti.content.lower())
      listb = candiate_coCurrEnts[entid]
      listentcs = candiate_ent[entid]
      if len(listb)==0:
        print enti.content
      else:
        tempval = []
        fbents = []
        for coentid in range(len(listb)):
          setb = set()
          for item in listb[coentid]:
            setb.add(item.split(u'\t')[1].lower())
          
          enti_name = enti.content
          ids = listentcs[coentid][u'ids']
          titles = listentcs[coentid][u'title']
          description = listentcs[coentid][u'description']
          if ids in w2fb:
            fbents.append((w2fb[ids],titles,description))
            #description 啥的还是用原来的就ok呢！
          else:
            if titles in wikititle2fb:
              fbents.append((wikititle2fb[titles],titles,description))
        if len(fbents)==0:
          noncandidates = noncandidates + 1
          print enti.content,listentcs
        else:
          havacandidates = havacandidates + 1
        temp_ent_candidateEnts.append(fbents)
    ent_candidateEnts.append(temp_ent_candidateEnts)
  print 'has no candidates:', noncandidates
  print ent_Mentions[0],ent_candidateEnts[0]
  print 'has  candidates:', havacandidates
  '''
  