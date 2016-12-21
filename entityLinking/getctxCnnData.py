# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
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

def is_contain_ents(enti,entj):
  enti = enti.lower()
  entj = entj.lower()
  if enti in entj:
    return True
  else:
    return False

def get_freebase_ent_cands(cantent_mid_prev,enti,tag,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum):
  #找出全部包含jordan的实体，然后使用NGD和REL进行消解，找到答案！今天下午把这个问题解决掉
  #部分与整体的coreference也能够解决一部分的问题呢！
  candi = 0
  #print 'go into entNGD...'
  distRet = {};
  #first find all the ents we need to process
  enti_title = enti.content.lower()
  enti_item = enti_title.split(u' ')
  for entit in enti_item[0:1]:
    totaldict=dict()
    if entit in wikititle_reverse_index:
      totaldict = wikititle_reverse_index[entit]
      
    for key in totaldict:
      if is_contain_ents(enti_title,key):
        addScore = 0
        if enti_title == key:  #completely match
          addScore += 0.3 
        if key in entstr2id: #在上下文中出现了的！可以解决一部分共指问题！
          addScore += 0.3
        for wmid in wikititle2fb[key]:
          distRet[wmid+u'\t'+key]=Levenshtein.ratio(enti_title,key) + addScore
  distRet= sorted(distRet.iteritems(), key=lambda d:d[1], reverse = True)
  cantent_mid={}
  for item in distRet:
    if freebaseNum==0:
      break
    item_it = item[0].split(u'\t')
    if item_it[0] not in cantent_mid_prev:
      cantent_mid[item_it[0]] = item_it[1]
      freebaseNum -=1
  return cantent_mid

def get_cantent_mid(listentcs,w2fb,wikititle2fb):
  flag = False
  cantent_mid={}
  for cent in listentcs:
    ids = cent[u'ids']
    titles = cent[u'title'].lower()
    if ids in w2fb:
      cantent_mid[w2fb[ids]] = titles
    elif titles in wikititle2fb:
      for wmid in wikititle2fb[titles]:
        cantent_mid[wmid] =titles
  return cantent_mid

def get_ent_word2vec_cands(enti,w2fb,wikititle2fb):
  cantent_mid = {}
  try: 
    coherent_ents = w2vModel.most_similar(enti.content.replace(u' ',u'_'),topn=10) #convert to gensim style
    for citems in coherent_ents:
      cents = citems[0].replace(u'_',u' ')  #convert to freebase style
      if cents.lower() in entstr2id:
        entids = entstr2id[cents.lower()]
        listentcs =[]
        for entid in entids:
          listentcs += candiate_ent[entid][0:1]
        if len(listentcs)>=1:
          cantent_mid =dict(cantent_mid,**get_cantent_mid(listentcs,w2fb,wikititle2fb))
      if cents.lower() in wikititle2fb:
        for wmid in wikititle2fb[cents.lower()]:
          cantent_mid[wmid] = cents.lower()
  except:
    pass
    
  return cantent_mid

def get_final_ent_cands(w2vModel,ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb,wikititle_reverse_index):
  '''
  @2016/12/15 目前可以达到89%的覆盖率了！ cut-off 设置为30
  '''
  all_candidate_mids = []
  fileName = '/home/wjs/demo/entityType/informationExtract/data/aida/AIDA-YAGO2-dataset.tsv'
  entstr_lower2mid = {}
  mid2entstr_lower={}
  with codecs.open(fileName,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      item = line.split(u'\t')
      if len(item)==7:
        entstr_lower2mid[item[2].lower()] = item[6]
        mid2entstr_lower[item[6]] = item[3].lower()
  print 'finish load all datas'
  #exit(-1)
  right_nums = 0;wrong_nums =0
  pass_nums = 0
  k = 0
  
  for i in tqdm(range(len(ent_Mentions))):
    ents = ent_Mentions[i]
    ent_ctx = ent_ctxs[i]  #ent_ctx.append([aNo,ctx])
    
    for j in range(len(ents)):
      totalCand = 0
      
      enti = ents[j]
      
      enti_name = enti.content.lower()
      if enti_name not in entstr_lower2mid:
        pass_nums = pass_nums + 1
        continue
      else:
        tag = entstr_lower2mid[enti_name]
        
      aNo = ent_ctx[j][0]
      entids = entstr2id[enti_name]
      listentcs = []
      for entid in entids:
        listentcs += (candiate_ent[entid])
      cantent_mid1 = get_cantent_mid(listentcs,w2fb,wikititle2fb)   #get wikidata&dbpedia search candidates
      if enti_name in wikititle2fb:
        for wmid in wikititle2fb[enti_name]:
          cantent_mid1[wmid] = enti_name
  
      cantent_mid2 = get_ent_word2vec_cands(enti,w2fb,wikititle2fb) #get word2vec coherent candidates
      cantent_mid = dict(cantent_mid1,**cantent_mid2)
      
      freebaseNum = max(0,30 - len(cantent_mid))
      
      cantent_mid3 = get_freebase_ent_cands(cantent_mid,enti,tag,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum) #search by freebase matching 这部分将花费很多的时间！
      
      final_mid = dict(cantent_mid,**cantent_mid3)
      
      totalCand = len(final_mid)
      
      #if tag in cantent_mid1 or tag in cantent_mid2 or tag in cantent_mid3:
      if tag in final_mid:
        right_nums += 1
        #print 'right:',enti_name,totalCand
      else:
        wrong_nums = wrong_nums + 1
        
        print 'wrong:',tag,mid2entstr_lower[tag],enti.content,len(cantent_mid1),'\t',len(cantent_mid2),'\t',len(cantent_mid),'\t',len(cantent_mid3),'\t',totalCand
        #print cantent_mid1,cantent_mid2,cantent_mid3
        #exit()
        
      all_candidate_mids.append(cantent_mid)
  
  print 'wrong_nums:',wrong_nums
  print 'right_nums:',right_nums
  print 'pass_nums:',pass_nums
  return all_candidate_mids    



if __name__=='__main__':
  if len(sys.argv) !=5:
    print 'usage: python pyfile dir_path inputfile train_entms.p100(test) train_ent_cand_mid.p'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  +'/process/'+ sys.argv[2]
  f_input_entMents = dir_path  +'/features/'+ sys.argv[3]
  f_output = dir_path  +'/features/'+ sys.argv[4] #record the total entity nums.
  
  #data context:  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
  data = cPickle.load(open(f_input,'r'))
  entstr2id_org = data['entstr2id']
  print 'entstr2id_org',len(entstr2id_org)
  id2entstr_org = {value:key for key,value in entstr2id_org.items()}
  entstr2id= collections.defaultdict(set)
  for key,value in entstr2id_org.items():
    entstr2id[key.lower()].add(value)
  print 'entstr2id',len(entstr2id)
  candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
  #print candiate_ent
  
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_entMents,'r'))
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents'];ent_ctxs=dataEnts['ent_ctxs']
  all_ents = set()
  w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
  wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
  wikititle_reverse_index  = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitleReverseIndex.p','rb'))
  #print wikititle_reverse_index
  
  print 'start to solve problems...'
  w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
  all_candidate_mids = get_final_ent_cands(w2vModel,ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb,wikititle_reverse_index)
  cPickle.dump(all_candidate_mids,open(f_output,'wb'))
  
