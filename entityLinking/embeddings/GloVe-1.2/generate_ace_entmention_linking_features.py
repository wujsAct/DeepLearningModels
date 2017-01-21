# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/1/14
"""

import sys
sys.path.append('/home/wjs/demo/entityType/informationExtract')
sys.path.append('/home/wjs/demo/entityType/informationExtract/utils')
from description_embed_model import MyCorpus,WordVec
import cPickle
import gensim
import numpy as np
from tqdm import tqdm
import time
import codecs
import collections
from getctxCnnData import get_cantent_mid,get_ent_word2vec_cands,get_freebase_ent_cands
from mongoUtils import mongoUtils
from generate_entmention_linking_features import processDescription
mongoutils= mongoUtils()


def get_final_ent_cands():
  all_candidate_mids=[]
  allentmention_numbers = 0
  
  for i in range(len(ent_Mentions)):
    ents = ent_Mentions[i]
    
    
    for j in range(len(ents)):
      allentmention_numbers += 1
      
      enti = ents[j][2]; 
      
      enti_name = enti.lower();entids = entstr2id[enti_name]   
      
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
  return all_candidate_mids
  

def get_all_candidate_mid_cocurrent():
  allentmention_numbers = 0
  allcandents=collections.defaultdict(set)
  k=-1
  for i in tqdm(range(len(ent_Mentions))):
    ents = ent_Mentions[i]
    
    
    for j in range(len(ents)):
      k += 1
      allentmention_numbers += 1
      
      #enti = ents[j][2];enti_name = enti.lower()
      cand_mid_dict = all_candidate_mids[k]
      for mid in cand_mid_dict:    #注意细节的处理，不然数据处理非常麻烦呢！
        midt = mid.replace(u'/',u'.')[1:]
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>' #/m/0h5k 转换成freebase中完整的格式啦！
        #print new_mid
        if new_mid not in allcandents:
          allcandents[new_mid] = mongoutils.get_coOccurent_ents(new_mid)
  cPickle.dump(allcandents,open('data/ace/features/ace_ent_relcoherent.ptemp','wb'))


def get_candidate_rel_features():
  ent_mention_relCoherent_feature=[]
  k=-1
  for i in tqdm(range(len(ent_Mentions))):
    ents = ent_Mentions[i]
    
    doc_ents_cand_mid_dict=[]
    for j in range(len(ents)):
      k+=1
      #enti = ents[j][2]; #enti_name = enti.lower()
      
      cand_mid_dict = all_candidate_mids[k]
      cand_mid_coocurmid = []
      for mid in cand_mid_dict:
        midt = mid.replace(u'/',u'.')[1:]
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>' #/m/0h5k 转换成freebase中完整的格式啦！
        #print new_mid
        
        cand_mid_coocurmid.append(allcandents_coents[new_mid])
      doc_ents_cand_mid_dict.append(cand_mid_coocurmid)
      
    doc_temprelCoherent=[]
    for i in range(len(doc_ents_cand_mid_dict)):
      temprelCoherent = np.zeros((30,))
      i_coocurmid=doc_ents_cand_mid_dict[i]
      for ci in range(len(i_coocurmid)):
        midi=i_coocurmid[ci]
        hasRel = 0
        for j in range(len(doc_ents_cand_mid_dict)):
          j_coocurmid=doc_ents_cand_mid_dict[j]
          if i!=j:
            for midj in j_coocurmid:
              if len(midi&midj)!=0:
                hasRel +=1
        temprelCoherent[ci] = hasRel
      doc_temprelCoherent.append(temprelCoherent)
    ent_mention_relCoherent_feature.append(doc_temprelCoherent)
  print len(ent_mention_relCoherent_feature),len(ent_mention_relCoherent_feature[3]),len(ent_mention_relCoherent_feature[3][0])
  cPickle.dump(ent_mention_relCoherent_feature,open('data/ace/features/ace_ent_relcoherent.p','wb'))
      
def get_candidate_ent_features():
  non_description_mid = set()
  descrip_lent=[]
  '''
  此处特征的产生比较拗口呢！
  '''
  ent_mention_index=[]
  ent_mention_link_feature=[] #shape: [line,ent_mention_num, 30(candidates number) * 100(dimension)]
  ent_mention_tag = []  #shape:[line,ent_mention_num,30]
  ent_mention_type_feature=[]   #shape:[line,ent_mention_num,30* 113(one hot type dimension)]
  ent_mention_cand_prob_feature=[]
  
  k  = -1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(range(100)):
    ents = ent_Mentions[i]
    temps =[]
    temps_type=[]
    temps_tag = []
    temps_cand_prob=[]
    temps_ent_index=[]
    #print i,'\tentM:',len(ents)
    for j in range(len(ents)):
      enti = ents[j][2];#enti_name = enti.lower()
      print ents[j]
      k += 1  
      ent_mention_tag_temp = np.zeros((30,))
      tdescip = [];tcanditetype=[];tcandprob = []
      
      cand_mid_dict = all_candidate_mids[k]
  
      for mid in cand_mid_dict:  #仅有description,实体共现！
        if mid not in mid2description:   #用来去抓取需要linking的实体啦！filter to ensure all the result candidates has the description!
          print 'mide not in mid2description', mid
          #exit(-1)
      
        twordv = np.zeros((100,)) 
        ttypev = np.zeros((113,))
        if mid in mid2figer:         #也有很多实体并没有figer的type所以这个特征也为0
          for midtype in mid2figer[mid]:
            ttypev[midtype]=1
        
        if mid in mid2description:  #不需要这个条件的哈，因为很多实体都缺少这个条件，对于那些不popular的候选实体啦！
          line = mid2description[mid]
          descript = processDescription(line)
          descrip_lent.append(len(descript))
          #print descript
          for i in range(min(15,len(descript))):
            word = descript[i]
            #print 'word:',word
            if word in descript_Words:
              twordv += descript_Words[word]
        '''
        @add candidate entity type features!
        '''
        if len(tcandprob)==0:
          tcandprob = np.asarray(cand_mid_dict[mid])
        else:
          tcandprob = np.concatenate((tcandprob,np.asarray(cand_mid_dict[mid])))
        
        if len(tcanditetype)==0:
          tcanditetype = ttypev
        else:
          tcanditetype = np.concatenate((tcanditetype,ttypev))
         
        if len(tdescip)==0:
          tdescip = twordv
        else:
          tdescip = np.concatenate((tdescip,twordv))
        assert np.shape(tcanditetype)[0]/113 == np.shape(tdescip)[0]/100
        assert np.shape(tcandprob)[0]/3 == np.shape(tcanditetype)[0]/113
      temps_type.append(tcanditetype)
      temps.append(tdescip)
      temps_tag.append(ent_mention_tag_temp)
      temps_cand_prob.append(tcandprob)
      #temps_ent_index.append((enti.startIndex,enti.endIndex))  #通过这个flag去抽取lstm最后一层的特征啦！
      temps_ent_index.append((enti[j][0],enti[j][1]))
    ent_mention_type_feature.append(temps_type)
    ent_mention_cand_prob_feature.append(temps_cand_prob)
    ent_mention_link_feature.append(temps)
    ent_mention_tag.append(temps_tag)
    ent_mention_index.append(temps_ent_index)   
  print len(ent_mention_link_feature),len(ent_mention_link_feature[3]),len(ent_mention_link_feature[3][0])
  print len(ent_mention_tag),len(ent_mention_tag[3]),len(ent_mention_tag[3][0])
  print len(ent_mention_index),len(ent_mention_index[3]),len(ent_mention_index[3][0])
  print  max(descrip_lent),min(descrip_lent),sum(descrip_lent) / float(len(descrip_lent))
  param_dict = {'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag}
  cPickle.dump(param_dict,open("data/ace/features/ace_ent_linking.p",'wb'))
  print len(ent_mention_type_feature),len(ent_mention_type_feature[3]),len(ent_mention_type_feature[3][0])
  cPickle.dump(ent_mention_type_feature,open("data/ace/features/ace_ent_linking_type.p",'wb'))
  cPickle.dump(ent_mention_cand_prob_feature,open("data/ace/features/testa_ent_linking_candprob.p",'wb'))     
              
            
            
ent_Mentions = cPickle.load(open('data/ace/features/ent_mention_index.p'))
print ent_Mentions

'''
#generate all_candidate_mids
f_input = "data/ace/features/ace_candEnts.p"
f_output = "data/ace/features/ace_ent_cand_mid.p"

data = cPickle.load(open(f_input,'r'))
entstr2id_org = data['entstr2id']
print 'entstr2id_org',len(entstr2id_org)
id2entstr_org = {value:key for key,value in entstr2id_org.items()}
entstr2id= collections.defaultdict(set)
for key,value in entstr2id_org.items():
  entstr2id[key.lower()].add(value)
print 'entstr2id',len(entstr2id)

candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']

w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
wikititle_reverse_index  = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitleReverseIndex.p','rb'))
#print wikititle_reverse_index
  
print 'start to solve problems...'
w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
all_candidate_mids = get_final_ent_cands()

cPickle.dump(all_candidate_mids,open(f_output,'wb'))
'''
stime = time.time()
mid2figer = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/mid2figer.p','rb'))
print 'load mid2figer cost time:',time.time()-stime
  

all_candidate_mids = cPickle.load(open("data/ace/features/ace_ent_cand_mid.p"))
#print all_candidate_mids
#get_all_candidate_mid_cocurrent()

#allcandents_coents = cPickle.load(open('data/ace/features/ace_ent_relcoherent.ptemp','rb'))   
#get_candidate_rel_features()

stime = time.time()
descript_Words = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wordvec_model_100.p', 'rb'))
print 'load wordvec_model_100 cost time: ', time.time()-stime

descript_Words = descript_Words.wvec_model

stime = time.time()
mid2description={}  #nearly 2.3G
with codecs.open('/home/wjs/demo/entityType/informationExtract/data/mid2description.txt','r','utf-8') as file:
  for line in file:
    items = line.strip().split('\t')
    if len(items) >=2:
      mid2description[items[0]] =items[1]
print 'load mid2descriptioon cost time: ', time.time()-stime

get_candidate_ent_features()