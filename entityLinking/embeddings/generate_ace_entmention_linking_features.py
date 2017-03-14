# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/1/14
"""

import sys
sys.path.append('/home/wjs/demo/entityType/informationExtract')
sys.path.append('/home/wjs/demo/entityType/informationExtract/utils')
from description_embed_model import MyCorpus,WordVec
import Levenshtein
import cPickle
import gensim
import numpy as np
import argparse
from tqdm import tqdm
import time
import codecs
import collections
from getctxCnnData import get_cantent_mid,get_ent_word2vec_cands,get_freebase_ent_cands,getfname2pageid
from mongoUtils import mongoUtils
from NGDUtils import NGDUtils
from generate_entmention_linking_features import processDescription
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
mongoutils= mongoUtils()
ngd = NGDUtils()

def getmid2Name():
  #mid2name = collections.defaultdict(list) 
  mid2name = {}
  with open('data/mid2name.tsv','r') as file:
    for line in tqdm(file):
      line =line.strip()
      #print line.split(u'\t')
      items= line.split('\t')
      
      if len(items)>=2:
        mid = items[0]; name = ' '.join(items[1:])
        mid2name[mid] = name
      else:
        print line
      
  print len(mid2name)
  return mid2name


def get_final_ent_cands():
  all_candidate_mids=[]
  allentmention_numbers = 0
  
  for i in tqdm(range(len(ent_Mentions))):
    aNosNo = sentId2aNosNo[i]
    docId = aNosNo.split('_')[0]
    context_ents = docId_entstr2id[docId]
    context_ent_pageId = set()
    for key in context_ents:
      #print 'context ents:',context_ents
      if key in title2pageId:
        context_ent_pageId.add(title2pageId[key])
          
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
        wmid_id = 0.0
        for wmid in wikititle2fb[enti_name]:
          wmid_id += 1.0
          #cantent_mid1[wmid] = enti_name
          cantent_mid1[wmid] = [1/wmid_id,0,0]
  
      cantent_mid2 = get_ent_word2vec_cands(enti,w2fb,wikititle2fb,w2vModel,context_ents,candiate_ent,cantent_mid1) #get word2vec coherent candidates
      
      freebaseNum = max(0,30 - len(cantent_mid2))
      
      cantent_mid3 = get_freebase_ent_cands(ngd,mid2name,cantent_mid2,enti,context_ent_pageId,wikititle2fb,wikititle_reverse_index,freebaseNum) #search by freebase matching è¿é¨åå°è±è´¹å¾å¤çæ¶é´ï¼
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
      for mid in cand_mid_dict:    
        midt = mid.replace(u'/',u'.')[1:]
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>'
        #print new_mid
        if new_mid not in allcandents:
          allcandents[new_mid] = mongoutils.get_coOccurent_ents(new_mid)
  cPickle.dump(allcandents,open(dir_path+'features/'+data_tag+'_ent_relcoherent.ptemp','wb'))


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
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>'
        #print new_mid
        cand_mid_coocurmid.append([new_mid,allcandents_coents[new_mid]])
      doc_ents_cand_mid_dict.append(cand_mid_coocurmid)
      
    doc_temprelCoherent=[]
    for icand in range(len(doc_ents_cand_mid_dict)):
      temprelCoherent = np.zeros((30,))
      cand_mid_coocurmid_i = doc_ents_cand_mid_dict[icand]
      for ci in range(len(cand_mid_coocurmid_i)):
        mid,i_coocurmid = cand_mid_coocurmid_i[ci]
#        print mid
#        print "''''''''''''''''''''''''''''"
#        print i_coocurmid
#        exit(0)
        nameI = None
        if mid in mid2Name:
          nameI = mid2Name.get(mid)
          
        hasRel = 0
        hasNGD = 0
        
        nums = 0
        for jcand in range(len(doc_ents_cand_mid_dict)):
          if icand !=jcand:
            cand_mid_coocurmid_j = doc_ents_cand_mid_dict[icand]
            for cj in range(len(cand_mid_coocurmid_j)):
              midj,j_coocurmid = cand_mid_coocurmid_j[cj]
#              print 'tyep cooccurmid:',type(i_coocurmid),type(j_coocurmid)
#              exit(0)
              
              if mid in j_coocurmid or midj in i_coocurmid:
                hasRel +=1
              if midj in mid2Name:
                nameJ = mid2Name.get(midj)
                if nameI is not None and nameJ is not None:
                  hasNGD += ngd.get_page_links(nameI,nameJ)
                  nums += 1
#        if hasRel !=0:
#          print hasRel*1.0/len(i_coocurmid)
#          temprelCoherent[ci] = hasRel*1.0/len(i_coocurmid)
        if hasNGD!=0:
          temprelCoherent[ci] += hasNGD/nums
      doc_temprelCoherent.append(temprelCoherent)
    ent_mention_relCoherent_feature.append(doc_temprelCoherent)
                       
                 
        
#    doc_temprelCoherent=[]
#    for icand in range(len(doc_ents_cand_mid_dict)):
#      print 'ents:',ents[i]
#      exit(0)
#      temprelCoherent = np.zeros((30,))
#      i_coocurmid=doc_ents_cand_mid_dict[icand]
#      print 'len i_coocurmid:',len(i_coocurmid)
#      for ci in range(len(i_coocurmid)):
#        midi=i_coocurmid[ci]
#        hasRel = 0
#        for jcand in range(len(doc_ents_cand_mid_dict)):
#          j_coocurmid=doc_ents_cand_mid_dict[jcand]
#          if icand!=jcand:
#            for midj in j_coocurmid:
#              if len(midi&midj)!=0:
#                hasRel +=1
#        temprelCoherent[ci] = hasRel
#      doc_temprelCoherent.append(temprelCoherent)
#    ent_mention_relCoherent_feature.append(doc_temprelCoherent)
  #print ent_mention_relCoherent_feature
  #print len(ent_mention_relCoherent_feature),len(ent_mention_relCoherent_feature[3]),len(ent_mention_relCoherent_feature[3][0])
  cPickle.dump(ent_mention_relCoherent_feature,open(dir_path+'features/'+data_tag+'_ent_relcoherent.p','wb'))

def cosineSimilarity(mention,cand_mid):
  cand_ents = []
  #print 'comput entity mention...'
  if cand_mid in fid2title:
    for item in fid2title[cand_mid]:    
      twordv = np.zeros((100,))
      ent_words = processDescription(item)
      for word in ent_words:
        if word in wordsVectors:
          twordv += wordsVectors[word]
      cand_ents.append(twordv) 
    
  twordv = np.zeros((100,))
  ent_words = processDescription(mention)
  for word in ent_words:
    if word in wordsVectors:
      twordv += wordsVectors[word]
  twordv = np.reshape(twordv,(1,-1))
  max_similar=0
  ret=np.zeros((100,))
  ikey = 0
  for key in cand_ents:
    tempkey = np.reshape(key,(1,-1))
    cosSim = cosine_similarity(tempkey,twordv)[0][0] + Levenshtein.ratio(mention.lower(),fid2title[cand_mid][ikey].lower())
    ikey+=1
    if cosSim > max_similar:
      max_similar = cosSim
      ret = key
  return ret

def get_candidate_ent_features():

  descrip_lent=[]
  
  ent_mention_surface_wordv=[]
  ent_mention_cand_prob_feature=[]
  
  ent_mention_index=[]
  ent_mention_link_feature=[] #shape: [line,ent_mention_num, 30(candidates number) * 100(dimension)]
  ent_mention_tag = []  #shape:[line,ent_mention_num,30]
  ent_mention_type_feature=[]   #shape:[line,ent_mention_num,30* 113(one hot type dimension)]
  rightCandHasNoDescripNums = 0
  
  k  = -1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(range(100)):
    aNosNo = sentId2aNosNo[i]
    
    ents = ent_Mentions[i]
    temps_mentwordv=[]
    temps_cand_prob=[]
    
    temps =[]
    temps_type=[]
    temps_tag = []
    
    temps_ent_index=[]
    
    #print i,'\tentM:',len(ents)
    for j in range(len(ents)):
      enti = ents[j][2]
      sindex = ents[j][0];eindex = ents[j][1]
      key = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)
      
      linkTag = ent_Mentions_link_tag[key]   #get entity mention linking mid!
      enti_name = enti.lower()
      #print ents[j]
      k += 1  
      #tmentwordv=[]
      tcandprob = []
      
      ent_mention_tag_temp = np.zeros((30,))
      tdescip = [];tcanditetype=[];
      
      cand_mid_dict = all_candidate_mids[k]
      imidNo = -1
      for mid in cand_mid_dict: 
        imidNo += 1
#        if mid not in mid2description:
#          print 'mid not in mid2description', mid
          #exit(-1)
        #tmentv = cosineSimilarity(enti_name,mid)
        
        if mid in linkTag:
          if mid not in mid2description:
            rightCandHasNoDescripNums+=1
            print 'right candidate entity mid not in mid2description', mid
          ent_mention_tag_temp[imidNo] = 1
        
        twordv = np.zeros((100,)) 
        ttypev = np.zeros((113,))
        if mid in mid2figer:
          for midtype in mid2figer[mid]:
            ttypev[midtype]=1
        
        if mid in mid2description:           
          line = mid2description[mid]
          descript = processDescription(line)
          descrip_lent.append(len(descript))
          #print descript
#          for i in range(min(15,len(descript))):  //abandon methods
#            word = descript[i]
#            #print 'word:',word
#            if word in descript_Words:
#              twordv += descript_Words[word]
          #@2017/1/25 position encoding(PE)
          qlent = 0
          tempWordEmbed=[]
          for idescrip in range(min(15,len(descript))):
            word = descript[idescrip]
            if word in wordsVectors:
              qlent +=1
              tempWordEmbed.append(wordsVectors[word])
              
          for idescrip in range(qlent):
            li =[]
            for jdescrip in range(100):  #100 stands for embedding dimension
              li.append(min((idescrip+1)*100/((jdescrip+1)*qlent),((jdescrip+1)*qlent)/((idescrip+1)*100)))
            twordv += tempWordEmbed[idescrip] * np.asarray(li)
        
        
        '''
        @add candidate entity type features!
        '''
        
        if len(tcanditetype)==0:
          #tcanditetype = ttypev   #may cause a lot of problem
          tcanditetype = np.array(ttypev)
        else:
          tcanditetype = np.concatenate((tcanditetype,ttypev))
         
        if len(tdescip)==0:
          tdescip = np.array(twordv)  #gurantee to copy the data, not refer to the same array
        else:
          tdescip = np.concatenate((tdescip,twordv))
        
        #temp_cand_prob_mid = []
        #for i_temp_cand_mid in range(len(cand_mid_dict[mid])):
        #  if cand_mid_dict[mid][i_temp_cand_mid] >0:
        #    temp_cand_prob_mid.append(1)
        #  else:
        #    temp_cand_prob_mid.append(0)
        if len(tcandprob)==0:
          tcandprob = np.array(cand_mid_dict[mid])
          #tcandprob = np.asarray(temp_cand_prob_mid)
        else:
          tcandprob = np.concatenate((tcandprob,np.array(cand_mid_dict[mid])))
          #tcandprob = np.concatenate((tcandprob,np.asarray(temp_cand_prob_mid))) 
#        if len(tmentwordv)==0:
#          tmentwordv = np.array(tmentv)
#        else:
#          tmentwordv = np.concatenate((tmentwordv,tmentv))
      assert np.shape(tcanditetype)[0]/113 == np.shape(tdescip)[0]/100
      assert np.shape(tcandprob)[0]/3 == np.shape(tcanditetype)[0]/113
      #temps_mentwordv.append(tmentwordv)
      temps_cand_prob.append(tcandprob)
      temps_type.append(tcanditetype)
      temps.append(tdescip)
      temps_tag.append(ent_mention_tag_temp)
      
      temps_ent_index.append((ents[j][0],ents[j][1]))
      
    #ent_mention_surface_wordv.append(temps_mentwordv)
    ent_mention_cand_prob_feature.append(temps_cand_prob)
    
    ent_mention_type_feature.append(temps_type)
    
    ent_mention_link_feature.append(temps)
    ent_mention_tag.append(temps_tag)
    ent_mention_index.append(temps_ent_index)  
    
  
#  print len(ent_mention_link_feature),len(ent_mention_link_feature[3]),len(ent_mention_link_feature[3][0])
#  print len(ent_mention_tag),len(ent_mention_tag[3]),len(ent_mention_tag[3][0])
#  print len(ent_mention_index),len(ent_mention_index[3]),len(ent_mention_index[3][0])
#  print  max(descrip_lent),min(descrip_lent),sum(descrip_lent) / float(len(descrip_lent))
  param_dict = {'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag}
  cPickle.dump(param_dict,open(dir_path+"features/"+data_tag+"_ent_linking.p",'wb'))
 # print len(ent_mention_type_feature),len(ent_mention_type_feature[3]),len(ent_mention_type_feature[3][0])
  cPickle.dump(ent_mention_type_feature,open(dir_path+"features/"+data_tag+"_ent_linking_type.p",'wb'))
  
  cPickle.dump(ent_mention_cand_prob_feature,open(dir_path+"features/"+data_tag+"_ent_linking_candprob.p",'wb'))     
  
  #cPickle.dump(ent_mention_surface_wordv,open(dir_path+"features/"+data_tag+"_ent_mentwordv.p",'wb'))         
            


parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
data_args = parser.parse_args()
data_tag = data_args.data_tag
dir_path = data_args.dir_path

start_time = time.time()
ent_Mentions = cPickle.load(open(dir_path+'features/ent_mention_index.p'))
ent_Mentions_link_tag = cPickle.load(open(dir_path+data_tag+'_entMentsTags.p')) #aNosNo \t start_index \t end_index
sentid = 0
sentId2aNosNo = {}
with codecs.open(dir_path+'sentid2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    sentId2aNosNo[sentid] = line
    sentid += 1
    
print 'load ent_Mentions cost time:',time.time()-start_time

#nilents = 0                                              
#for i in range(len(ent_Mentions)):
#  #for i in tqdm(range(100)):
#    aNosNo = sentId2aNosNo[i]
#    ents = ent_Mentions[i]
#    
#    #print i,'\tentM:',len(ents)
#    for j in range(len(ents)):
#      enti = ents[j][2]
#      sindex = ents[j][0];eindex = ents[j][1]
#      key = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)
#      
#      linkTag = ent_Mentions_link_tag[key]   #get entity mention linking mid!
#      if linkTag=='NIL':
#        nilents+=1
#        print enti, linkTag
#print 'all NIL Ents:',nilents
#print 'all ents:',len(ent_Mentions_link_tag)
'''
Step1: get the complete candidates entities..
'''

f_input = dir_path+"features/"+data_tag+"_candEnts.p"
f_output = dir_path+"features/"+data_tag+"_ent_cand_mid.p"

data = cPickle.load(open(f_input,'r'))
entstr2id_org = data['entstr2id']
print 'entstr2id_org',len(entstr2id_org)
id2entstr_org = {value:key for key,value in entstr2id_org.items()}
entstr2id= collections.defaultdict(set)
for key,value in entstr2id_org.items():
  entstr2id[key.lower()].add(value)
print 'entstr2id',len(entstr2id)

docId_entstr2id= collections.defaultdict(dict)
'''
@2017/3/1, we revise the entstr2id  to docid_entstr2id
'''
for i in tqdm(range(len(ent_Mentions))):
  aNosNo = sentId2aNosNo[i]
  docId = aNosNo.split('_')[0]
  ents = ent_Mentions[i]
  
  for j in range(len(ents)):
    enti = ents[j]
    #print enti
    enti_name = enti[2]
    value = entstr2id_org.get(enti_name)
    if enti_name.lower() not in docId_entstr2id[docId]:
      
      docId_entstr2id[docId][enti_name]= {value}
    else:
      docId_entstr2id[docId][enti_name].add(value)

candiate_ent = data['candiate_ent']#;candiate_coCurrEnts = data['candiate_coCurrEnts']
mid2name = getmid2Name()
title2pageId = getfname2pageid()
w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
wikititle_reverse_index  = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitleReverseIndex.p','rb'))
#print wikititle_reverse_index
print 'start to solve problems...'
w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
all_candidate_mids = get_final_ent_cands()

cPickle.dump(all_candidate_mids,open(f_output,'wb'))


'''
Step2: @generate all candidate mid cocurrent features!
'''
all_candidate_mids = cPickle.load(open(dir_path+"features/"+data_tag+"_ent_cand_mid.p"))
#print all_candidate_mids
get_all_candidate_mid_cocurrent()

'''
Step3: generate rel cocurrent features...
'''
mid2Name = getmid2Name()
all_candidate_mids = cPickle.load(open(dir_path+"features/"+data_tag+"_ent_cand_mid.p"))
allcandents_coents = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_relcoherent.ptemp','rb'))   
get_candidate_rel_features()

'''
Step4 other features...
'''
'''
#@time: 2017/2/8
#@function: load re-rank type
##'''
all_candidate_mids = cPickle.load(open(dir_path+"features/"+data_tag+"_ent_cand_mid.p"))
print 'load fid2title...'
fid2title = collections.defaultdict(list)
fid2vec = collections.defaultdict(list)
wtitle2fid = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
for key in tqdm(wtitle2fid):
  for item in wtitle2fid[key]:
    fid2title[item].append(key)

'''
@time: 2016/12/23
@function: load re-rank featureÂ£Âºentity type
'''
stime = time.time()
descript_Words = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wordvec_model_100.p', 'rb'))
print 'load wordvec_model_100 cost time: ', time.time()-stime

'''
compute entity vector===>adpot the type vector is more feasible!
'''
wordsVectors = descript_Words.wvec_model
        
stime = time.time()
mid2figer = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/mid2figer.p','rb'))
print 'load mid2figer cost time:',time.time()-stime


stime = time.time()
mid2description={}  #nearly 2.3G
with codecs.open('/home/wjs/demo/entityType/informationExtract/data/mid2description.txt','r','utf-8') as file:
  for line in tqdm(file):
    items = line.strip().split('\t')
    if len(items) >=2:
      mid2description[items[0]] =items[1]
print 'load mid2descriptioon cost time: ', time.time()-stime

get_candidate_ent_features()
