# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
@editor: wujs
@function: to generate the final candidate
'''

import os
import sys
import math
import time
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
from NGDUtils import NGDUtils
import numpy as np

def getfname2pageid():
  title2pageId = {}
  with open('data/name2pageId.txt','r') as file:
    for line in tqdm(file):
      line = line.strip().split('\t')
      pageId = line[0]; title = line[1]
      title2pageId[title] = pageId       
  return title2pageId

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

def is_contain_ents(enti,entj):
  enti = enti.lower()
  entj = entj.lower()
  if enti in entj:
    return True
  else:
    return False

def get_freebase_ent_cands(ngd,mid2name,cantent_mid2,enti,context_ent_pageId,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum):

  #print 'go into entNGD...'
  distRet = {};
  #first find all the ents we need to process
  enti_title = enti.lower()
  enti_item = enti_title.split(u' ')
  enti_f = enti_item[0]
  totaldict=dict()
  '''
  @exits a very 
  '''
  #print context_ent_pageId
  if enti_title in wikititle_reverse_index:
    totaldict = wikititle_reverse_index[enti_title]
  else:
    if enti_f in wikititle_reverse_index:
      totaldict = wikititle_reverse_index[enti_f]
#  enti_list = ngd.getLinkedEnts(enti)
  ids_key=0
  for key in totaldict:
    if is_contain_ents(enti_title,key):
      addScore = 0
      if enti_title == key or key in entstr2id:  #completely match, score is too low!
        for wmid in wikititle2fb[key]:
          if wmid not in cantent_mid2:
            freebaseNum-=1
            cantent_mid2[wmid] = list([0,0,len(context_ent_pageId)])
            freebaseNum -=1
          else:
            temp = list(cantent_mid2[wmid])
            temp[2]= len(context_ent_pageId)
            cantent_mid2[wmid]= list(temp)
      else:
        for wmid in wikititle2fb[key]:
          #need to re-rank using NGD ...
          wmid_set = ngd.getLinkedEnts(mid2name[wmid])
          
          
          if len(context_ent_pageId&wmid_set) !=0:
            addScore += len(context_ent_pageId&wmid_set)
            #print 'addScore:',addScore
          if ids_key%1000==0:
            print ids_key,len(totaldict),addScore
          ids_key += 1
          '''
          exits the bottle_neck, however I have no idea to deal with it!
          '''
          distRet[wmid+u'\t'+key]= addScore
  distRet= sorted(distRet.iteritems(), key=lambda d:d[1], reverse = True)
  
  #cantent_mid={}
  #freebaseNum=50
  for item in distRet:
    if freebaseNum==0:
      break
    item_it = item[0].split(u'\t')
    wmid = item_it[0]
    if wmid not in cantent_mid2:
      #cantent_mid2[item_it[0]] = item_it[1]
      #cantent_mid2[wmid] = [0,0,item[1]]
      cantent_mid2[wmid] = list([0,0,item[1]])
      freebaseNum -=1
    else:
      temp = list(cantent_mid2[wmid])
      temp[2]=temp[2]+item[1]
      cantent_mid2[wmid]= list(temp)
  #print cantent_mid2
  return cantent_mid2

def get_cantent_mid(listentcs,w2fb,wikititle2fb):
  cantent_mid={}
  mid_index = 0.0
  for cent in listentcs:
    mid_index +=1.0
    if isinstance(cent,dict):
      ids = cent[u'ids']
      titles = cent[u'title'].lower()
    else:
      ids = cent
      titles = cent.lower()
    if ids in w2fb:
      #cantent_mid[w2fb[ids]] = titles
      cantent_mid[w2fb[ids]] = list([1/mid_index,0,0])
    elif titles in wikititle2fb:
      for wmid in wikititle2fb[titles]:
        #cantent_mid[wmid] =titles
        cantent_mid[wmid] =list([1/mid_index,0,0])
  return cantent_mid

def get_ent_word2vec_cands(enti,w2fb,wikititle2fb,w2vModel,entstr2id,candiate_ent,cantent_mid1):
  entiw = enti.replace(u' ',u'_')
  if entiw in w2vModel:
    coherent_ents = w2vModel.most_similar(entiw,topn=10) #convert to gensim style
    k=1
    for citems in coherent_ents:
      cents = citems[0].replace(u'_',u' ')  #convert to freebase style
      if cents.lower() in entstr2id:
        entids = entstr2id[cents.lower()]
        listentcs =[]
        for entid in entids:
          listentcs += candiate_ent[entid][0:1]  #very important things!
        if len(listentcs)>=1:
          #cantent_mid =dict(cantent_mid,**get_cantent_mid(listentcs,w2fb,wikititle2fb))
          for wmid in get_cantent_mid(listentcs,w2fb,wikititle2fb):
            if wmid not in cantent_mid1:
              #cantent_mid1[wmid] = [0,1/k,0]
              cantent_mid1[wmid] = list([0,citems[1],0])
            else:
              temp = list(cantent_mid1[wmid])
              #temp[1]=temp[1]+1/k
              temp[1]=citems[1]
              cantent_mid1[wmid]= list(temp)
      if cents.lower() in wikititle2fb:
        for wmid in wikititle2fb[cents.lower()]:
          #cantent_mid[wmid] = cents.lower()
          if wmid not in cantent_mid1:
            cantent_mid1[wmid] = list([0,citems[1],0])
          else:
            temp = list(cantent_mid1[wmid])
            temp[1]= temp[1]+citems[1]
            cantent_mid1[wmid]= list(temp)
      k += 1
      
  return cantent_mid1

def get_final_ent_cands():
  all_candidate_mids = []
  allentmention_numbers = 0
  '''
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
  '''
  ent_ment_link_tags = cPickle.load(open('data/aida/aida-annotation.p_new','rb'))
  if data_flag=='train':
    ent_id = 0
  if data_flag=='testa':
    ent_id = 23396
  if data_flag=='testb':
    ent_id = 29313
  print 'finish load all datas'
  #exit(-1)
  right_nums = 0;wrong_nums =0
  pass_nums = 0
  totalRepCand = 0
  for i in tqdm(range(len(ent_Mentions))):
    aNosNo = id2aNosNo[i]
    docId = aNosNo.split('_')[0]
    ents = ent_Mentions[i]
    context_ents = docId_entstr2id[docId]
    context_ent_pageId = set()
    for key in context_ents:
      #print 'context ents:',context_ents
      if key in title2pageId:
        context_ent_pageId.add(title2pageId[key])
        
    for j in range(len(ents)):
      isRepflag =False 
      
      allentmention_numbers+=1
      
      
      enti = ents[j]
      
      enti_name = enti.content.lower()
      #mention = enti.content
      startI = enti.startIndex; endI = enti.endIndex
      aNosNoMSE = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      '''
      if enti_name not in entstr_lower2mid:
        pass_nums = pass_nums + 1
        continue
      else:
        tag = entstr_lower2mid[enti_name]
      '''
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]
      
      if tag == 'NIL':
        pass_nums = pass_nums + 1
        ent_id += 1
        continue
      ent_id += 1
      if aNosNoMSE in entMent2repMent:
        isRepflag = True
        enti_name = entMent2repMent[aNosNoMSE].split('\t')[-1].lower()
        #print 'step into the entMent2repMent'
      
#      if enti_name != 'wall street':
#        continue
      entids = entstr2id[enti_name]
      
      
      listentcs = []
      for entid in entids:
        for entid_mid in candiate_ent[entid]:
          if entid_mid not in listentcs:
            listentcs.append(entid_mid)
            
      cantent_mid1 = get_cantent_mid(listentcs,w2fb,wikititle2fb)   #get wikidata&dbpedia search candidates
      
      if enti_name in wikititle2fb:
        wmid_i = 0
        for wmid in wikititle2fb[enti_name]:
          wmid_i +=1
          #cantent_mid1[wmid] = enti_name
          cantent_mid1[wmid] = [1/wmid_i,0,0]
  
      cantent_mid2 = get_ent_word2vec_cands(enti.content,w2fb,wikititle2fb,w2vModel,context_ents,candiate_ent,cantent_mid1) #get word2vec coherent candidates
      #cantent_mid2 = get_ent_word2vec_cands(enti.content,w2fb,wikititle2fb,w2vModel,context_ents,candiate_ent,cantent_mid1) #get word2vec coherent candidates
      
      freebaseNum = max(0,30 - len(cantent_mid2))
      final_mid = get_freebase_ent_cands(ngd,mid2name,cantent_mid2,enti.content,context_ent_pageId,context_ents,wikititle2fb,wikititle_reverse_index,freebaseNum)
      #cantent_mid3 = get_freebase_ent_cands(cantent_mid2,enti.content,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum) #search by freebase matching
      
      
      #final_mid = list(cantent_mid2)
      totalCand = len(final_mid)
      #if tag in cantent_mid1 or tag in cantent_mid2 or tag in cantent_mid3:
      if tag in final_mid:
        if isRepflag:
          totalRepCand += 1
        right_nums += 1
      else:
        wrong_nums = wrong_nums + 1
        print 'wrong:',tag,enti.content,totalCand#,final_mid
        #print cantent_mid1,cantent_mid2,cantent_mid3
        #exit()
        
      all_candidate_mids.append(final_mid)
  
  print 'wrong_nums:',wrong_nums
  print 'right_nums:',right_nums
  print 'pass_nums:',pass_nums
  print len(all_candidate_mids), allentmention_numbers
  print 'totalRep right:',totalRepCand
  return all_candidate_mids    



if __name__=='__main__':
  if len(sys.argv) !=6:
    print 'usage: python pyfile dir_path inputfile train_entms.p100(test) train_ent_cand_mid.p flag'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  +'/process/'+ sys.argv[2]
  f_input_entMents = dir_path  +'/features/'+ sys.argv[3]
  f_output = dir_path  +'/features/'+ sys.argv[4] #record the total entity nums.
  data_flag = sys.argv[5]
  #data context:  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
  data = cPickle.load(open(f_input,'r'))
  entstr2id_org = data['entstr2id']
  #print entstr2id_org
  candiate_ent = data['candiate_ent']#;candiate_coCurrEnts = data['candiate_coCurrEnts']
  print 'entstr2id_org',len(entstr2id_org)
  id2entstr_org = {value:key for key,value in entstr2id_org.items()}
  entstr2id= collections.defaultdict(set)
  for key,value in entstr2id_org.items():
    entstr2id[key.lower()].add(value)
  print 'entstr2id',len(entstr2id)
  mid2name = getmid2Name()
  ngd = NGDUtils()
  w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
  wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
  ids = entstr2id['wall street']
#  for idi in ids:
#    print idi
#    for kk in candiate_ent[idi]:
#      print kk
#      print w2fb[kk['ids']],kk['title']
#  exit(0)
  averages = []
  max_candidate = 0
  new_candiate_ent=[]
  for key in candiate_ent:
    #print key
    temnum = 0
    new_item = []
    for item in key:
      if item in w2fb:
        new_item.append(item)
        temnum += 1
      else:
        if item.lower() in wikititle2fb:
          new_item.append(item)
          temnum += 1
      if len(new_item) ==10:
        break
    new_candiate_ent.append(new_item)
    
    averages.append(temnum)
    if max_candidate < temnum:
      max_candidate = temnum
  print max_candidate
  print np.average(averages)
  
  entMent2repMent_org = cPickle.load(open(dir_path+'process/'+data_flag+'_entMent2repMent.p','rb'))  
  #print entMent2repMent_org
  
  print len(entMent2repMent_org)
  
  entMent2repMent = {}
  for key in entMent2repMent_org:
    keyr = '\t'.join(key.split('\t')[0:3])
    val = entMent2repMent_org[key].split('\t')[-1]
  
    entMent2repMent[keyr] = val
             
  data = cPickle.load(open(dir_path+'process/'+ data_flag+'.p','r'))
  
  aNosNo2id = data['aNosNo2id']
  id2aNosNo = {val:key for key,val in aNosNo2id.items()}
  
  
  #print candiate_ent
  
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_entMents,'r'))
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents'];ent_ctxs=dataEnts['ent_ctxs']
  print len(ent_Mentions)
  #print ent_Mentions
  #exit(0)
  all_ents = set()
  
  docId_entstr2id= collections.defaultdict(dict)
  '''
  @2017/3/1, we revise the entstr2id  to docid_entstr2id
  '''
  for i in tqdm(range(len(ent_Mentions))):
    aNosNo = id2aNosNo[i]
    docId = aNosNo.split('_')[0]
    ents = ent_Mentions[i]
    
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content
      value = entstr2id_org.get(enti_name)
      if enti_name.lower() not in docId_entstr2id[docId]:
        
        docId_entstr2id[docId][enti_name]= {value}
      else:
        docId_entstr2id[docId][enti_name].add(value)
  #print docId_entstr2id
  print 'start to load wikititle...'
  s_time = time.time()
  #w2vModel=[]
  #wikititle_reverse_index=[]
  w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
  print 'w2vModel:',time.time()-s_time
  wikititle_reverse_index  = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitleReverseIndex.p','rb'))
  print 'wikititle_reverse_index:',time.time()-s_time
  #w2vModel=[]
  #wikititle2fb=[]
  #wikititle_reverse_index=[]
  
  print 'start to solve problems...'
  #
  title2pageId = getfname2pageid()
  all_candidate_mids = get_final_ent_cands()
  cPickle.dump(all_candidate_mids,open(f_output,'wb'))
  
  