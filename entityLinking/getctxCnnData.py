# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
@editor: wujs
@function: to generate the final candidate
@revise: 2017/3/20 revise all the score! it may help us to score the results!
'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import math
import time
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import cPickle
from utils import getWikititle2Freebase,getWikidata2Freebase,getreverseIndex
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
      title2pageId[title.lower()] = pageId    
      #print title.lower()   
  return title2pageId

def getmid2Name():
  mid2name = collections.defaultdict(list) 
  #mid2name = {}
  '''
  @we ignore that a mid may have several names!
  '''
  with open('data/mid2name.tsv','r') as file:
    for line in tqdm(file):
      line =line.strip()
      #print line.split(u'\t')
      items= line.split('\t')
      
      if len(items)>=2:
        mid = items[0]; name = ' '.join(items[1:])
        mid2name[mid].append(name)
      else:
        items = line.split(' ')
        if len(items)>=2:
          mid = items[0]; name = ' '.join(items[1:])
          mid2name[mid].append(name)
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

def getLinkedEntsFmid(ngd,mid2name,fmid):
  wmid_set = set()
  if fmid in mid2name:
    for title in mid2name[fmid]:
      wmid_set = wmid_set | ngd.getLinkedEnts(title)
  return wmid_set
  

def get_freebase_ent_cands(mid2incomingLinks,ngd,mid2name,cantent_mid2,enti,context_ent_pageId,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum,word2vecEnts):
  #print 'go into entNGD...'
  distRet = {}
  #first find all the ents we need to process
  enti_title = enti.lower()
  enti_item = enti_title.split(' ')
  enti_f = enti_item[0]
  totaldict=dict()
  '''
  @need to revise the scores
  '''
  #print context_ent_pageId
  if enti_title in wikititle_reverse_index:
    totaldict = wikititle_reverse_index[enti_title]
  elif enti_title.replace(' ','') in wikititle_reverse_index:
    totaldict = wikititle_reverse_index[enti_title.replace(' ','')]
#  else:
#    if enti_f in wikititle_reverse_index:
#      totaldict = wikititle_reverse_index[enti_f]
#  enti_list = ngd.getLinkedEnts(enti)
  freebase_cants = collections.defaultdict(int)
  ids_key=0
  for key in totaldict:
    if is_contain_ents(enti_title,key) or is_contain_ents(enti_title.replace(' ',''),key):
      addScore = 0
      if enti_title == key or enti_title.replace(' ','')== key or key in entstr2id:  #completely match, score is too low!
        for wmid in wikititle2fb[key]:
          freebase_cants[wmid] = 1
          if wmid not in cantent_mid2:
            tempScore = 0
            if wmid in mid2name:
              if wmid in mid2incomingLinks:
                tempScore = len(mid2incomingLinks[wmid])
              else:
                wmid_set = getLinkedEntsFmid(ngd,mid2name,wmid)
                mid2incomingLinks[wmid] = wmid_set
                tempScore = len(wmid_set)
            item2=0
            if key.lower() in word2vecEnts:
              item2 = word2vecEnts[key.lower()]
            cantent_mid2[wmid] = list([tempScore,item2,len(context_ent_pageId)])
          else:
            temp = list(cantent_mid2[wmid])
            temp[2]= len(context_ent_pageId)
            cantent_mid2[wmid]= list(temp)
      else:
        for wmid in wikititle2fb[key]:
          freebase_cants[wmid] = 1
          #need to re-rank using NGD ...
          wmid_set=set()
          if wmid in mid2name:
            if wmid in mid2incomingLinks:
              wmid_set = mid2incomingLinks[wmid]
            else:
              wmid_set = getLinkedEntsFmid(ngd,mid2name,wmid)
              mid2incomingLinks[wmid] = wmid_set
            
          if len(context_ent_pageId&wmid_set) !=0:
            addScore += len(context_ent_pageId&wmid_set)
            #print 'addScore:',addScore
          if ids_key%1000==0:
            print ids_key,len(totaldict),addScore
          ids_key += 1
          '''
          exits the bottle_neck, however I have no idea to deal with it!
          '''
          if addScore >= 0:
            distRet[wmid+'\t'+key]= addScore
  distRet= sorted(distRet.iteritems(), key=lambda d:d[1], reverse = True)
  
  
  for item in distRet:
    #if freebaseNum==0:
    #  break
    item_it = item[0].split('\t')
    wmid = item_it[0]
    if wmid not in cantent_mid2:
      tempScore=0
      if wmid in mid2name:
        if wmid in mid2incomingLinks:
          wmid_set = mid2incomingLinks[wmid]
        else:
          wmid_set = getLinkedEntsFmid(ngd,mid2name,wmid)
          mid2incomingLinks[wmid] = wmid_set
        tempScore=len(wmid_set)
      item2=0
      if key.lower() in word2vecEnts:
        item2 = word2vecEnts[key.lower()]
      #cantent_mid2[item_it[0]] = item_it[1]
      #cantent_mid2[wmid] = [0,0,item[1]]
      cantent_mid2[wmid] = list([tempScore,item2,item[1]])
      #freebaseNum -=1
    else:
      temp = list(cantent_mid2[wmid])
      temp[2]=temp[2]+item[1]
      cantent_mid2[wmid]= list(temp)
  #print cantent_mid2
  return cantent_mid2,freebase_cants

'''
@we should score as the number of inconming links!!! ==>reasonable
'''
def get_cantent_mid(mid2incomingLinks,ngd,listentcs,w2fb,wikititle2fb,mid2name):
  cantent_mid={}
  mid_index = 0.0
  for cent in listentcs:
    mid_index +=1.0
    if isinstance(cent,dict):
      ids = cent['ids']
      titles = cent['title'].lower()
    else:
      ids = cent
      titles = cent.lower()
    if ids in w2fb:
      tempScore=0
      wmid = w2fb[ids]
      if w2fb[ids] in mid2name:
        if wmid in mid2incomingLinks:
          wmid_set = mid2incomingLinks[wmid]
        else:
          wmid_set = getLinkedEntsFmid(ngd,mid2name,wmid)
          mid2incomingLinks[wmid] = wmid_set
        tempScore = len(wmid_set)
      #cantent_mid[w2fb[ids]] = titles
      cantent_mid[w2fb[ids]] = list([tempScore,0,0])
    elif titles in wikititle2fb:
      for wmid in wikititle2fb[titles]:
        tempScore = 0
        if wmid in mid2name:
          if wmid in mid2incomingLinks:
            wmid_set = mid2incomingLinks[wmid]
          else:
            wmid_set = getLinkedEntsFmid(ngd,mid2name,wmid)
            mid2incomingLinks[wmid] = wmid_set
          tempScore = len(wmid_set)
        #cantent_mid[wmid] =titles
        cantent_mid[wmid] =list([tempScore,0,0])
  return cantent_mid

def get_ent_word2vec_cands(mid2incomingLinks,ngd,mid2name,enti,w2fb,wikititle2fb,w2vModel,entstr2id,candiate_ent,cantent_mid1):
  entiw = enti.replace(' ','_')
  word2vecEnts = {}
  if entiw in w2vModel:
    coherent_ents = w2vModel.most_similar(entiw,topn=20) #convert to gensim style
    k=1
    for citems in coherent_ents:
      if citems[1] >=0.5:
        cents = citems[0].replace('_',' ')  #convert to freebase style
        if cents.lower() in entstr2id:
          entids = entstr2id[cents.lower()]
          listentcs =[]
          for entid in entids:
            listentcs += candiate_ent[entid][0:1]  #very important things!
          if len(listentcs)>=1:
            #cantent_mid =dict(cantent_mid,**get_cantent_mid(listentcs,w2fb,wikititle2fb))
            cantent_mid = get_cantent_mid(mid2incomingLinks,ngd,listentcs,w2fb,wikititle2fb,mid2name)
            for wmid in cantent_mid:
              if wmid not in cantent_mid1:
                '''
                @time: 2017/3/20
                we need to add the numbers of incoming links!
                '''
                temps = list(cantent_mid[wmid])
                temps[1] = citems[1]
                cantent_mid1[wmid] = list(temps)
              else:
                temp = list(cantent_mid1[wmid])
                #temp[1]=temp[1]+1/k
                temp[1]=citems[1]
                cantent_mid1[wmid]= list(temp)
        '''
        @revise: time:2017/3/23 这个地方会出现很大的漏洞，我感觉呢！
        '''
#        if cents.lower() in wikititle2fb:
#          for wmid in wikititle2fb[cents.lower()]:
#            #cantent_mid[wmid] = cents.lower()
#            if wmid not in cantent_mid1:
#              wmid_set = ngd.getLinkedEnts(wmid)
#              cantent_mid1[wmid] = list([len(wmid_set),citems[1],0])
#            else:
#              temp = list(cantent_mid1[wmid])
#              temp[1]= citems[1]
#              cantent_mid1[wmid]= list(temp)
        if cents.lower() not in entstr2id:
          word2vecEnts[cents.lower()] = citems[1]
        k += 1
      
  return cantent_mid1,word2vecEnts


def getTop1Wikipages(enti_name):
  entids = entstr2id[enti_name]

  listentcs = []
  for entid in entids:
    for entid_mid in candiate_ent[entid]:
      if entid_mid not in listentcs:
        listentcs.append(entid_mid)
        
def get_final_ent_cands():
  all_candidate_mids = []
  wiki_candidate_mids=[]
  freebase_candidate_mids=[]
  
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
      '''
      @利用top1的page吧！
      '''
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
#        ent_id += 1
#        continue
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
            
      cantent_mid1 = get_cantent_mid(mid2incomingLinks,ngd,listentcs,w2fb,wikititle2fb,mid2name)   #get wikidata&dbpedia search candidates
      
#      if enti_name in wikititle2fb:
#        wmid_i = 0
#        for wmid in wikititle2fb[enti_name]:
#          wmid_i +=1
#          #cantent_mid1[wmid] = enti_name
#          cantent_mid1[wmid] = [1/wmid_i,0,0]
  
      cantent_mid2,word2vecEnts =  get_ent_word2vec_cands(mid2incomingLinks,ngd,mid2name,enti.content,w2fb,wikititle2fb,w2vModel,context_ents,candiate_ent,dict(cantent_mid1)) #get word2vec coherent candidates
      #cantent_mid2 = get_ent_word2vec_cands(enti.content,w2fb,wikititle2fb,w2vModel,context_ents,candiate_ent,cantent_mid1) #get word2vec coherent candidates
      freebaseNum = max(0,90 - len(cantent_mid2))
      final_mid,freebase_cants = get_freebase_ent_cands(mid2incomingLinks,ngd,mid2name,dict(cantent_mid2),enti.content,context_ent_pageId,context_ents,wikititle2fb,wikititle_reverse_index,freebaseNum,word2vecEnts)
      #cantent_mid3 = get_freebase_ent_cands(cantent_mid2,enti.content,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum) #search by freebase matching
      
      
      #final_mid = list(cantent_mid2)
      #totalCand = len(final_mid)
      #if tag in cantent_mid1 or tag in cantent_mid2 or tag in cantent_mid3:
      if tag in final_mid:
        if isRepflag:
          totalRepCand += 1
        right_nums += 1
      else:
        wrong_nums = wrong_nums + 1
        #print 'wrong:',tag,enti.content,totalCand#,final_mid
        #print cantent_mid1,cantent_mid2,cantent_mid3
        #exit()
        
      all_candidate_mids.append(dict(final_mid))
      wiki_candidate_mids.append(dict(cantent_mid1).keys())
      freebase_candidate_mids.append(dict(freebase_cants).keys())
  print '--------------'
  print 'data tag:',data_flag
  print 'wrong_nums:',wrong_nums
  print 'right_nums:',right_nums
  print 'pass_nums:',pass_nums
  print len(all_candidate_mids), allentmention_numbers
  print 'totalRep right:',totalRepCand
  return all_candidate_mids,wiki_candidate_mids,freebase_candidate_mids



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
  w2fb = getWikidata2Freebase()
  wikititle2fb = getWikititle2Freebase()
  ids = entstr2id['wall street']

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
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents']#;ent_ctxs=dataEnts['ent_ctxs']
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
        docId_entstr2id[docId][enti_name.lower()]= {value}
      else:
        docId_entstr2id[docId][enti_name.lower()].add(value)
  #print docId_entstr2id
  print 'start to load wikititle...'
  s_time = time.time()
  w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
  print 'w2vModel:',time.time()-s_time
  wikititle_reverse_index  = getreverseIndex()

  print 'start to solve problems...'
  #
  title2pageId = getfname2pageid()
  mid2incomingLinks={}
  #if os.path.isfile(dir_path+'mid2incomingLinks.p'):
  #  print 'start to load mid2incomingLinks ... '
  #  mid2incomingLinks=cPickle.load(open(dir_path+'mid2incomingLinks.p','rb'))
  all_candidate_mids,wiki_candidate_mids,freebase_candidate_mids = get_final_ent_cands()
  data_param={'all_candidate_mids':all_candidate_mids,'wiki_candidate_mids':wiki_candidate_mids,'freebase_candidate_mids':freebase_candidate_mids}
  cPickle.dump(data_param,open(f_output,'wb'))
 