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
import Levenshtein
from PhraseRecord import EntRecord
import codecs
import gensim
import string

def is_contain_ents(enti,entj):
  enti = enti.lower()
  entj = entj.lower()
  
  enti_item = enti.split(u' ')
  entj_item = entj.split(u' ')
  if (len(enti_item)==1):
    if (enti in entj):
      return True
    else:
      return False
  else:
    for entii in enti_item:
      if entii not in entj_item:
        return False
    return True

def get_freebase_ent_cands(cantent_mid_prev,enti,tag,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum):
  #找出全部包含jordan的实体，然后使用NGD和REL进行消解，找到答案！今天下午把这个问题解决掉
  #部分与整体的coreference也能够解决一部分的问题呢！
  candi = 0
  #print 'go into entNGD...'
  distRet = {};
  #first find all the ents we need to process
  enti_item = enti.split(u' ')
  for entit in enti_item[0:1]:
    totaldict=dict()
    if entit in wikititle_reverse_index:
      totaldict = dict(totaldict,**wikititle_reverse_index[entit])
    if entit.lower() in wikititle_reverse_index:
      totaldict = dict(totaldict,**wikititle_reverse_index[entit.lower()])
      
    for key in totaldict:
      addScore = 0
      if enti.lower() == key.lower():  #completely match
        addScore += 0.5 
      if key in entstr2id: #在上下文中出现了的！可以解决一部分共指问题！
        addScore += 0.2
      distRet[wikititle2fb[key]]=Levenshtein.ratio(enti.lower(),key.lower()) + addScore
  distRet= sorted(distRet.iteritems(), key=lambda d:d[1], reverse = True)
  cantent_mid=set()
  for item in distRet:
    if freebaseNum==0:
      break
    if item[0] not in cantent_mid_prev:
      cantent_mid.add(item[0])
      freebaseNum -=1
  return cantent_mid

def get_cantent_mid(listentcs,w2fb,wikititle2fb):
  flag = False
  cantent_mid=set()
  for cent in listentcs:
    ids = cent[u'ids']
    titles = cent[u'title']
    if ids in w2fb:
      cantent_mid.add(w2fb[ids])
    elif titles in wikititle2fb:
      cantent_mid.add(wikititle2fb[titles])
  return cantent_mid

def get_ent_word2vec_cands(enti,w2fb,wikititle2fb):
  cantent_mid = set()
  try: 
    coherent_ents = w2vModel.most_similar(enti.content.replace(u' ',u'_'),topn=10) #convert to gensim style
    for citems in coherent_ents:
      cents = citems[0].replace(u'_',u' ')  #convert to freebase style
      if cents in entstr2id:
        entid = entstr2id[cents]
        listentcs = candiate_ent[entid]
        if len(listentcs)>=1:
          cantent_mid |= get_cantent_mid(listentcs[0:2],w2fb,wikititle2fb)
      if cents in wikititle2fb:
        cantent_mid.add(wikititle2fb[cents])
  except:
    pass
    
  return cantent_mid

def get_final_ent_cands(w2vModel,ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb,wikititle_reverse_index):
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
  
  for i in range(len(ent_Mentions)):
    if i%100==0:
      print i
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
      seta = set(aNo_has_ents[aNo])  #不要随意把一个对象赋值给另一个对象，否则可能会出错呢！
      entid = entstr2id[enti.content]
      seta.remove(enti.content.lower())  
      
      listentcs = candiate_ent[entid]
      
      cantent_mid = get_cantent_mid(listentcs,w2fb,wikititle2fb)   #get wikidata&dbpedia search candidates
      if enti.content in wikititle2fb:
        cantent_mid.add(wikititle2fb[enti.content])
      if enti.content.lower() in wikititle2fb:
        cantent_mid.add(wikititle2fb[enti.content.lower()])
      cantent_mid |= get_ent_word2vec_cands(enti,w2fb,wikititle2fb) #get word2vec coherent candidates
      freebaseNum = max(0,40 - len(cantent_mid))
      cantent_mid |= get_freebase_ent_cands(cantent_mid,enti.content,tag,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum) #search by freebase matching 这部分将花费很多的时间！
      
      totalCand += len(cantent_mid)
      
      if tag in cantent_mid:
        right_nums += 1
        #print 'right:',enti_name,totalCand
      else:
        wrong_nums = wrong_nums + 1
        print 'wrong:',tag,mid2entstr_lower[tag],enti.content.lower(),totalCand#,listentcs,'\n'
        
  print 'wrong_nums:',wrong_nums
  print 'right_nums:',right_nums
  print 'pass_nums:',pass_nums
        

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
  entstr2id = data['entstr2id']
  id2entstr = {value:key for key,value in entstr2id.items()}
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
  get_final_ent_cands(w2vModel,ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb,wikititle_reverse_index)
  
  
  '''
  #避免反复去计算这个值
  if os.path.isfile(f_output1):
    allents = cPickle.load(open(f_output1,'rb'))
  else:
    allents = getAllEnts(ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb)
    cPickle.dump(allents,open(f_output1,'wb'))
    
  allents = math.log(allents,2)

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
#      aNo = ent_ctx[j][0]
#      seta = set(aNo_has_ents[aNo])  #不要随意把一个对象赋值给另一个对象，否则可能会出错呢！
#      entid = entstr2id[enti.content]
#      seta.remove(enti.content.lower())
#      listb = candiate_coCurrEnts[entid]
#      listentcs = candiate_ent[entid]
#      if len(listb)==0:
#        print enti.content
#      else:
#        tempval = []
#        fbents = []
#        for coentid in range(len(listb)):
#          setb = set()
#          for item in listb[coentid]:
#            setb.add(item.split(u'\t')[1].lower())
#          
#          enti_name = enti.content
#          ids = listentcs[coentid][u'ids']
#          titles = listentcs[coentid][u'title']
#          description = listentcs[coentid][u'description']
#          if ids in w2fb:
#            fbents.append((w2fb[ids],listentcs[coentid]))
#            #description 啥的还是用原来的就ok呢！
#          elif titles in wikititle2fb:
#            fbents.append((wikititle2fb[titles],listentcs[coentid]))
#        #此处将candidates转化成freebase的id信息啦！
#        if len(fbents)==0:
#          noncandidates = noncandidates + 1
#          print enti.content,listentcs
#        else:
#          havacandidates = havacandidates + 1
#        temp_ent_candidateEnts.append(fbents)
#    ent_candidateEnts.append(temp_ent_candidateEnts)
#  
#  print 'has no candidates:', noncandidates
#  print ent_Mentions[0],ent_candidateEnts[0]
#  print 'has  candidates:', havacandidates
  '''
  
