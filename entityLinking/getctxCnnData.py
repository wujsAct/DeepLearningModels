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
import gensim
import string
'''
def entCoherentInWord2vec():
  #利用google提供的pretrain的东西去做topic coherent，简化问题呗
  

def entNGD():
  #利用别人提供好的entity coherent来进行ranker

'''
def getAllEnts(ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb):
  allents = set()
  for i in range(len(ent_Mentions)):
    if i%100==0:
      print i
    ents = ent_Mentions[i]
    ent_ctx = ent_ctxs[i]  #ent_ctx.append([aNo,ctx])
      
    for j in range(len(ents)):
      enti = ents[j]
      aNo = ent_ctx[j][0]
      seta = set(aNo_has_ents[aNo])  #不要随意把一个对象赋值给另一个对象，否则可能会出错呢！
      entid = entstr2id[enti.content]
      seta.remove(enti.content.lower())
      allents = allents|seta
      listb = candiate_coCurrEnts[entid]
      listentcs = candiate_ent[entid]
      print enti.content,listentcs
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
    exit(-1)
  return len(allents)

def hasRightCandidates(tag,listentcs,w2fb,wikititle2fb):
  flag = False
  cantent_title=[]
  cantent_mid=[]
  for cent in listentcs:
    ids = cent[u'ids']
    titles = cent[u'title']
    cantent_title.append(titles.lower())
    if ids in w2fb:
      cantent_mid.append(w2fb[ids])
    elif titles in wikititle2fb:
      cantent_mid.append(wikititle2fb[titles])
    if tag in cantent_mid:
      flag = True
      break
  return flag,cantent_title
    
def getLinkTags(w2vModel,ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb):
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
  print entstr_lower2mid
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
      enti = ents[j]
      
      enti_name = ents[j].content.lower()
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
      
      flag,cantent_title =  hasRightCandidates(tag,listentcs,w2fb,wikititle2fb)
      if flag:
        right_nums += 1
        #print 'right:',enti_name
      else:
        enti_name = string.capwords(enti_name)
        enti_name = enti_name.replace(u' ',u'_')  #revise into gensim style
        #print 'search enti_name',enti_name
        try:
          coherent_ents = w2vModel.most_similar(enti_name)
          coFlag = False
          for citems in coherent_ents:
            cents = citems[0]
            if cents not in cantent_title:
              if cents.capitalize() in entstr2id:
                entid = entstr2id[cents.capitalize()]
                listentcs = candiate_ent[entid]
                flag,cantent_title =  hasRightCandidates(tag,listentcs,w2fb,wikititle2fb)
                if flag ==True:
                  coFlag= True
                  break
              if cents.upper() in entstr2id:
                entid = entstr2id[cents.upper()]
                listentcs = candiate_ent[entid]
                flag,cantent_title =  hasRightCandidates(tag,listentcs,w2fb,wikititle2fb)
                if flag ==True:
                  coFlag= True
                  break
          if coFlag==True:
            right_nums += 1
            #print 'right:',enti_name
          else:
            wrong_nums = wrong_nums + 1
            print 'wrong:',tag,mid2entstr_lower[tag]#,enti.content.lower(),listentcs,'\n'
        except:
          wrong_nums = wrong_nums + 1
          print 'wrong:',tag,mid2entstr_lower[tag]#,enti.content.lower(),listentcs,'\n'
          print '--------------\n'
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
  entstr2id = data['entstr2id'];
  id2entstr = {value:key for key,value in entstr2id.items()}
  print id2entstr
  candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
  
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_entMents,'r'))
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents'];ent_ctxs=dataEnts['ent_ctxs']
  print ent_Mentions
  all_ents = set()
  w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
  wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
  print 'start to solve problems...'
  w2vModel = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin',binary=True)
  getLinkTags(w2vModel,ent_ctxs,entstr2id,ent_Mentions,aNo_has_ents,candiate_coCurrEnts,candiate_ent,w2fb,wikititle2fb)
  
  
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
  
