# -*- coding: utf-8 -*-
'''
@time: 2016/12/15
@editor: wujs
@function: to generate the entity linking features
@description: 目前先使用freebase提供的entity embedding的结果。后期可以修改成transE等其他embedding的结果
'''

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
sys.path.append('embeddings')
from spacyUtils import spacyUtils
from mongoUtils import mongoUtils
from PhraseRecord import EntRecord
from gensim.models.word2vec import Word2Vec
from random_vec import RandomVec
from description_embed_model import WordVec,MyCorpus
import time
import cPickle
import numpy as np
import gensim
import codecs
from tqdm import tqdm
import collections
from spacy.en import English

nlp = English()
mongoutils= mongoUtils()

def processDescription(line):
  strs = line.split(u'@en')[0]          
  strs = strs.replace(u'\\n',u' ') #此处有点奇怪呢！否则分句的时候会出现问题！
  
  doc = nlp(strs)
  descript=[]
  for sentence in doc.sents:
    for token in sentence:
      if token.pos_ !='PUNCT' and token.pos_ != 'SPACE':
        descript.append(token.text.lower())
        
  return descript

def get_all_candidate_mid_cocurrent(ent_Mentions,all_candidate_mids,fout):
  '''
  @2016/12/26 抽取候选实体间，是否在freebase中有共现
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
  print 'finish load all datas'
  allcandents=collections.defaultdict(set)
  
  k=-1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(xrange(145,200)):
    ents = ent_Mentions[i]  #ents 中的entities表示在同一个doc中的实体！
    tag = False
    doc_ents_cand_mid_dict=[]
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content.lower()
      if enti_name not in entstr_lower2mid:
        continue
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      
      cand_mid_dict = all_candidate_mids[k]
      cand_mid_coocurmid = []
      for mid in cand_mid_dict:    #注意细节的处理，不然数据处理非常麻烦呢！
        midt = mid.replace(u'/',u'.')[1:]
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>' #/m/0h5k 转换成freebase中完整的格式啦！
        #print new_mid
        if new_mid not in allcandents:
          allcandents[new_mid] = mongoutils.get_coOccurent_ents(new_mid)
  cPickle.dump(allcandents,open(fout,'wb'))

def get_candidate_rel_features(ent_Mentions,all_candidate_mids,fout,allcandents_coents):
  '''
  @2016/12/26 抽取候选实体间，是否在freebase中有共现
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
  print 'finish load all datas'
  ent_mention_relCoherent_feature=[]
  k=-1
  #for i in tqdm(range(len(ent_Mentions))):
  for i in tqdm(xrange(145,200)):
    ents = ent_Mentions[i]  #ents 中的entities表示在同一个doc中的实体！
    tag = False
    doc_ents_cand_mid_dict=[]
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content.lower()
      if enti_name not in entstr_lower2mid:
        continue
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      
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
      #print temprelCoherent
    
      doc_temprelCoherent.append(temprelCoherent)
    ent_mention_relCoherent_feature.append(doc_temprelCoherent)
  print len(ent_mention_relCoherent_feature),len(ent_mention_relCoherent_feature[3]),len(ent_mention_relCoherent_feature[3][0])
  cPickle.dump(ent_mention_relCoherent_feature,open(fout,'wb'))
     
def get_candidate_ent_features(ent_Mentions,all_candidate_mids,mid2description,descript_Words,mid2figer,f_output,f_output1):
  '''
  @2016/12/15  v1: 最好的方法，根据mid去找相对应的wiki页面，然后进行训练！
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
  print 'finish load all datas'
  pass_nums = 0
  
  non_description_mid = set()
  descrip_lent=[]
  
  '''
  此处特征的产生比较拗口呢！
  '''
  ent_mention_index=[]
  ent_mention_link_feature=[] #shape: [line,ent_mention_num, 30(candidates number) * 100(dimension)]
  ent_mention_tag = []  #shape:[line,ent_mention_num,30]
  ent_mention_type_feature=[]   #shape:[line,ent_mention_num,30* 113(one hot type dimension)]
  non_mid2figer = 0
  k  = -1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(range(100)):
    ents = ent_Mentions[i]
    temps =[]
    temps_type=[]
    temps_tag = []
    temps_ent_index=[]
    #print i,'\tentM:',len(ents)
    for j in range(len(ents)):
      totalCand = 0
      enti = ents[j]
      tag = False
      enti_name = enti.content.lower()
      if enti_name not in entstr_lower2mid:
        continue   # pass entity我们不进行处理啦！
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      
      if tag !=False:
        tag_t = 0
        ent_mention_tag_temp = np.zeros((30,))
        tdescip = []
        tcanditetype=[]
        cand_mid_dict = all_candidate_mids[k]
    
        for mid in cand_mid_dict:  #仅有description,实体共现！
          if mid not in mid2description and mid == tag:   #用来去抓取需要linking的实体啦！,filter to ensure all the result candidates has the description!
            print 'mide not in mid2description', mid
          if mid == tag:
            ent_mention_tag_temp[tag_t] = 1
          tag_t += 1
          if mid in mid2description:
            twordv = np.zeros((100,)) 
            ttypev = np.zeros((113,))
            if mid in mid2figer:
              for midtype in mid2figer[mid]:
                ttypev[midtype]=1
  
            line = mid2description[mid]
            descript = processDescription(line)
            descrip_lent.append(len(descript))
            for i in range(min(15,len(descript))):
              word = descript[i]
              if word in descript_Words:
                twordv += descript_Words[word]
            '''
            @add candidate entity type features!
            '''
            if len(tcanditetype)==0:
              tcanditetype = ttypev
            else:
              tcanditetype = np.concatenate((tcanditetype,ttypev))
              
            if len(tdescip)==0:
              tdescip = twordv
            else:
              tdescip = np.concatenate((tdescip,twordv))
        temps_type.append(tcanditetype)
        temps.append(tdescip)
        temps_tag.append(ent_mention_tag_temp)
        temps_ent_index.append((enti.startIndex,enti.endIndex))  #通过这个flag去抽取lstm最后一层的特征啦！
    ent_mention_type_feature.append(temps_type)
    ent_mention_link_feature.append(temps)
    ent_mention_tag.append(temps_tag)
    ent_mention_index.append(temps_ent_index)   
  print len(ent_mention_link_feature),len(ent_mention_link_feature[3]),len(ent_mention_link_feature[3][0])
  print len(ent_mention_tag),len(ent_mention_tag[3]),len(ent_mention_tag[3][0])
  print len(ent_mention_index),len(ent_mention_index[3]),len(ent_mention_index[3][0])
  print  max(descrip_lent),min(descrip_lent),sum(descrip_lent) / float(len(descrip_lent))
  param_dict = {'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag}
  #cPickle.dump(param_dict,open(f_output,'wb'))
  print len(ent_mention_type_feature),len(ent_mention_type_feature[3]),len(ent_mention_type_feature[3][0])
  cPickle.dump(ent_mention_type_feature,open(f_output1,'wb'))
  
  print non_mid2figer
  
if __name__=='__main__':
  if len(sys.argv) !=8:
    print 'usage: python pyfile dir_path testa_entms.p100(test) test_a_embed.p testa_ent_cand_mid.p testa_ent_linking.p testa_ent_linking_type.p testa_ent_relcoherent.p'
    exit(1)
  dir_path = sys.argv[1]
  f_input_ent_ments = dir_path + '/features/' + sys.argv[2]
  f_input_ent_embed = dir_path +'/features/' + sys.argv[3]
  f_input_ent_cand_mid = dir_path  +'/features/'+ sys.argv[4]
  f_output = dir_path  +'/features/'+ sys.argv[5]
  f_output1 = dir_path  +'/features/'+ sys.argv[6]
  f_outputtemp = dir_path  +'/features/'+ sys.argv[7]+'temp'
  f_output2 = dir_path  +'/features/'+ sys.argv[7]
  '''
  @time: 2016/12/23
  @function: load re-rank feature：entity type
  '''
  stime = time.time()
  #mid2figer = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/mid2figer.p','rb'))
  #print 'load mid2figer cost time:',time.time()-stime
  #print mid2figer
  #exit()
  
  stime = time.time()
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_ent_ments,'rb'));ent_Mentions = dataEnts['ent_Mentions']
  all_candidate_mids = cPickle.load(open(f_input_ent_cand_mid,'rb'))
  print 'load all data cost time:',time.time()-stime
  get_all_candidate_mid_cocurrent(ent_Mentions,all_candidate_mids,f_outputtemp)
  allcandents_coents = cPickle.load(open(f_outputtemp,'rb'))
  get_candidate_rel_features(ent_Mentions,all_candidate_mids,f_output2,allcandents_coents)

  '''
  totalEntNum = 0
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(range(100)):
    ents = ent_Mentions[i]
    temps =[]
    temps_tag = []
    temps_ent_index=[]
    #print i,'\tentM:',len(ents)
    for j in range(len(ents)):
      totalEntNum += 1
  print totalEntNum
  
  
  #sent_embed = cPickle.load(open(f_input_ent_embed,'rb'))
  #assert len(ent_Mentions) == len(sent_embed)
  
  descript_Words = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wordvec_model_100.p', 'rb'))
  print 'load all related data cost time: ', time.time()-stime
  
  stime = time.time()
  mid2description={}  #nearly 2.3G
  with codecs.open('/home/wjs/demo/entityType/informationExtract/data/mid2description.txt','r','utf-8') as file:
    for line in file:
      items = line.strip().split('\t')
      if len(items) >=2:
        mid2description[items[0]] =items[1]
  print 'load mid2descriptioon cost time: ', time.time()-stime
  #get_candidate_ent_features(ent_Mentions,all_candidate_mids,mid2description,descript_Words.wvec_model,mid2figer,f_output,f_output1)
  '''
  '''
  param_dict = cPickle.load(open(f_output,'rb'))
  ent_mention_link_feature = param_dict['ent_mention_link_feature']
  assert len(ent_mention_link_feature) == len(ent_Mentions)
  if len(ent_mention_link_feature) == len(ent_Mentions):
    print 'right'
  else:
    print 'wrong'
  '''
  
  
  
  
  
  
  