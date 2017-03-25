# -*- coding: utf-8 -*-
'''
@time: 2016/12/15
@editor: wujs
@function: to generate the entity linking features
@description: Ä¿Ç°ÏÈÊ¹ÓÃfreebaseÌá¹©µÄentity embeddingµÄ½á¹û¡£ºóÆÚ¿ÉÒÔÐÞ¸Ä³ÉtransEµÈÆäËûembeddingµÄ½á¹û
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
from NGDUtils import NGDUtils
ngd = NGDUtils()
nlp = English()
mongoutils= mongoUtils()
stopwords={}

with codecs.open('data/stopwords.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    stopwords[line] =1


def processDescription(line):
  strs = line.split(u'@en')[0]          
  strs = strs.replace(u'\\n',u' ')
  
  doc = nlp(strs)
  descript=[]
  for sentence in doc.sents:
    for token in sentence:
      if token.pos_ !='PUNCT' and token.pos_ != 'SPACE' and token.text.lower() not in stopwords:
        descript.append(token.text.lower())
        
  return descript
        
def get_all_candidate_mid_cocurrent(data_flag,ent_Mentions,all_candidate_mids,fout):
  '''
  @2016/12/26
  '''
  ent_ment_link_tags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
  if data_flag=='train':
    ent_id = 0
  if data_flag=='testa':
    ent_id = 23396
  if data_flag=='testb':
    ent_id = 29313
  print 'ent_id:', ent_id
  print 'finish load all datas'
  allcandents=collections.defaultdict(set)
  
  k=-1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(xrange(145,200)):
    ents = ent_Mentions[i]  #ents
    tag = False
    doc_ents_cand_mid_dict=[]
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content.lower()
      '''
      if enti_name not in entstr_lower2mid:
        continue
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      '''
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]

      if tag == 'NIL':
        ent_id += 1
        continue
      k += 1
      ent_id += 1
        
      cand_mid_dict = all_candidate_mids[k]
      for mid in cand_mid_dict:    
        midt = mid.replace(u'/',u'.')[1:]
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>' 
        if new_mid not in allcandents:
          allcandents[new_mid] = mongoutils.get_coOccurent_ents(new_mid)
  cPickle.dump(allcandents,open(fout,'wb'))


def get_candidate_rel_features(data_flag,ent_Mentions,all_candidate_mids,fout,allcandents_coents):
  candidate_nums = 40
  '''
  @2016/12/26
  '''
#  param_dict = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p','rb'))
#  entstr_lower2mid=param_dict['entstr_lower2mid']; mid2entstr_lower=param_dict['mid2entstr_lower']
  ent_ment_link_tags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
  if data_flag=='train':
    ent_id = 0
  if data_flag=='testa':
    ent_id = 23396
  if data_flag=='testb':
    ent_id = 29313
    
  print 'finish load all datas'
  
  ent_mention_relCoherent_feature=[]
  k=-1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(xrange(145,200)):
    ents = ent_Mentions[i]  
    doc_ents_cand_mid_dict=[]
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content.lower()
      '''
      if enti_name not in entstr_lower2mid:
        continue
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      '''
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]
      if tag == 'NIL':
        ent_id+=1
        continue
      k += 1
      ent_id+=1
        
      cand_mid_dict = all_candidate_mids[k]
      cand_mid_coocurmid = []
      for mid in cand_mid_dict:
        midt = mid.replace(u'/',u'.')[1:]
        new_mid = u'<http://rdf.freebase.com/ns/'+midt+u'>'
        
        #cand_mid_coocurmid.append([new_mid,allcandents_coents[new_mid]])
        cand_mid_coocurmid.append([new_mid,0])
      doc_ents_cand_mid_dict.append(cand_mid_coocurmid)
    
    doc_temprelCoherent=[]
    for icand in range(len(doc_ents_cand_mid_dict)):
      temprelCoherent = np.zeros((candidate_nums,))
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
              
              #if mid in j_coocurmid or midj in i_coocurmid:
              #  hasRel +=1
#              if midj in mid2Name:
#                nameJ = mid2Name.get(midj)
#                if nameI is not None and nameJ is not None:
#                  hasNGD += ngd.get_page_links(nameI,nameJ)
#                  nums += 1
#        if hasRel !=0:
#          print hasRel*1.0/len(i_coocurmid)
#          temprelCoherent[ci] = hasRel*1.0/len(i_coocurmid)
        if hasNGD!=0:
          temprelCoherent[ci] += hasNGD/nums
      doc_temprelCoherent.append(temprelCoherent)
    ent_mention_relCoherent_feature.append(doc_temprelCoherent)
  cPickle.dump(ent_mention_relCoherent_feature,open(fout,'wb'))
     
def get_candidate_ent_features(data_flag,ent_Mentions,all_candidate_mids,mid2description,descript_Words,mid2figer,fid2vec,f_output,f_output1,f_output2,f_output4):
  '''
  @2016/12/15
  '''
  candidate_nums = 40
  ent_ment_link_tags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
  if data_flag=='train':
    ent_id = 0
  if data_flag=='testa':
    ent_id = 23396
  if data_flag=='testb':
    ent_id = 29313
  print 'finish load all datas'
  
  non_description_mid = set()
  descrip_lent=[]
  
  
  ent_mention_surface_wordv=[]
  
  ent_mention_index=[]
  ent_mention_link_feature=[] #shape: [line,ent_mention_num, 30(candidates number) * 100(dimension)]
  ent_mention_tag = []  #shape:[line,ent_mention_num,30]
  ent_mention_type_feature=[]   #shape:[line,ent_mention_num,30* 113(one hot type dimension)]
  ent_mention_cand_prob_feature=[]
  
  non_mid2figer = 0
  k  = -1
  for i in tqdm(range(len(ent_Mentions))):
  #for i in tqdm(range(100)):
    ents = ent_Mentions[i]
    
    temps_mentwordv=[]
    
    temps =[]
    temps_type=[]
    temps_tag = []
    temps_cand_prob=[]
    temps_ent_index=[]
    
    #print i,'\tentM:',len(ents)
    for j in range(len(ents)):
      totalCand = 0
      enti = ents[j]
      
      enti_name = enti.content.lower()
      '''
      if enti_name not in entstr_lower2mid:
        continue   
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      '''
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]
     
      if tag == 'NIL':
        ent_id += 1
        continue
      ent_id += 1
      k += 1  
      tag_t = 0
      tmentwordv=[]
      
      #ent_mention_tag_temp = np.zeros((30,))
      ent_mention_tag_temp=[]
      tdescip = []
      tcanditetype=[]
      tcandprob = []
      
      
      cand_mid_dict = all_candidate_mids[k]
  
      for mid in cand_mid_dict:  
        if mid not in mid2description and mid == tag:   
          print 'mide not in mid2description', mid
        
        if mid == tag:
          ent_mention_tag_temp.append(tag_t)
        
        tag_t += 1
        
        if mid in fid2vec:
          tmentv = fid2vec[mid]
        else:
          tmentv = np.zeros((100,))
        
        twordv = np.zeros((100,))        
       
        ttypev = np.zeros((113,))
        if mid in mid2figer:
          for midtype in mid2figer[mid]:
            ttypev[midtype]=1
           
        if mid in mid2description:
          line = mid2description[mid]
          descript = processDescription(line)
          descrip_lent.append(len(descript))
          #@2017/1/25 position encoding(PE)
          qlent = 0
          tempWordEmbed=[]
          for idescrip in range(min(20,len(descript))):
            
            word = descript[idescrip]
            if word in descript_Words:
              qlent +=1
              tempWordEmbed.append(descript_Words[word])
              
          for idescrip in range(qlent):
            li =[]
            for jdescrip in range(100):  #100 stands for embedding dimension
              li.append(min((idescrip+1)*100/((jdescrip+1)*qlent),((jdescrip+1)*qlent)/((idescrip+1)*100)))
            twordv += tempWordEmbed[idescrip] * np.asarray(li)
              
        
        
        '''
        @add candidate entity type features!
        '''
        #revise 2017/2/2
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
        
        if len(tmentwordv)==0:
          tmentwordv = tmentv
        else:
          tmentwordv = np.concatenate((tmentwordv,tmentv))  
      temps_mentwordv.append(tmentwordv)
      
      temps_type.append(tcanditetype)
      temps.append(tdescip)
      temps_tag.append(ent_mention_tag_temp)
      temps_cand_prob.append(tcandprob)
      temps_ent_index.append((enti.startIndex,enti.endIndex)) 
      
    ent_mention_surface_wordv.append(temps_mentwordv)
    
    ent_mention_type_feature.append(temps_type)
    ent_mention_cand_prob_feature.append(temps_cand_prob)
    ent_mention_link_feature.append(temps)
    
    ent_mention_tag.append(temps_tag)
    ent_mention_index.append(temps_ent_index)
    
  '''
  print len(ent_mention_link_feature),len(ent_mention_link_feature[3]),len(ent_mention_link_feature[3][0])
  print len(ent_mention_tag),len(ent_mention_tag[3]),len(ent_mention_tag[3][0])
  print len(ent_mention_index),len(ent_mention_index[3]),len(ent_mention_index[3][0])
  print  max(descrip_lent),min(descrip_lent),sum(descrip_lent) / float(len(descrip_lent))
  '''
  cPickle.dump(ent_mention_surface_wordv,open(f_output4,'wb'))
  
  param_dict = {'ent_mention_index':ent_mention_index,'ent_mention_link_feature':ent_mention_link_feature,'ent_mention_tag':ent_mention_tag}
  cPickle.dump(param_dict,open(f_output,'wb'))
  print len(ent_mention_type_feature),len(ent_mention_type_feature[3]),len(ent_mention_type_feature[3][0])
  cPickle.dump(ent_mention_type_feature,open(f_output1,'wb'))
  cPickle.dump(ent_mention_cand_prob_feature,open(f_output2,'wb'))
  
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

def recalls():
  ent_ment_link_tags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
  if data_flag=='train':
    ent_id = 0
  if data_flag=='testa':
    ent_id = 23396
  if data_flag=='testb':
    ent_id = 29313
  print 'finish load all datas'
  pass_num = 0
  right_num = 0
  wrong_num = 0
  k = -1
  for i in tqdm(range(len(ent_Mentions))):
    ents = ent_Mentions[i] 
    for j in range(len(ents)):
      enti = ents[j]
      
      enti_name = enti.content.lower()
      '''
      if enti_name not in entstr_lower2mid:
        continue   
      else:
        k += 1
        tag = entstr_lower2mid[enti_name]
      '''
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]
       
      if tag == 'NIL':
        ent_id += 1
        pass_num += 1
        continue
      ent_id += 1
      k += 1  
      #print 'k:',k
      cand_mid_dict = all_candidate_mids[k]
      rightCand_flag = False
      for mid in cand_mid_dict:  
        if mid == tag:
          rightCand_flag = True
      if rightCand_flag:
        right_num += 1
      else:
        wrong_num += 1
  print 'pass_num:',pass_num
  print 'right num:',right_num
  print 'wrong num:',wrong_num
  print 'total num:',right_num+wrong_num
      
          
          
          
if __name__=='__main__':
  if len(sys.argv) != 11:
    print 'usage: python embeddings/generate_entmention_linking_features.py ${dir_path} testa_entms.p100 test_a_embed.p100 testa_ent_cand_mid.p testa_ent_linking.p testa_ent_linking_type.p testa_ent_linking_candprob.p testa_ent_relcoherent.p data_flag'
    exit(1)
  dir_path = sys.argv[1]
  f_input_ent_ments = dir_path + '/features/' + sys.argv[2]
  f_input_ent_embed = dir_path +'/features/' + sys.argv[3]
  f_input_ent_cand_mid = dir_path  +'/features/'+ sys.argv[4]
  f_output = dir_path  +'/features/'+ sys.argv[5]
  f_output1 = dir_path  +'/features/'+ sys.argv[6]
  f_output2 = dir_path  +'/features/'+ sys.argv[7]
  f_outputtemp = dir_path  +'/features/'+ sys.argv[8]+'temp'
  f_output3 = dir_path  +'/features/'+ sys.argv[8]
  f_output4 = dir_path  +'/features/' + sys.argv[9]
  data_flag = sys.argv[10]
  
  
 
  
  stime = time.time()
  print 'start to load datas....'
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_ent_ments,'rb'));ent_Mentions = dataEnts['ent_Mentions']
  all_candidate_mids = cPickle.load(open(f_input_ent_cand_mid,'rb'))
  print len(all_candidate_mids)
  
  #exit(0)
  print 'load dataEnts candidates cost time:',time.time()-stime
  print 'start to comput recall....'
  recalls()
  #get_all_candidate_mid_cocurrent(data_flag,ent_Mentions,all_candidate_mids,f_outputtemp)   
  #allcandents_coents = cPickle.load(open(f_outputtemp,'rb'))
  allcandents_coents={}
  mid2Name = getmid2Name()
  get_candidate_rel_features(data_flag,ent_Mentions,all_candidate_mids,f_output3,allcandents_coents)
  
  
  '''
  #sent_embed = cPickle.load(open(f_input_ent_embed,'rb'))
  #assert len(ent_Mentions) == len(sent_embed)
  '''
  '''
  @time: 2017/2/8
  @function: load re-rank feature£ºentity type
  '''
  fid2title = collections.defaultdict(list)
  fid2vec = {}
  wtitle2fid = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))
  for key in wtitle2fid:
    for item in wtitle2fid[key]:
      fid2title[item].append(key)
  '''
  @time: 2016/12/23
  @function: load re-rank feature£ºentity type
  '''  
  stime = time.time()
  descript_Words = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wordvec_model_100.p', 'rb'))
  print 'load wordvec_model_100 cost time: ', time.time()-stime
  
  wordsVectors = descript_Words.wvec_model
      
  for key in fid2title:
    num = 0
    fwordv = np.zeros((100,))
    for item in fid2title[key]:    
      num += 1
      twordv = np.zeros((100,))
      for word in item.split(u' '):
        if word in wordsVectors:
          twordv += wordsVectors[word]
      fwordv += twordv
    fwordv /=num
    fid2vec[key] = fwordv  
  
  stime = time.time()
  mid2figer = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/mid2figer.p','rb'))
  print 'load mid2figer cost time:',time.time()-stime
                                             

  stime = time.time()
  mid2description={}  #nearly 2.3G
  with codecs.open('/home/wjs/demo/entityType/informationExtract/data/mid2description.txt','r','utf-8') as file:
    for line in file:
      items = line.strip().split('\t')
      if len(items) >=2:
        mid2description[items[0]] =items[1]
  print 'load mid2descriptioon cost time: ', time.time()-stime
#
  get_candidate_ent_features(data_flag,ent_Mentions,all_candidate_mids,mid2description,descript_Words.wvec_model,mid2figer,fid2vec,f_output,f_output1,f_output2,f_output4)
  
  '''
  param_dict = cPickle.load(open(f_output,'rb'))
  ent_mention_link_feature = param_dict['ent_mention_link_feature']
  assert len(ent_mention_link_feature) == len(ent_Mentions)
  if len(ent_mention_link_feature) == len(ent_Mentions):
    print 'right'
  else:
    print 'wrong'
  '''
  
  

  
  
  
  