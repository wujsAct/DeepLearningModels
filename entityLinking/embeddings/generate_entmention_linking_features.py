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
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
import time

import cPickle
import gensim
from tqdm import tqdm

def get_candidate_ent_features(ent_Mentions,all_candidate_mids,w2v_ent_model):
  '''
  @2016/12/15  v1: 使用google提供的embedding结果，可能不太准确啦！
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
  
    for j in range(len(ents)):
     
    
if __name__=='__main__':
  if len(sys.argv) !=5:
    print 'usage: python pyfile dir_path testa_entms.p100(test) test_a_embed.p testa_ent_cand_mid.p'
    exit(1)
  dir_path = sys.argv[1]
  f_input_ent_ments = dir_path + '/features/' + sys.argv[2]
  f_input_ent_embed = dir_path +'/features/' + sys.argv[3]
  f_input_ent_cand_mid = dir_path  +'/features/'+ sys.argv[4]
  
  stime = time.time()
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_ent_ments,'rb'));ent_Mentions = dataEnts['ent_Mentions']
  
  sent_embed = cPickle.load(open(f_input_ent_embed,'rb'))
  assert len(ent_Mentions) == len(sent_embed)
  
  all_candidate_mids = cPickle.load(open(f_input_ent_cand_mid,'rb'))
  print 'load all related data cost time: ', time.time()-stime
  
  stime = time.time()
  w2v_ent_model = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/sdb1/data/freebase-vectors-skipgram1000-en.bin',binary=True)
  print 'load w2v_ent_model cost time: ', time.time()-stime
  
  
  
  
  
  
  
  
  