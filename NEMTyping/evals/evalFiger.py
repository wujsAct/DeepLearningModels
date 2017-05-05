# -*- coding: utf-8 -*-
"""
Created on Tue May 02 10:39:03 2017

@author: DELL
function: evaluate figer test
"""
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/embedding/")
from description_embed_model import WordVec,MyCorpus
from get_type_embeddings import get_input_figerTest_chunk
import numpy as np
import cPickle
import time

start_time = time.time()
#word2vecModel = cPickle.load(open('data/wordvec_model_100.p'))
word2vecModel=None
print 'load word2vec model cost time:',time.time()-start_time


dir_path = 'data/figer_test/'
input_file_obj = open(dir_path+'features/figerData.txt')
  
entMents = cPickle.load(open(dir_path+'features/'+'figer_entMents.p','rb'))

'''
#load ner result
'''
ner_ret =  cPickle.load(open(dir_path + 'nerFeatures/figer_NERret.p','rb'))
length = ner_ret['length']

test_entment_mask,test_sentence,test_tag = get_input_figerTest_chunk(dir_path,model=word2vecModel,word_dim=100,sentence_length=80)
test_pred= cPickle.load(open(dir_path + 'fulltypeFeatures/figer_TypeRet.p','rb'))



allEnts = 0
entid = 0
rightEnts = 0
for entlist in entMents:
  for ent in entlist:
    target = ent[2]
    lenttype = len(target)*(-1)
    pred = test_pred[entid]
    preds=np.argsort(pred)[lenttype:]
  
    allEnts += 1
    print np.argsort(pred)[10:]
    print target,preds
    if sorted(target) == sorted(preds):
      rightEnts += 1
    entid += 1
    
print 'all ents:', allEnts,'right ents:',rightEnts,''

print np.argsort([1,2,4,-1])