import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
sys.path.append('embeddings')
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord

import numpy as np
import collections
import pickle as pkl
import cPickle as cpkl
import argparse
from wordvec_model import WordVec
from glove_model import GloveVec
from rnnvec_model import RnnVec
import codecs


def get_similar_entities(entstr):
  entstr = entstr.lower()
  items = entstr.split(u' ')
  temp = np.zeros(100)
  try:
    temp = word2vec_model.wvec_model[entstr.replace(u' ',u'_')]
  except:
    try:
      for key in items:
        temp += word2vec_model.wvec_model[key]
    except:
      pass
  print '--------------'
  print entstr, word2vec_model.wvec_model.most_similar(positive=[temp],topn=20)     
if __name__=='__main__':
  if len(sys.argv) !=4:
    print 'usage: python pyfile dir_path inputfile inputfile2'
    exit(1)
  dir_path = sys.argv[1]
  f_input1 = dir_path  +sys.argv[2]  # testa_candEnts.p
  f_input2 = dir_path + sys.argv[3]  # models
  word2vec_model = pkl.load(open(f_input2, 'rb'))
  
  # 'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts
  para_dict = cpkl.load(open(f_input1,'r'))
  entstr2id = para_dict['entstr2id']; candiate_ent = para_dict['candiate_ent']; candiate_coCurrEnts = para_dict['candiate_coCurrEnts']
  #print word2vec_model.wvec_model.vocab()
  get_similar_entities('jordan')
  get_similar_entities('collins')
  get_similar_entities('roberts')
  get_similar_entities('oscar')
  get_similar_entities('roman')
  get_similar_entities('kurdish')
  get_similar_entities('chinese')