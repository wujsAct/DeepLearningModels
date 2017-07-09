#from __future__ import print_function
# -*- coding: utf-8 -*-
'''
@editor: wujs
function: generate entity mention using NER model results
revise: 2017/1/11
'''

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
from TFRecordUtils import ner_d3array_TFRecord
import numpy as np
import collections
import cPickle as cpkl
import argparse
from description_embed_model import WordVec,MyCorpus
from random_vec import RandomVec
import codecs
import time
import gensim


def find_max_length(file_name):
  temp_len = 0
  max_length = 0
  for line in open(file_name):
      if line in ['\n', '\r\n']:
          if temp_len > max_length:
              max_length = temp_len
          temp_len = 0
      else:
          temp_len += 1
  return max_length


def pos(tag):
  one_hot = np.zeros(5)
  if tag == 'NN' or tag == 'NNS':
      one_hot[0] = 1
  elif tag == 'FW':
      one_hot[1] = 1
  elif tag == 'NNP' or tag == 'NNPS':
      one_hot[2] = 1
  elif 'VB' in tag:
      one_hot[3] = 1
  else:
      one_hot[4] = 1
  return one_hot


def chunk(tag):
  one_hot = np.zeros(5)
  if 'NP' in tag:
      one_hot[0] = 1
  elif 'VP' in tag:
      one_hot[1] = 1
  elif 'PP' in tag:
      one_hot[2] = 1
  elif tag == 'O':
      one_hot[3] = 1
  else:
      one_hot[4] = 1
  return one_hot

'''
revise time: 2017/1/11, add more extrac features!
'''
def capital(word,prevword,wtitleIndex):
  wordl = word.lower()
  ret = np.array([0,0,0,0])
  if ord('A') <= ord(word[0]) <= ord('Z'):
    ret[0]=1
  if wordl in wtitleIndex:
    ret[1]=1
    if wordl in wtitleIndex[wordl]:
      ret[2]=1
  if ord('A') <= ord(prevword[0]) <= ord('Z'):  #previous tags!
    ret[3]=1
  return ret

def get_input_aida(model,word_dim,input_file,sentence_length):
  print('processing %s' % input_file)
  vocabs = model.vocab
  word = []
  tag = []
  sentence = []
  sentence_tag = []
 
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:
      sentence.append(word)
      sentence_tag.append(np.array(tag))

      sentence_length = 0
      word = []
      tag = []
    else:
      assert (len(line.split()) == 4)
      sentence_length += 1
      wd = line.split()[0]
      if wd in vocabs:
        temp = model[wd]
      elif wd.lower() in vocabs:
        temp = model[wd.lower()]
      else:
        temp = np.zeros((300,))[:word_dim]
      #print len(temp), word_dim
      assert len(temp) == word_dim
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      #temp = np.append(temp ,capital(line.split()[0],prevword,wtitleIndex))  # adding capital embedding
      assert len(temp) == word_dim + 10
      word.append(temp)
      t = line.split()[3]
      # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
      if t.endswith('PER'):
        tag.append(np.array([1, 0, 0, 0, 0]))
      elif t.endswith('LOC'):
        tag.append(np.array([0, 1, 0, 0, 0]))
      elif t.endswith('ORG'):
        tag.append(np.array([0, 0, 1, 0, 0]))
      elif t.endswith('MISC'):
        tag.append(np.array([0, 0, 0, 1, 0]))
      elif t.endswith('O'):
        tag.append(np.array([0, 0, 0, 0, 1]))
      else:
        print("error in input tag {%s}" % t)
        sys.exit(0)
  assert (len(sentence) == len(sentence_tag))
  return sentence,sentence_tag
  

def get_input(model,word_dim, input_file,output_entms,id2aNosNo,sents2id,ents,sentence_length=-1):
  print('processing %s' % input_file)
  sent=[]
  sentence = []
  sentence_tag = []
  ent_Mentions = []
  
  aNo_has_ents=collections.defaultdict(set)
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:

      senti = u' '.join(sent)
      if senti in sents2id:
        ids = sents2id[senti]
        aNosNo = id2aNosNo[ids]
        aNo = aNosNo.split('_')[0]
        entm = ents[ids][0]
        ent_Mentions.append(entm)
        for enti in entm:
          aNo_has_ents[aNo].add(enti.getContent().lower())
          s,e  = enti.getIndexes()
      else:
        print(senti)

      sentence_length = 0
      sent=[]
    else:
      assert (len(line.split()) == 4)
      sentence_length += 1
      sent.append(line.split()[0])
  print('finished!!')
  assert (len(sentence) == len(sentence_tag))
  print('start to save the data!!')
  
  param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents}
  cpkl.dump(param_dict, open(output_entms, 'wb'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str, help='data file', required=True)
  parser.add_argument('--data_train', type=str, help='all raw data e.g. entity mentions(train.p)', required=True)
  parser.add_argument('--data_testa', type=str, help='all raw data e.g. entity mentions(testa.p)', required=True)
  parser.add_argument('--data_testb', type=str, help='all raw data e.g. entity mentions(testb.p)', required=True)
  parser.add_argument('--train', type=str, help='train file location', required=True)
  parser.add_argument('--test_a', type=str, help='test_a file location', required=True)
  parser.add_argument('--test_b', type=str, help='test_b location', required=True)
  parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
  parser.add_argument('--use_model', type=str, help='model location', required=True)
  parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
  
  print 'start to load wtitleReverseIndex'
  start_time = time.time()
  #wtitleIndex = cpkl.load(open('data/wtitleReverseIndex.p','rb')) 
  
  args = parser.parse_args()
  
  print 'start to load word2vec models!'
  trained_model = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
  print 'load word2vec model cost time:',time.time()-start_time
  #print trained_model.wvec_model.vocab
  
  
  
  data = cpkl.load(open(args.data_train,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents']
  sents2id = {sent:i for i,sent in enumerate(sents)}
  get_input(trained_model, args.model_dim, args.train,
            args.dir_path+'/features/train_entms.p'+str(args.model_dim),
            id2aNosNo,sents2id,ents,
            sentence_length=args.sentence_length)
  
  
  data = cpkl.load(open(args.data_testa,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents']

  sents2id = {sent:i for i,sent in enumerate(sents)}
  get_input(trained_model, args.model_dim, args.test_a,
            args.dir_path+'/features/testa_entms.p'+str(args.model_dim),
            id2aNosNo,sents2id,ents,
            sentence_length=args.sentence_length)
  
  
  data = cpkl.load(open(args.data_testb,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents']
  sents2id = {sent:i for i,sent in enumerate(sents)}
  get_input(trained_model,args.model_dim, args.test_b,
            args.dir_path+'/features/testb_entms.p'+str(args.model_dim),
            id2aNosNo,sents2id,ents,
            sentence_length=-1)
  