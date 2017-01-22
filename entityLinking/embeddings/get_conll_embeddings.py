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
revise time: 2017/1/11, 需要加上特征是否在知识库中存在这个实体，data/wtitleReverseIndex.p  ==？来提升对没见过的实体进行抽取哈�?不然使用conell进行特征训练的话，太依赖word embedding这个变量了！
'''
def capital(word,wtitleIndex):
  wordl = word.lower()
  if ord('A') <= ord(word[0]) <= ord('Z'):
    if wordl not in wtitleIndex:
      if wordl in wtitleIndex[wordl]:
        return np.array([1,0,0,0])  #whole in wtitle
      else:
        return np.array([0,1,0,0]) #part in wtitle
    else:
      return np.array([0,0,1,0]) #only big words
  else:
    return np.array([0,0,0,1]) #lower words


def get_input(model,wtitleIndex,word_dim, input_file, output_embed, output_tag,output_entms,id2aNosNo,sents2id,ents,tags, sentence_length=-1):
  print('processing %s' % input_file)
  word = []
  tag = []
  sent=[]
  sentence = []
  sentence_tag = []
  ent_Mentions = []
  ent_ctxs = []
  aNo_has_ents=collections.defaultdict(set)
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        tag.append(np.array([0] * 5))
        temp = np.array([0 for _ in range(word_dim + 14)])   #�˴�������һ���������
        word.append(temp)
      #ctx information and candidates information
      senti = u' '.join(sent)
      lent = len(sent)
      if senti in sents2id:
        ids = sents2id[senti]
        aNosNo = id2aNosNo[ids]
        aNo = aNosNo.split('_')[0]
        entm = ents[ids][0]
        ent_Mentions.append(entm)
        ent_ctx=[]
        for enti in entm:
          aNo_has_ents[aNo].add(enti.getContent().lower())
          s,e  = enti.getIndexes()
          ctx = u' '.join(sent[max(0,s-5):min(lent,e+5)])
          ent_ctx.append([aNo,ctx])
        ent_ctxs.append(ent_ctx)
        sentence.append(word)
        sentence_tag.append(np.array(tag))
      else:
        print(senti)

      sentence_length = 0
      word = []
      tag = []
      sent=[]
    else:
      assert (len(line.split()) == 4)
      sentence_length += 1
      temp = model[line.split()[0]]
      sent.append(line.split()[0])
      #print(line.split()[0])
      assert len(temp) == word_dim
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      temp = np.append(temp, capital(line.split()[0],wtitleIndex))  # adding capital embedding
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
  print('finished!!')
  assert (len(sentence) == len(sentence_tag))
  print('start to save the data!!')
  
  cpkl.dump(sentence, open(output_embed, 'wb'))
  cpkl.dump(sentence_tag, open(output_tag, 'wb'))
  '''
  @author:wujs
  revise time:2017/1/9, utilzie tf record to store the data
  '''
  #print np.shape(sentence)
  #print np.shape(sentence_tag)
  #ner_d3array_TFRecord(sentence,sentence_tag,output_embed+'.tfrecords',output_embed+'.shape')
  param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs}
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
  wtitleIndex = cpkl.load(open('data/wtitleReverseIndex.p','rb')) 
  
  args = parser.parse_args()
  
  print 'start to load word2vec models!'
  trained_model = cpkl.load(open(args.use_model, 'rb'))
  print 'finish load cost time:',time.time()-start_time
  #print trained_model.wvec_model.vocab
  
  data = cpkl.load(open(args.data_train,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents'];tags=data['tags']
  sents2id = {sent:i for i,sent in enumerate(sents)}
  get_input(trained_model,wtitleIndex, args.model_dim, args.train,
            args.dir_path+'/features/train_embed.p'+str(args.model_dim),
            args.dir_path+'/features/train_tag.p'+str(args.model_dim),
            args.dir_path+'/features/train_entms.p'+str(args.model_dim),
            id2aNosNo,sents2id,ents,tags,
            sentence_length=args.sentence_length)
  
  
  '''
  data = cpkl.load(open(args.data_testa,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents'];tags=data['tags']

  sents2id = {sent:i for i,sent in enumerate(sents)}
  get_input(trained_model,wtitleIndex, args.model_dim, args.test_a,
            args.dir_path+'/features/test_a_embed.p'+str(args.model_dim),
            args.dir_path+'/features/test_a_tag.p'+str(args.model_dim),
            args.dir_path+'/features/testa_entms.p'+str(args.model_dim),
            id2aNosNo,sents2id,ents,tags,
            sentence_length=args.sentence_length)
  
  
  data = cpkl.load(open(args.data_testb,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents'];tags=data['tags']
  sents2id = {sent:i for i,sent in enumerate(sents)}
  get_input(trained_model,wtitleIndex,args.model_dim, args.test_b,
            args.dir_path+'/features/test_b_embed.p'+str(args.model_dim),
            args.dir_path+'/features/test_b_tag.p'+str(args.model_dim),
            args.dir_path+'/features/testb_entms.p'+str(args.model_dim),
            id2aNosNo,sents2id,ents,tags,
            sentence_length=args.sentence_length)
  '''