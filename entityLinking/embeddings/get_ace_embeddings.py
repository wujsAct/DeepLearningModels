#from __future__ import print_function

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
import cPickle as pkl
import numpy as np
import gensim
import codecs
from tqdm import tqdm
import collections
import argparse


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


def capital(word):
  if ord('A') <= ord(word[0]) <= ord('Z'):
      return np.array([1])
  else:
      return np.array([0])


def get_input(model, word_dim, input_file,output_embed,sentence_length=-1):
  print('processing %s' % input_file)
  word = []
  tag = []
  sent=[]
  sentence = []
  ent_Mentions = []
  ent_ctxs = []
  aNo_has_ents=collections.defaultdict(set)
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file)
  else:
    max_sentence_length = sentence_length
  print 'max sentence length:',max_sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        tag.append(np.array([0] * 5))
        temp = np.array([0 for _ in range(word_dim + 11)])
        word.append(temp)
      
      sentence.append(word)
      
      sentence_length = 0
      word = []
      tag = []
    else:
      assert (len(line.split()) == 3)
      sentence_length += 1
      temp = model[line.split()[0]]
      assert len(temp) == word_dim
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
      word.append(temp)
      '''
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
      '''
  print('finished!!')
  print('start to save the data!!')
    
  pkl.dump(sentence, open(output_embed, 'wb'))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str, help='data file', required=True)
  parser.add_argument('--train', type=str, help='train file location', required=True)
  parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
  parser.add_argument('--use_model', type=str, help='model location', required=True)
  parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
  
  args = parser.parse_args()
  
  trained_model = pkl.load(open(args.use_model, 'rb'))
  #print trained_model.wvec_model.vocab
  get_input(trained_model, args.model_dim, args.train, 
            args.dir_path+'/features/ace_embed.p'+str(args.model_dim),
            sentence_length=args.sentence_length)
