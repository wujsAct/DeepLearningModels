#from __future__ import print_function

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
sys.path.append('embeddings')
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
  maxlenLine = ""
  with codecs.open(file_name,'r','utf-8') as content_file:
    text = content_file.read()
    for items in text.split('\n\n'):
      temp_len = len(items.split('\n'))
      if temp_len > max_length:
        max_length = temp_len
        maxlenLine = items
  #print 'maxlenLine:',maxlenLine
  #print max_length
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
#def capital(word):
#  ret =np.array([0])
#  if ord('A') <= ord(word[0]) <= ord('Z'): #inital words are capital
#    ret[0]=1
#  return ret

#def capital(word,prevword,wtitleIndex):
#  wordl = word.lower()
#  ret = np.array([0,0,0,0])
#  if ord('A') <= ord(word[0]) <= ord('Z'):
#    ret[0]=1
#  if wordl in wtitleIndex:
#    ret[1]=1
#    if wordl in wtitleIndex[wordl]:
#      ret[2]=1
#  if ord('A') <= ord(prevword[0]) <= ord('Z'):  #previous tags!
#    ret[3]=1
#  return ret


def get_input(model,wtitleIndex, word_dim, input_file,output_embed,sentence_length=-1):
  print('processing %s' % input_file)
  word = []
  #tag = []
  sentence = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file)
  else:
    max_sentence_length = sentence_length
  print 'max sentence length:',max_sentence_length
  sentence_length = 0
  #prevword=''
  print("max sentence length is %d" % max_sentence_length)
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:
#      for _ in range(max_sentence_length - sentence_length):
#        tag.append(np.array([0] * 5))
#        temp = np.array([0 for _ in range(word_dim + 14)])
#        word.append(temp)
      
      sentence.append(word)
      
      sentence_length = 0
      word = []
      #tag = []
    else:
      #assert (len(line.split()) == 3)
      #if sentence_length==0:
      #  prevword=' '
      sentence_length += 1
      temp = model[line.split()[0]]
      assert len(temp) == word_dim
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      #temp = np.append(temp, capital(line.split()[0],prevword,wtitleIndex))  # adding capital embedding
      assert len(temp)==word_dim+10
      word.append(temp)
      #prevword = line.split()[0]
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
  print len(sentence)
  pkl.dump(sentence, open(output_embed, 'wb'))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str, help='data file', required=True)
  parser.add_argument('--data_tag', type=str, help='data tag', required=True)
  parser.add_argument('--train', type=str, help='train file location', required=True)
  parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
  parser.add_argument('--use_model', type=str, help='model location', required=True)
  parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
  args = parser.parse_args()
  max_sentence_length = find_max_length(args.train)
  print max_sentence_length
  print 'start to load wtitleReverseIndex'
  start_time = time.time()
  wtitleIndex = pkl.load(open('data/wtitleReverseIndex.p','rb')) 
  
  print 'load data cost time:',time.time()-start_time
  
  print 'start to load word2vec'
  start_time = time.time()
  trained_model = pkl.load(open(args.use_model, 'rb'))
  print 'load data cost time:',time.time()-start_time
  #print trained_model.wvec_model.vocab
  get_input(trained_model,wtitleIndex, args.model_dim, args.train, 
            args.dir_path+'/features/'+args.data_tag+'_embed.p'+str(args.model_dim),
            sentence_length=args.sentence_length)
