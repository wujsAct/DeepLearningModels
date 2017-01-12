# -*- coding: utf-8 -*-
'''
@time: 2016/12/17--19
@editor: wujs
@function: generate word2vec feature for entity words
ï¿½ï¿½ï¿½ï¿½feed ï¿½ï¿½word2vecÖ®Ç°ï¿½ï¿½È«ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿½Ó¦ï¿½Ã¸ï¿½ï¿½ï¿½ï¿½Ë¡ï¿½ï¿½ï¿½È»ï¿½áµ¼ï¿½ï¿½Ò»ï¿½ï¿½Ñµï¿½ï¿½ï¿½ï¿½Ò»ï¿½ß´ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ý£ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä£ï¿½
'''
#encoding utf-8

import os
from tqdm import *
from gensim.models.word2vec import Word2Vec
from random_vec import RandomVec
import pickle as pkl
import argparse
import codecs
from glob import glob
import linecache
from spacy.en import English
nlp = English()
randvec = pkl.load(open('/home/wjs/demo/entityType/informationExtract/data/randomvec.p'))

class MyCorpus:
  def __init__(self,args):
    print('processing corpus')
    self.fname = args.dir_path+args.fname
    self.outfname= args.dir_path+args.fname+'_out'
    self.outfname_s2= args.dir_path+args.fname+'_out1'

  def __iter__(self):
    try:
      with codecs.open(self.outfname_s2,'r','utf-8') as file:
        for line in tqdm(file):
          sent = line.strip().lower()
          yield sent.split(u'\t')
    except:
      print(" [!] Error occured for %s" % self.outfname_s2)
  '''
  def preprocess(self):
    fout = codecs.open(self.outfname,'w','utf-8')
    with codecs.open(self.fname,'r','utf-8') as file:
      for line in tqdm(file):
        descriptions = line.split(u'\t')[1]
        strs = descriptions.split(u'@en')[0]

        strs = strs.replace(u'\\n',u' ')
        fout.write(strs+u'\n')
    fout.close()
  '''
  #we do not need to delete symbol, because those can effect depnedecy semantic!
  def preprocess_s2(self):
    fout = codecs.open(self.outfname_s2,'w','utf-8')
    texts = codecs.open(self.outfname,'r','utf-8')
    no = 0
    for doc in nlp.pipe(texts):
      for sentence in doc.sents:
        no +=1
        for token in sentence:
          #if token.pos_ !='PUNCT' and token.pos_ != 'SPACE':
          fout.write(token.text.lower()+u'\t')
        fout.write(u'\n')
        if no%10000==0:
          print no

class WordVec:
  def __init__(self, args):
    self.restore = args.restore
    self.corpuss=MyCorpus(args)
    
  def __train__(self,):
    if self.restore == None:
      print('start to train word2vec models ... ')
      self.wvec_model = Word2Vec(sentences=self.corpuss, size=args.dimension, window=args.window,
                                 workers=args.workers,
                                 sg=args.sg,
                                 batch_words=args.batch_size, min_count=3#, max_vocab_size=args.vocab_size
                                 )
    else:
      self.wvec_model = Word2Vec.load_word2vec_format(args.restore, binary=True)
    #self.rand_model = RandomVec(args.dimension)
    '''
    @revise 2017/1/12 ËùÓÐµÄÎÞ·¨embeddingµÄword¶¼±»embeddingµ½'None'Õâ¸öÊµÌåÉÏÈ¥£¡
    '''
    #self.randvec = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/randomvec.p'))
  def __getitem__(self, word):
    word = word.lower()
    try:
      return self.wvec_model[word]
    except KeyError:
      print(word, 'is random initialize the words')
      return randvec


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str,default='/home/wjs/demo/entityType/informationExtract/data/',help='data file')
  parser.add_argument('--fname', type=str, default='mid2description.txt',help='corpus location')
  parser.add_argument('--dimension', type=int,default=100, help='vector dimension')
  parser.add_argument('--window', type=int, default=5, help='window size')
  #parser.add_argument('--vocab_size', type=int,default=50000, help='vocabulary size', required=True)
  parser.add_argument('--workers', type=int, default=4, help='number of threads')
  parser.add_argument('--sg', type=int, default=1, help='if skipgram 1 if cbow 0')
  parser.add_argument('--batch_size', type=int, default=80000, help='batch size of training')
  parser.add_argument('--restore', type=str, default=None, help='word2vec format save')
  args = parser.parse_args()
  corpuss=MyCorpus(args)
  #corpuss.preprocess()
  corpuss.preprocess_s2()
  #corpuss.__iter__()
  #sentence_split_by_spacy(strs)
  #ï¿½ï¿½ï¿½ï¿½ï¿½È½ï¿½ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿½ï¿½É£ï¿½

  model = WordVec(args)
  model.__train__()
  print('save the models...')
  pkl.dump(model, open(args.dir_path+'wordvec_model_' + str(args.dimension) + '.p', 'wb'))
  #model = pkl.load(open(args.dir_path+'wordvec_model_' + str(args.dimension) + '.p', 'rb'))
  #print len(model.wvec_model.vocab)
