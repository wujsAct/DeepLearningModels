#from __future__ import print_function
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from RandomVec import RandomVec
import pickle as pkl
import argparse
import codecs
from tqdm import *
class WordVec:
  def __init__(self, args):
    print('processing corpus')
    if args.restore is None:
      sentences=[]
      with codecs.open(args.corpus,'r','utf-8') as file:
        for line in tqdm(file):
          line = line.strip().lower()
          sentence = line.split(u' ')
          #print(sentence)
          sentences.append(sentence)
      #bigram_transformer = Phrases(sentences)
      #print(bigram_transformer[sentences])
      print('start to train word2vec embeddings')
      self.wvec_model = Word2Vec(sentences=sentences, size=args.dimension, window=args.window,
                                 workers=args.workers,
                                 sg=args.sg,
                                 batch_words=args.batch_size, min_count=1
                                 #max_vocab_size=args.vocab_size
                                 )
    else:
      self.wvec_model = Word2Vec.load_word2vec_format(args.restore, binary=True)
    self.rand_model = RandomVec(args.dimension)

  def __getitem__(self, word):
    word = word.lower()
    try:
      return self.wvec_model[word]
    except KeyError:
      print(word, 'is random initialize the words')
      return self.rand_model[word]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str, help='data file', required=True)
  parser.add_argument('--corpus', type=str, help='corpus location', required=True)
  parser.add_argument('--dimension', type=int, help='vector dimension', required=True)
  parser.add_argument('--window', type=int, default=5, help='window size')
  #parser.add_argument('--vocab_size', type=int, help='vocabulary size', required=True)
  parser.add_argument('--workers', type=int, default=3, help='number of threads')
  parser.add_argument('--sg', type=int, default=1, help='if skipgram 1 if cbow 0')
  parser.add_argument('--batch_size', type=int, default=10000, help='batch size of training')
  parser.add_argument('--restore', type=str, default=None, help='word2vec format save')
  args = parser.parse_args()
  model = WordVec(args)
  pkl.dump(model, open(args.dir_path+'/wordvec_model_' + str(args.dimension) + '.p', 'wb'))
