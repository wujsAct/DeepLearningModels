import sys
import numpy as np
import cPickle
import argparse
import time
import gzip
from description_embed_model import WordVec,MyCorpus

dir_path = 'data/figer/'
def figerData():
  input_file_obj = open(dir_path+'figerData.txt')
  sentence = []
  for line in input_file_obj:
    if line in ['\n']:
      yield sentence
      sentence=[]
    else:
      line = line.strip()
      word = line.split('\t')[0].lower()
      sentence.append(word)



print 'start to laod models...'
stime =time.time()
use_model = '/home/wjs/demo/entityType/informationExtract/data/wordvec_model_100.p'
print 'cost times:',time.time()-stime

print 'before vocab size:',len(use_model.wvec_model.vocab)

trained_model = cPickle.load(open(use_model, 'rb'))
trained_model.wvec_model.train(sentences=figerData())  #continue to train the datasets

print 'after vocab size:',len(use_model.wvec_model.vocab)             
cPickle.dump(trained_model, open(dir_path+'wordvec_model_' + str(100) + '.p', 'wb'))