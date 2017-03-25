# -*- coding: utf-8 -*-
'''
@editor: wujs
function: generate entity mention using NER model results
revise: 2017/1/11
'''

import cPickle
import argparse
import codecs
from TFRecordUtils import ner_d3array_TFRecord


def get_length(data_tag):
  print 'start to load datas'
  with codecs.open(dir_path+'process/'+data_tag+'.out','r','utf-8') as context:
     texts =context.read()
     sents = texts.split('\n\n')
     sent_length = []
     for senti in sents:
       sent_length.append(len(senti.split('\n')))

  cPickle.dump(sent_length,open(dir_path+data_tag+'_sentlength.p','wb'))
  return sent_length


parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
parser.add_argument('--model_dim', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)

data_args = parser.parse_args()
dir_path = data_args.dir_path
model_dim = data_args.model_dim
'''for train test, we need revise the tf record things! '''
sent_length = get_length('train')
output_embed = dir_path+'/features/train_embed.p'+str(model_dim)

print 'start to load train data sets...'
sentence = cPickle.load(open(dir_path+'/train_embed.p'+model_dim,'rb')) 
sentence_tag = cPickle.load(open(dir_path+'/train_tag.p'+model_dim,'rb'))

ner_d3array_TFRecord(sentence,sentence_tag,output_embed+'.tfrecords',output_embed+'.shape')
                                           
get_length('testa')
get_length('testb')