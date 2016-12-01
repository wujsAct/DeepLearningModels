from __future__ import print_function

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord

import numpy as np
import pickle as pkl
import cPickle as cpkl
import argparse
from wordvec_model import WordVec
from glove_model import GloveVec
from rnnvec_model import RnnVec
import codecs


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


def get_input(model, word_dim, input_file, output_embed, output_tag,sents2id,ents,tags, sentence_length=-1):
    print('processing %s' % input_file)
    word = []
    tag = []
    sent=[]
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
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 5))
                temp = np.array([0 for _ in range(word_dim + 11)])
                word.append(temp)
            
            senti = u' '.join(sent)
            if senti in sents2id:
                ids = sents2id[senti]
                entm = ents[ids][0]
                
            sentence.append(word)
            sentence_tag.append(np.array(tag))
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
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
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
    pkl.dump(sentence, open(output_embed, 'wb'))
    pkl.dump(sentence_tag, open(output_tag, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, help='data file', required=True)
    parser.add_argument('--data', type=str, help='all raw data e.g. entity mentions(train.p)', required=True)
    parser.add_argument('--train', type=str, help='train file location', required=True)
    parser.add_argument('--test_a', type=str, help='test_a file location', required=True)
    parser.add_argument('--test_b', type=str, help='test_b location', required=True)
    parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
    parser.add_argument('--use_model', type=str, help='model location', required=True)
    parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
    
    args = parser.parse_args()
    data = cpkl.load(open(args.data,'r'))
    aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['sents']; ents=data['ents'];tags=data['tags']
    sents2id = {sent:i for i,sent in enumerate(sents)}
    
    trained_model = pkl.load(open(args.use_model, 'rb'))
    get_input(trained_model, args.model_dim, args.train, args.dir_path+'/train_embed.p', args.dir_path+'/train_tag.p',
              sents2id,ents,tags,
              sentence_length=args.sentence_length)
    get_input(trained_model, args.model_dim, args.test_a, args.dir_path+'/test_a_embed.p', args.dir_path+'/test_a_tag.p',
              sents2id,ents,tags,
              sentence_length=args.sentence_length)
    get_input(trained_model, args.model_dim, args.test_b, args.dir_path+'/test_b_embed.p', args.dir_path+'/test_b_tag.p',
              sents2id,ents,tags,
              sentence_length=args.sentence_length)
