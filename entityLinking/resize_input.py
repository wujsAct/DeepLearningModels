from __future__ import print_function
import argparse
import cPickle
import os
import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
from PhraseRecord import EntRecord

def remove_crap(input_file):
    f = open(input_file)
    lines = f.readlines()
    l = list()
    for line in lines:
        if "-DOCSTART-" in line:
            pass
        else:
            l.append(line)
    ff = open('temp.txt', 'w')
    ff.writelines(l)
    ff.close()


def modify_data_size(output_file, trim,sents):
    final_list = list()
    l = list()
    temp_len = 0
    count = 0
    for line in open('temp.txt', 'r'):
        if line in ['\n', '\r\n']:
            if temp_len == 0:
                l = []
            elif temp_len > trim:
                #print(l)
                count += 1
                l = []
                temp_len = 0
            else:
                l.append(line)
                final_list.append(l)
                l = []
                temp_len = 0
        else:
            l.append(line)
            temp_len += 1
    f = open(output_file, 'w')
    #we need to filter the sentence contians non-ner! 
    non_ents = 0
    for i in final_list:
        senti = u' '.join(i)
        if ('LOC' in senti) or ('PER' in senti) or ('MISC' in senti) or ('ORG' in senti):
            f.writelines(i)
        else:
            non_ents = non_ents + 1
            print(i)
            print('wrong....')
    f.close()
    print('non entmentions sentences:%d' %(non_ents))
    print('%d sentences trimmed out of %d total sentences' % (count, len(final_list)))
    #os.system('rm temp.txt')


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file location', required=True)
parser.add_argument('--entMentInput', type=str, help='entity mention files such as:train.p', required=True)
parser.add_argument('--output', type=str, help='output file location', required=True)
parser.add_argument('--trim', type=int, help='trimmed sentence length', required=True)
args = parser.parse_args()
remove_crap(args.input)
data = cPickle.load(open(args.entMentInput,'r'))
sents=data['sents']
#print(sents)
modify_data_size(args.output, args.trim,sents)
