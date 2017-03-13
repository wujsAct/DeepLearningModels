# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:36:51 2017

@author: DELL
"""

import codecs
import collections
from tqdm import tqdm
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
data_args = parser.parse_args()

data_tag = data_args.data_tag
dir_path = data_args.dir_path
#ent_mention_index = cPickle.load(open(dir_path+'features/ent_mention_index.p','rb'))
#print ent_mention_index
#
#
#'''load sentid2aNosNoid'''
#
#sentid2aNosNoid = {}
#sentid = 0
#with open(dir_path+'sentid2aNosNoid.txt','r') as file:
#  for line in file:
#    line = line.strip()
#    sentid2aNosNoid[sentid] = line
#    sentid += 1
#aNosNoid2sentid ={}
#for key,value in sentid2aNosNoid.items():
#  aNosNoid2sentid[value]=key
                 
                 

'''read mid2name'''
fname = 'data/mid2name.tsv'
wikititle2fb = collections.defaultdict(list)
fb2wikititle={}
with codecs.open(fname,'r','utf-8') as file:
  for line in tqdm(file):
    line = line.strip()
    items = line.split('\t')
    if len(items)==2:
      fbId = items[0]; title = items[1]  
      fb2wikititle[fbId] = title.lower()
      wikititle2fb[title.lower()].append(fbId)
fb2wikititle['NIL'] = 'NIL'        
    
'''read entmention 2 aNosNoid'''
if data_tag == 'msnbc':
  entsFile = dir_path+'new_entMen2aNosNoid.txt'
else:
  entsFile = dir_path+'entMen2aNosNoid.txt'

hasMid = 0
entMentsTags={}
entMents2surfaceName={}
with codecs.open(entsFile,'r','utf-8') as file:
  for line in file:
    line = line.strip()
    items = line.split('\t')
    entMent = items[0]; linkingEnt = items[1].lower(); aNosNo = items[2]; start = items[3]; end = items[4]; repEntStr=items[5]
    
    key = aNosNo + '\t' + start+'\t'+end
    if linkingEnt == 'NIL':
      hasMid += 1
      entMentsTags[key]='NIL'
    

    if linkingEnt in wikititle2fb:
      #print wikititle2fb[linkingEnt]
      hasMid +=1 
      entMentsTags[key] =wikititle2fb[linkingEnt]
      entMents2surfaceName[key] = entMent
    else:
      entMentsTags[key]='NIL'
      
print 'entMentsTags nums:',len(entMentsTags)
cPickle.dump(entMentsTags,open(dir_path+data_tag+'_entMentsTags.p','wb'))