# -*- coding: utf-8 -*-
'''
@editor: wujs
function: generate entity mention using NER model results
revise: 2017/1/10
'''
import cPickle
import numpy as np
import codecs
import time
import argparse
import re
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
data_args = parser.parse_args()

data_tag = data_args.data_tag
dir_path = data_args.dir_path

if data_tag =='ace':
  all_standardlEnts = 306
elif data_tag == 'msnbc':
  all_standardlEnts = 739
class_size = 5
start_time = time.time()
#wtitle2ments = cPickle.load(open('data/wtitleReverseIndex.p'))
print 'end time:',time.time()-start_time


def getSents():
  input_file = dir_path+data_tag+'Data.txt'
  sents=[];senti =[]; chunki =[];posi=[];
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:
      sents.append([senti,chunki,posi])
      senti=[];chunki=[];posi=[]
    else:
      line = line.strip()
      items = line.split('\t')
      senti.append(items[0]); chunki.append(items[2]);posi.append(items[1])
  return sents

def getNERTag(index):
  ret = np.zeros([class_size],dtype=np.int32)
  ret[index]=1
  ret = map(str, ret)
  return ''.join(ret)

'''
continues 0,1,2,3 is the right entity mentions!
'''
def getNERIndex(pred):
  pred = ''.join(map(str,list(pred)))
  ret = []
  #for i in range(4):
    #classType = r''+str(i)+r'+'
  classType = r'01*'   #greed matching, find the longest substring.
  pattern = re.compile(classType)

  matchList = re.finditer(pattern,pred)  
  for match in matchList:
    ret.append(str(match.start())+'\t'+str(match.end()))
  return ret
  

'''load sentid2aNosNoid'''

sents = getSents()
sentid2aNosNoid = {}
sentid = 0
with open(dir_path+'sentid2aNosNoid.txt','r') as file:
  for line in file:
    line = line.strip()
    sentid2aNosNoid[sentid] = line
    sentid += 1
aNosNoid2sentid ={}
for key,value in sentid2aNosNoid.items():
  aNosNoid2sentid[value]=key


sentlent = len(sentid2aNosNoid)

'''load the standard entity recognition result'''
AFP_dict = {}

standard_entment = {}
standard_entment_name={}
non_link_ents = {}
non_link_ents_nums = 0
print 'data tag:',data_tag
print 'dir path:',dir_path
allents =0
#if data_tag=='msnbc':
with codecs.open(dir_path+'new_entMen2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    #print line
    line = line.strip()
    items = line.split(u'\t')
    
    #print items
    #ids=items[2]+'\t'+items[3]#+'\t'+items[4]
     
    ids = items[2]+'\t'+items[3]+'\t'+items[4]
    
    if items[0] != items[len(items)-1]:
      non_link_ents[ids] = 1
      non_link_ents_nums += 1
      #print 'non linking ents:',line
    if items[0] == u'AFP':
      AFP_dict[ids] = 1
      
    standard_entment[ids]=0
    standard_entment_name[ids]=items[0]
    
    allents += 1
print 'all entmentions:',allents
print 'coreference ents:',len(non_link_ents), non_link_ents_nums

'''
@revise: 2017/7/1
@entity mention into line sequence
'''
wrong = 0
allEnt = 0
sentid_entmention=collections.defaultdict(list)
for key in standard_entment:
  i = aNosNoid2sentid[key.split('\t')[0]]
  j1 = int(key.split('\t')[1])
  j2 = int(key.split('\t')[2])
  
  if key  in non_link_ents:
    continue
  
  tt = standard_entment_name[key]
  
  strtemp = ' '.join(sents[i][0][j1:j2]).replace(' .','.')
  strtemp = strtemp.replace(' \'','\'')
  allEnt += 1
  if tt==strtemp:
    pass
  else:
    wrong += 1
  #print allEnt
  sentid_entmention[i].append([j1,j2,strtemp]) 

allEnts = 0
ent_mention_index = [] 
for i in xrange(sentlent):
  ent_index = []
  if i in sentid_entmention:
    ent_index = list(sentid_entmention[i])
  allEnts += len(ent_index)
  ent_mention_index.append(ent_index)

print 'allEnts',allEnts
print sentlent

cPickle.dump(ent_mention_index,open(dir_path+'features/ent_mention_index.p','wb'))