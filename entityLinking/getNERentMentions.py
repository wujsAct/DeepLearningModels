# -*- coding: utf-8 -*-
'''
@editor: wujs
function: generate entity mention using NER model results
revise: 2017/1/10
'''
import cPickle
import numpy as np
import codecs
import collections
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
  
data_args = parser.parse_args()

data_tag = data_args.data_tag
dir_path = data_args.dir_path

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

def getNERTag(retlist):
  retlist = np.asarray(retlist)
  index = retlist.argmax()
  ret = np.zeros([class_size],dtype=np.int32)
  ret[index]=1
  ret = map(str, ret)
  return ''.join(ret)

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


'''load the standard entity recognition result'''
AFP_dict = {}

standard_entment = {}
standard_entment_name={}
non_link_ents = {}
print 'data tag:',data_tag
print 'dir path:',dir_path
if data_tag=='msnbc':
  with codecs.open(dir_path+'new_entMen2aNosNoid.txt','r','utf-8') as file:
    for line in file:
      print line
      line = line.strip()
      items = line.split(u'\t')
      
      #print items
      #ids=items[2]+'\t'+items[3]#+'\t'+items[4]
      ids = items[2]+'\t'+items[3]+'\t'+items[4]
      '''
      @f
      '''
      if items[0] != items[len(items)-1]:
        non_link_ents[ids] = 1
        print 'non linking ents:',line
      if items[0] == u'AFP':
        AFP_dict[ids] = 1
        
      standard_entment[ids]=0
      standard_entment_name[ids]=items[0]
else:
  with codecs.open(dir_path+'entMen2aNosNoid.txt','r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split(u'\t')
      #print items
      #ids=items[2]+'\t'+items[3]#+'\t'+items[4]
      ids = items[2]+'\t'+items[3]+'\t'+items[4]
      if items[0] == u'AFP':
        AFP_dict[ids] = 1
        
      standard_entment[ids]=0
      standard_entment_name[ids]=items[0]
print AFP_dict

'''
generate entity mention using ace_NERresult.p
one-hot ç¹å¾å?PER [1,0,0,0,0]; LOC [0, 1, 0, 0, 0] ; ORG [0, 0, 1, 0, 0]; MISC [0, 0, 0, 1, 0];
'''
tag_dict = {'10000':'I-PER','01000':'I-LOC','00100':'I-ORG','00010':'I-MISC','00001':'O'}
NERresult = cPickle.load(open(dir_path+'features/'+data_tag+'_NERresult.p'))

Shape = np.shape(NERresult)
print Shape

'''
we need to merge the adjacent same type 
'''
sentlent = Shape[0]; seqLent = 124;

'''
entity mentions!
'''

print 'non linking ents:',len(non_link_ents)
sentid_entmention=collections.defaultdict(list)
allents = 0
right = 0
for key in standard_entment:
  if key in non_link_ents:
    continue
  allents +=1
  i = aNosNoid2sentid[key.split('\t')[0]]
  j1 = int(key.split('\t')[1])
  j2 = int(key.split('\t')[2])
  #print i,'\t',j1,'\t',j2
  temp = 0
  tempj=[]
  for j in xrange(j1,j2):  
    tt = standard_entment_name[key]#.replace("'"," ")
    #print tt.split(" ")
   
    #if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j]:
    #if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j]: #or 'PP' in sents[i][1][j]:
    #or tt.split(" ")[j-j1]==u'of' or tt.split(" ")[j-j1]==u'and':
    #if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j] or 'O' in sents[i][1][j]: #or 'PP' in sents[i][1][j]:
    #  temp +=1
    
      
    if getNERTag(NERresult[i][j])!='00001':
#      if sents[i][0][j] == 'Ford':  #all predict person!
#        print getNERTag(NERresult[i][j])
      temp +=1
      tempj.append([j,getNERTag(NERresult[i][j])])
      
    else:
      if 'VP' in sents[i][1][j] or 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j] or 'O' in sents[i][1][j]:
        tempj.append([j,getNERTag(NERresult[i][j])])
        temp+=1
  if temp >= j2-j1:
    right += 1
    strtemp = ' '.join(sents[i][0][j1:j2]).replace(' .','.')
    strtemp = strtemp.replace(' \'','\'')
    sentid_entmention[i].append([j1,j2,strtemp])  #stands for all mentions!
   
    if strtemp!= tt:
      print key,strtemp, tt,standard_entment_name[key]
  else:
    print 'ents:',key,standard_entment_name[key],'\t',temp,j1,'\t',j2,sents[i][0][j1:j2],sents[i][1][j1:j2]
    print tempj
    print '-------------------------'
print 'all mentions:',allents
print 'right mentions:',right
print 'all sents:',len(sentid_entmention)
print 'aNosNoid2sentid:',len(aNosNoid2sentid)

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
print len(ent_mention_index)
#print ent_mention_index

cPickle.dump(ent_mention_index,open(dir_path+'features/ent_mention_index.p','wb'))
