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

dir_path = 'data/ace/'
class_size = 5

def getSents():
  input_file = dir_path+'aceData.txt'
  sents=[];senti =[]; chunki =[];
  for line in codecs.open(input_file,'r','utf-8'):
    if line in [u'\n', u'\r\n']:
      sents.append([senti,chunki])
      senti=[];chunki=[]
    else:
      line = line.strip()
      items = line.split('\t')
      senti.append(items[0]); chunki.append(items[2])
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
one-hot 特征啦
PER [1,0,0,0,0]; LOC [0, 1, 0, 0, 0] ; ORG [0, 0, 1, 0, 0]; MISC [0, 0, 0, 1, 0];
'''
tag_dict = {'10000':'PER','01000':'LOC','00100':'ORG','00010':'MISC','00001':'o'}
ace_NERresult = cPickle.load(open(dir_path+'features/ace_NERresult.p'))

aceShape = np.shape(ace_NERresult)

'''
we need to merge the adjacent same type 
'''
sentlent = aceShape[0]; seqLent = 124;

right = 0
for i in xrange(sentlent):
  for j in xrange(seqLent):
    ids = sentid2aNosNoid[i]+'\t'+str(j)
    if ids in standard_entment:
      if getNERTag(ace_NERresult[i][j])!='00001':
        right+=1
      else:
        print ids,standard_entment_name[ids],ace_NERresult[i][j]#getNERTag(ace_NERresult[i][j])
print len(standard_entment)
print right
print right*1.0/len(standard_entment)

'''
@这种entity recognition的方法啦！假如已经全部处理完毕了哈
'''

'''
entity mentions!
'''
sentid_entmention=collections.defaultdict(list)

right = 0
for key in standard_entment:
  i = aNosNoid2sentid[key.split('\t')[0]]
  j1 = int(key.split('\t')[1])
  j2 = int(key.split('\t')[2])
  print i,'\t',j1,'\t',j2
  temp = 0
  for j in xrange(j1,j2):  
    tt = standard_entment_name[key].replace("'"," ").replace("."," ")
    #print tt.split(" ")
   
    if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j]:
    #or tt.split(" ")[j-j1]==u'of' or tt.split(" ")[j-j1]==u'and':
      temp +=1
      
  if temp == j2-j1:
    right += 1
    sentid_entmention[i].append([j1,j2,' '.join(sents[i][0][j1:j2])])  #代表所有的实体啦！
  else:
    print 'ents:',key,standard_entment_name[key],'\t',temp,j1,'\t',j2
print right
print len(sentid_entmention)

ent_mention_index = [] 
for i in xrange(sentlent):
  ent_index = []
  if i in sentid_entmention:
    ent_index = sentid_entmention[i]
  ent_mention_index.append(ent_index)
  
print sentlent
print len(ent_mention_index)
print ent_mention_index
cPickle.dump(ent_mention_index,open(dir_path+'features/ent_mention_index.p','wb'))
