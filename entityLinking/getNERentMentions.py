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
if data_tag=='msmbc':
  with codecs.open(dir_path+'new_entMen2aNosNoid.txt','r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split(u'\t')
      
      #print items
      #ids=items[2]+'\t'+items[3]#+'\t'+items[4]
      ids = items[2]+'\t'+items[3]+'\t'+items[4]
      if items[0] != items[len(items)-1]:
        non_link_ents[ids] = 1
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
one-hot 特征啦
PER [1,0,0,0,0]; LOC [0, 1, 0, 0, 0] ; ORG [0, 0, 1, 0, 0]; MISC [0, 0, 0, 1, 0];
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
aNotag = -1
fret = codecs.open(dir_path+"eng.ace",'w','utf-8')
right = 0
for i in xrange(sentlent):
  for j in xrange(seqLent):
    aNosNoid = sentid2aNosNoid[i]
    aNo = aNosNoid.split('_')[0]
    if(int(aNo)!=aNotag):
      fret.write('-DOCSTART- -X- -X- O'+'\n\n')
      aNotag += 1
    if(j<len(sents[i][2])):
      mtag= getNERTag(ace_NERresult[i][j])
      fret.write(sents[i][0][j]+' '+sents[i][2][j]+' '+sents[i][1][j]+' '+tag_dict[mtag]+'\n')
  fret.write('\n')
fret.close()
  #print ids,standard_entment_name[ids],ace_NERresult[i][j]#getNERTag(ace_NERresult[i][j])
'''

'''
@an reasonable parts! 
'''
'''
for i in xrange(sentlent):
  for j in xrange(seqLent):
    ids = sentid2aNosNoid[i]+'\t'+str(j)
    
    if getNERTag(ace_NERresult[i][j])!='00001':
      print 'sent:',i,j,sents[i][0][j],sents[i][1][j],getNERTag(ace_NERresult[i][j])
    else:
      if j< len(sents[i][1]):
        if 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j]:
          if j >= 1:
            twogram = sents[i][0][j-1].lower()+' '+sents[i][0][j].lower() #:
            flaggram = False
            for keyi in wtitle2ments[sents[i][0][j-1].lower()]:
              if twogram in keyi:
                flaggram = True
              
            twogram1 = sents[i][0][j-1].lower()+''+sents[i][0][j].lower()
            for keyi in wtitle2ments[sents[i][0][j-1].lower()]:
              if twogram1 in keyi:
                flaggram = True
            if flaggram:
              print 'sent:',i,j,sents[i][0][j],sents[i][1][j],getNERTag(ace_NERresult[i][j-1])
          else:
            if sents[i][0][j].lower() in wtitle2ments:
              print 'sent:',i,j,sents[i][0][j],sents[i][1][j],getNERTag(ace_NERresult[i][j])
        else:
          print 'sent:',i,j,sents[i][0][j],sents[i][1][j],getNERTag(ace_NERresult[i][j])
'''           
'''
@we assume entity recognition has done
'''

'''
entity mentions!
'''

print len(non_link_ents)
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
    tt = standard_entment_name[key].replace("'"," ").replace("."," ")
    #print tt.split(" ")
   
    #if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j]:
    #if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j]: #or 'PP' in sents[i][1][j]:
    #or tt.split(" ")[j-j1]==u'of' or tt.split(" ")[j-j1]==u'and':
    #if getNERTag(ace_NERresult[i][j])!='00001' or 'NP' in sents[i][1][j] or 'O' in sents[i][1][j]: #or 'PP' in sents[i][1][j]:
    #  temp +=1
    
      
    if getNERTag(NERresult[i][j])!='00001':
      temp +=1
      tempj.append([j,getNERTag(NERresult[i][j])])
      
    else:
      if 'VP' in sents[i][1][j] or 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j] or 'O' in sents[i][1][j]:
        tempj.append([j,getNERTag(NERresult[i][j])])
        temp+=1
      
    '''
    else:
      tempj.append([j,getNERTag(ace_NERresult[i][j])])
      if 'NP' in sents[i][1][j] or 'PP' in sents[i][1][j] or 'O' in sents[i][1][j]:
        if j-j1>=1:
          twogram = sents[i][0][j-1].lower()+' '+sents[i][0][j].lower() #:
          flaggram = False
          for keyi in wtitle2ments[sents[i][0][j-1].lower()]:
            if twogram in keyi:
              flaggram = True
            
          twogram1 = sents[i][0][j-1].lower()+sents[i][0][j].lower()
          for keyi in wtitle2ments[sents[i][0][j-1].lower()]:
            if twogram1 in keyi:
              flaggram = True
          if flaggram:
            temp+=1
        else:
          if sents[i][0][j].lower() in wtitle2ments:
            temp += 1
    '''
      
  if temp >= j2-j1:
    right += 1
    sentid_entmention[i].append([j1,j2,' '.join(sents[i][0][j1:j2])])  #stands for all mentions!
  else:
    print 'ents:',key,standard_entment_name[key],'\t',temp,j1,'\t',j2,sents[i][0][j1:j2],sents[i][1][j1:j2]
    print tempj
    print '-------------------------'
print 'all mentions:',allents
print 'right mentions:',right
print 'all sents:',len(sentid_entmention)

ent_mention_index = [] 
for i in xrange(sentlent):
  ent_index = []
  if i in sentid_entmention:
    ent_index = sentid_entmention[i]
  ent_mention_index.append(ent_index)
  
#print sentlent
print len(ent_mention_index)
#print ent_mention_index

cPickle.dump(ent_mention_index,open(dir_path+'features/ent_mention_index.p','wb'))
