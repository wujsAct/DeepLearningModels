# -*- coding: utf-8 -*-
'''
@editor: wujs
function: generate entity mention using NER model results
revise: 2017/1/10
'''
import cPickle
import numpy as np
import codecs

dir_path = 'data/ace/'
class_size = 5

def getNERTag(retlist):
  retlist = np.asarray(retlist)
  index = retlist.argmax()
  ret = np.zeros([class_size],dtype=np.int32)
  ret[index]=1
  ret = map(str, ret)
  return ''.join(ret)

'''load sentid2aNosNoid'''
sentid2aNosNoid = {}
sentid = 0
with open(dir_path+'sentid2aNosNoid.txt','r') as file:
  for line in file:
    line = line.strip()
    sentid2aNosNoid[sentid] = line
    sentid += 1

'''load the standard entity recognition result'''
AFP_dict = {}

standard_entment = {}
standard_entment_name={}
with codecs.open(dir_path+'entMen2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    items = line.split(u'\t')
    #print items
    ids=items[2]+'\t'+items[3]#+'\t'+items[4]
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
#right = 0
#entMent=[]
sents = aceShape[0]; seqLent = 124;
#for i in xrange(sents):
#  for j in xrange(seqLent):
#    ret = getNERTag(ace_NERresult[i][j])
#    item = [sentid2aNosNoid[i],j,tag_dict[ret]]
#    afp_ids = item[0]+'\t'+str(item[1])
#    if afp_ids in AFP_dict:
#      print item,ace_NERresult[i][j]    
#    if ret in tag_dict and ret !='00001':
#      t1 = j -1
#      if len(entMent) == 0:
#        entMent.append(item)
#      else:
#        if item[0]==entMent[-1][0] and t1==entMent[-1][1] and item[2]==entMent[-1][2]:
#          entMent.append(item)
#        else:
#          ids = entMent[0][0]+'\t'+str(entMent[0][1])#+'\t'+str(entMent[-1][1]+1)
#          if ids in AFP_dict:
#            print 'AFP:',item
#          if ids in standard_entment:
#            standard_entment[ids] = 1
#            right += 1
#          #print entMent[0][0],'\t',entMent[0][1],'\t',entMent[-1][1]+1,'\t',entMent[0][2]
#          entMent=[item]
#for key in standard_entment:
#  if standard_entment[key]==0:
#    print standard_entment_name[key]
#print len(standard_entment)
#print right*1.0
#print right*1.0/len(standard_entment)
right = 0
for i in xrange(sents):
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