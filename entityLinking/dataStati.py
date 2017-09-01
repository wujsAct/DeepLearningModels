# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:12:03 2017

@author: DELL
"""
import cPickle
import codecs
from collections import defaultdict
import numpy as np

#dir_path = 'data/kbp/2014/evaluation/'
#data_tag = 'kbp'
#candidateEntNum = 90

dir_path ='data/msnbc/'
data_tag = 'msnbc'
candidateEntNum = 90

feature_path =dir_path + 'features/'+str(candidateEntNum)+'/'
#test_entliking= cPickle.load(open(feature_path+data_tag+'_ent_linking.p'+str(candidateEntNum),'rb'))

'''
ent_Mentions = cPickle.load(open(dir_path+'features/ent_mention_index.p'))
entMentsTags = cPickle.load(open(dir_path+data_tag+'_entMentsTags.p','rb'))
print len(entMentsTags)
NERentCands = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_ent_cand_mid_new.p'+str(candidateEntNum)))  #[300,candsNums]


cand_list=[]
#read sents2aNosNo
sentid = 0
sentId2aNosNo = {}
with codecs.open(dir_path+'sentid2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    sentId2aNosNo[sentid] = line
    sentid += 1

cand_2_linkTag = defaultdict(list)
ent_doc = defaultdict(set)
cand_doc = defaultdict(set)
cand_pair_doc = defaultdict(int)
NILnums = 0
aNo2mentNums = defaultdict(int)

entids = -1
for i in range(len(ent_Mentions)):
  aNosNo = sentId2aNosNo[i]
  aNo = aNosNo.split('_')[0]
    
  #print aNosNo
  ents = ent_Mentions[i]
  
  
  
  for j in range(len(ents)):
    entids += 1

    entName =ents[j][2]

    ent_doc[entName].add(i)


    sindex = ents[j][0];eindex = ents[j][1]

    key = aNosNo+'\t'+str(sindex)+'\t'+str(eindex)

    linkTag = entMentsTags[key]   #get entity mention linking mid!
    
    if linkTag !='NIL':
      aNo2mentNums[aNo] += 1
      linkTag = linkTag[0]
    else:
      NILnums += 1
      
    #print linkTag

    candMids = NERentCands[entids].keys()
    cand_list.append(len(candMids))
    
    k_entids = entids+1
    for k in range(j+1,len(ents)):
      k_candMids = NERentCands[k_entids].keys()

      for candj in candMids:
        for candk in k_candMids:
          key_candPair = candj +"_"+candk
          key_candPair1 = candk +"_"+candj

          if key_candPair1 in cand_pair_doc:
            cand_pair_doc[key_candPair1] +=1
          elif key_candPair in cand_pair_doc:
            cand_pair_doc[key_candPair] +=1
          else:
            cand_pair_doc[key_candPair] +=1

      k_entids += 1


    if linkTag!='NIL':
      for candi in candMids:
        if candi == linkTag:
          cand_2_linkTag[candi].append([linkTag,True])
        else:
          cand_2_linkTag[candi].append([linkTag,False])
        cand_doc[candi].add(i)


ent_docNum =[]
cand_docNum = []
cand_pair_docNum=[]
for key in ent_doc:
  ent_docNum.append(len(ent_doc[key]))

for key in cand_doc:
  cand_docNum.append(len(cand_doc[key]))

for key in cand_pair_doc:
   cand_pair_docNum.append(cand_pair_doc[key])

print 'NILnums:',NILnums
print np.average(cand_list)
print np.max(cand_list)
print '-----------------'


aNoEntNums =[]
for key in aNo2mentNums:
  aNoEntNums.append(aNo2mentNums[key])
  
print data_tag, np.average(aNoEntNums), np.max(aNoEntNums)

#print ent_docNum
print np.average(ent_docNum)
print np.average(cand_docNum)
print np.average(cand_pair_docNum)

params={'ent_doc':ent_doc,'cand_doc':cand_doc,'cand_2_linkTag':cand_2_linkTag}

cPickle.dump(params,open(feature_path+data_tag+'dataStatisc.p','wb'))

'''
def candStatic(testa_cands,train_cands):
  isRight = 0
  all_testa_cands_1=0
  for key in testa_cands:
    all_testa_cands_1 += len(testa_cands[key])
    for testa_items in testa_cands[key]:
      testa_linking_tag,testa_isRight = testa_items
      if testa_isRight:
        isRight += 1
    
  print 'test:',all_testa_cands_1,isRight
  
  
  #for key in testa_cand_doc:
  #  all_testa_cands_2 += len(testa_cand_doc[key])
  #print all_testa_cands_1
  #print all_testa_cands_2
  
  testa_right_in_train_right=0
  testa_wrong_in_train_right=0
  testa_in_train_right = 0
  
  testa_right_in_train_wrong=0
  testa_wrong_in_train_wrong=0
  testa_in_train_wrong = 0
  
  
  testa_right_not_in_train=0
  testa_wrong_not_in_train=0
  testa_not_in_train =0
  for key in testa_cands:
    for testa_items in testa_cands[key]:
      testa_linking_tag,testa_isRight = testa_items
      is_in_train_right = False
      if key in train_cands:
        for items in train_cands[key]:
          train_linking_tag,train_isRight = items
          
          if train_isRight:
            is_in_train_right = True
            
        if is_in_train_right:
          if testa_isRight:
            testa_right_in_train_right += 1
          else:
            testa_wrong_in_train_right += 1
          testa_in_train_right += 1
        else:
          if testa_isRight:
            testa_right_in_train_wrong += 1
          else:
            testa_wrong_in_train_wrong+= 1
          testa_in_train_wrong += 1
        
      else:
        if testa_isRight:
          testa_right_not_in_train += 1
        else:
          testa_wrong_not_in_train += 1
        testa_not_in_train += 1
      
      
      
    

  print testa_right_in_train_right,testa_wrong_in_train_right,testa_in_train_right
  print testa_right_in_train_wrong,testa_wrong_in_train_wrong,testa_in_train_wrong
  print testa_right_not_in_train,testa_wrong_not_in_train,testa_not_in_train
  print testa_in_train_right+testa_in_train_wrong+testa_not_in_train


dir_path = 'data/kbp/2014/'
testa_params  = cPickle.load(open(dir_path+'evaluation/features/50/'+'kbp'+'dataStatisc.p','rb'))
train_params = cPickle.load(open(dir_path+'training/features/50/'+'kbp'+'dataStatisc.p','rb'))


testa_ent_doc = testa_params['ent_doc']; testa_cand_doc = testa_params['cand_doc'];testa_cands = testa_params['cand_2_linkTag']
train_ent_doc = train_params['ent_doc']; train_cand_doc = train_params['cand_doc'];train_cands = train_params['cand_2_linkTag']


train_cands_nums=0
isRight =0
for key in train_cands:
  train_cands_nums += len(train_cands[key])
  
  for items in train_cands[key]:
    train_linking_tag,train_isRight = items
    if train_isRight:
      isRight += 1
print 'train_cands_nums:',train_cands_nums,isRight



candStatic(testa_cands,train_cands)

#testa_ent_in_train = 0
#testa_cand_in_train =0
#for key in testa_ent_doc:
#  if key in train_ent_doc:
#    testa_ent_in_train += 1
#
#
#for key in testa_cand_doc:
#  if key in train_cand_doc:
#    testa_cand_in_train += 1
#    
#print 'testa cand:',testa_cand_in_train,len(testa_cand_doc),testa_cand_in_train*1.0/len(testa_cand_doc)
#
#print 'testa ent:',testa_ent_in_train,len(testa_ent_doc),testa_ent_in_train*1.0/len(testa_ent_doc)
