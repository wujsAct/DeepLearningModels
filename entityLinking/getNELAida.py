# -*- coding: utf-8 -*-
"""
@author wujs
time: 2017/1/23
"""
import sys
sys.path.append('/home/wjs/demo/entityType/informationExtract')
sys.path.append('/home/wjs/demo/entityType/informationExtract/utils')

import codecs
import collections
from tqdm import tqdm
import cPickle
from entityRecog import nameEntityRecognition,pp,flags,args
import numpy as np
from sklearn.metrics.pairwise  import cosine_similarity

import argparse
import time
candidateEntNum=90
print candidateEntNum

def printfeatuesThreshold(ent_id):
  all_candidate_mids = []
  allEnts=0.0; all_linkable_ents = 0.0
  passEnts=0; passEnts_right = 0.0
  rightEnts=0
  allRightEnts= 0
  noDefineRightEnts=0
  
  entids=0
  relentid = -1
  k  = -1
  rightWiki = 0
  rightFb=0
  rightWord2vec = 0
  disambiguation = []
  disambiguation_withoutThread=[]
  disambiguation_withoutDefine=[]
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    tagid=-1
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      k+=1
      candMids = NERentCands[k].keys()
      
      linkTag = entMentsTags[ent_id][1]   #get entity mention linking mid!
#      #print linkTag
#      enti_name = ents[j].content.lower()
      
      if linkTag=='NIL':
        passEnts+=1
        if len(candMids)==0:
          passEnts_right += 1
        #ent_id +=1
        #continue
      else:
        all_linkable_ents +=1
      
      ent_id += 1
    
      tagid +=1
     
      relentid += 1
      rel_cand_mid_dict={}
      oreder_cand_mid_dict={}
      
      wikicands = wiki_candidate_mids[k]
      FBcands = freebase_candidate_mids[k]
      if linkTag in candMids:
        noDefineRightEnts += 1
      disambiguation_withoutDefine.append(len(candMids))
  
#      if 'surrey' in enti_name:
#        print 'candMids:',len(candMids)
#        
      '''
      @2017/3/23 need to reserve the top 10 wikipedia, revise tomorrow!
      '''
      deleteCands = {}
      for key in NERentCands[k]:
        if NERentCands[k][key][1]==0 and key not in wikicands:
          candMids.remove(key)
          deleteCands[key]=NERentCands[k][key]  
#          print 'candMids:',len(candMids)
#          print linkTag,NERentCands[k][linkTag]
      #print len(candMids)
      disambiguation_withoutThread.append(len(candMids))
      if linkTag in candMids:
        allRightEnts += 1
      for key in candMids:
        if key in wikicands:
          rel_cand_mid_dict[key] = NERentCands[k][key]
        else:
          oreder_cand_mid_dict[key] = NERentCands[k][key]
      
      sorted_cand_mid_dict= sorted(oreder_cand_mid_dict.items(), key=lambda oreder_cand_mid_dict:oreder_cand_mid_dict[1])[0:max(0,candidateEntNum-len(rel_cand_mid_dict))]
      
      sorted_cand_mid_dict = dict(sorted_cand_mid_dict)
      
      dictMerged2=dict(rel_cand_mid_dict, **sorted_cand_mid_dict)
      
      disambiguation.append(len(dictMerged2))
      
      all_candidate_mids.append(dict(dictMerged2))  
      if linkTag in dictMerged2:
        rightEnts+=1
        if linkTag in wikicands:
          rightWiki += 1
        if linkTag in FBcands:
          rightFb+= 1
#        if linkTag not in FBcands and linkTag not in wikicands:
#          rightWord2vec += 1
        if NERentCands[k][linkTag][1] !=0:
          rightWord2vec += 1
  
  cPickle.dump(all_candidate_mids,open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_ent_cand_mid_new.p'+str(candidateEntNum),'wb'))
  #print len(set(candMids))

  print 'data tags:',data_tag
  print 'disambiguation:',len(disambiguation)
  print 'all_candidate_mids:',len(all_candidate_mids)
  print 'datatag',data_tag
  print 'allRightEnts:',allRightEnts
  print 'right:',rightEnts
  print 'pass:',passEnts, ' passEnts_right:',passEnts_right
  print 'all ents:',allEnts
  print '---------------------------'
  print 'linkable recall:',noDefineRightEnts/all_linkable_ents
  print 'refine linkable recall:', allRightEnts/all_linkable_ents
  print 'fixed c linkable recall:',rightEnts*1.0/all_linkable_ents
  print '---------------------------------------'
  print 'average disambiguation:',np.average(disambiguation)
  print 'withou thread disamguation:',np.average(disambiguation_withoutThread)
  print 'withoud define:',np.average(disambiguation_withoutDefine)
  print '---------------------------------------'
        

def printfeatues(ent_id):
  allEnts=0
  passEnts=0
  rightEnts=0
  wrongEnts=0
  
  entids=0
  relentid = -1
  k  = -1
  rightWiki = 0
  rightFb=0
  rightWord2vec = 0
  disambiguation = []
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      #key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      
      linkTag = entMentsTags[ent_id][1]   #get entity mention linking mid!
      #print linkTag
      enti_name = ents[j].content.lower()
      
      if linkTag=='NIL':
        passEnts+=1

        #ent_id +=1
        #continue      
      k+=1
      print 'k:',k
      ent_id += 1

     
      relentid += 1
      
      candMids = list(NERentCands[k].keys())
      wikicands = wiki_candidate_mids[k]
      FBcands = freebase_candidate_mids[k]
    
  
#      if 'surrey' in enti_name:
#        print 'candMids:',len(candMids)
#        
      '''
      @2017/3/23 need to reserve the top 10 wikipedia, revise tomorrow!
      '''
      deleteCands = {}
      for key in NERentCands[k]:
        if NERentCands[k][key][2]==0 and NERentCands[k][key][1]==0 and key not in wikicands:
          candMids.remove(key)
          deleteCands[key] = NERentCands[k][key]
#          print 'candMids:',len(candMids)
#          print linkTag,NERentCands[k][linkTag]
      #print len(candMids)
      disambiguation.append(len(candMids))
      
      
        
      if linkTag in candMids:
        rightEnts+=1
        if linkTag in wikicands:
          rightWiki += 1
        if linkTag in FBcands:
          rightFb+= 1
#        if linkTag not in FBcands and linkTag not in wikicands:
#          rightWord2vec += 1
        if NERentCands[k][linkTag][1] !=0:
          rightWord2vec += 1
      else:
        wrongEnts += 1
      #print len(set(candMids))
  
  print '---------------------------' 
  print 'datatag',data_tag
  print 'right:',rightEnts
  print 'wrong:',wrongEnts
  print 'pass:',passEnts
  print 'all ents:',allEnts
  print 'rightWiki:',rightWiki, rightWiki*1.0/(rightEnts+wrongEnts)
  print 'rightFB:',rightFb, rightFb*1.0/(rightEnts+wrongEnts)
  print 'rightWord2vec:',rightWord2vec, rightWord2vec*1.0/(rightEnts+wrongEnts)
  print rightEnts*1.0/(rightEnts+wrongEnts)
  
  print 'average disambiguation:',np.average(disambiguation)
  print '---------------------------------------'
        
      

def getAccuracy(ent_id):
  rightPred = 0
  wrongEnts =0
  entids = -1
  
  TN=0.0;FP=0.0
  FN=0.0;TP=0.0
  
  right = 0.0; total=0.0
  print 'NERentCands:',len(NERentCands)
  print 'NERets shape:',np.shape(NERrets)
  ent_mention_index={};ent_mention_index_right ={};ent_mention_index_wrong_nil={};ent_mention_index_wrong_oent={}
  NILEntKeys={};NILEnt_right_Keys = {};NILEnt_wrong_Keys={}
  relentid = -1
  passEnts = 0
  wrongEnts = 0
  notrecall = 0
  
  allEnts = 0
  k  = -1
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    tagid=-1
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      
      linkTag = entMentsTags[ent_id][1]   #get entity mention linking mid!
      ent_id +=1
      #print linkTag
      enti_name = ents[j].content.lower()
      k+= 1
      candMids = NERentCands[k].keys()
      if linkTag=='NIL':
        #print 'NIL entity:',enti_name,candMids
        passEnts+=1
        
        NILEntKeys[key]='NIL'
        candMids = NERentCands[entids].keys()
        ret = NERrets[relentid]
        
        right_id = np.argmax(ret,0)
        
        if len(candMids) ==0:
          predMid = 'NIL'
          TN += 1
          NILEnt_right_Keys[key]='NIL'
        else:
          if right_id >= len(candMids):
            TN += 1
            NILEnt_right_Keys[key]='NIL'
          else:
            FP+= 1
            NILEnt_wrong_Keys[key] = candMids[right_id]
            
        
        continue
      relentid += 1
      total+= 1
      tagid +=1
      ent_mention_index[key]= linkTag
      
      if len(candMids) ==0:
        FN += 1
        ent_mention_index_wrong_nil[key]=linkTag
        wrongEnts += 1
      else:
        ret = NERrets[relentid]
        #if len(candMids) !=0:  
        ret = ret[0:len(candMids)]
        
        right_id = np.argmax(ret,0)
        
        if right_id < len(candMids):
          predMid = candMids[right_id]
        
      
        if predMid in linkTag: 
          TP += 1
          rightPred += 1
          right += 1
          ent_mention_index_right[key] = linkTag
        else:
          wrongEnts += 1
          ent_mention_index_wrong_oent[key] = linkTag
          FP += 1  
  recall = TP/(TP+FN)
  precision = TP/(TP+FP)
  print 'linkable accuracy:',right, total,right/total
  print 'recall:',TP/(TP+FN)
  print 'precision:',TP/(TP+FP)
  print 'F1 score:', 2 * precision*recall/(precision + recall)
  print 'accuracy:',(TP+TN)/(TP+TN+FP+FN)
  print 'rightPred:',rightPred
  print 'pass ents:',passEnts
  print 'wrong ents:',wrongEnts
  print 'total:',wrongEnts+rightPred
  print 'precision:',rightPred*1.0/(wrongEnts+rightPred)
  print 'Nil ents:',len(NILEntKeys)
  print 'all ents:',allEnts
  print 'not recall:', notrecall
  print 'rel ent id:',k
  
  #cPickle.dump(ent_mention_index_right,open(dir_path+'rightPred.p','wb'))
  
  total = wrongEnts+rightPred
  
  
  
  print 'len corefEnts:',len(corefEnts)
  for key in corefEnts:
    items = key.split('\t')
    new_key =  '\t'.join(items[0:3])
    ent_name = items[3]
    
    #if key not in NILEntKeys and key not in ent_mention_index_right:
    val = corefEnts[key]
    coref_items = val.split('\t')
    coref_key = '\t'.join(coref_items[0:3])
    coref_name = coref_items[3]
    #we need to consider the coref
    if ent_name != coref_name:
      if new_key in ent_mention_index_wrong_nil or new_key in ent_mention_index_wrong_oent:
        if coref_key in ent_mention_index:
          if ent_mention_index[new_key] == ent_mention_index[coref_key]:
            print 'wrong 2 right:',ent_mention_index[new_key], ent_mention_index[coref_key]
            right += 1
      else:
        if new_key in ent_mention_index_right:
          
          if coref_key in ent_mention_index:
            if ent_mention_index[new_key] != ent_mention_index[coref_key]:
              print 'right 2 wrong:',ent_mention_index[new_key], ent_mention_index[coref_key]
              right -=1
            if coref_key in NILEntKeys:
              print 'right 2 wrong:',ent_mention_index[new_key], ent_mention_index[coref_key]
              right -= 1
  recall = TP/(TP+FN)
  precision = TP/(TP+FP)
  print 'linkable accuracy:',right, total,right/total
  print 'recall:',TP/(TP+FN)
  print 'precision:',TP/(TP+FP)
  print 'F1 score:', 2 * precision*recall/(precision + recall)
  print 'accuracy:',(TP+TN)/(TP+TN+FP+FN)
  print 'right:',rightPred
  print 'wrong ents:',wrongEnts
  print 'total:',total
  print 'pass ents:',passEnts
  print 'precison:',rightPred*1.0/(total)

def getSimilarEnts(ent_id):
  entids = -1
  rel_entids = 0
  NILEntKeys = {}
  ids2entName=[]
  ids2linkTag=[]
  allEnts = 0
  k  = -1
  EntMentVector=[]
  went_ids=[]
  went_name=[]
  for i in range(len(ent_Mentions)):
    aNosNo = id2aNosNo[i]
    ents = ent_Mentions[i]
    
    tagid=-1
    for j in range(len(ents)):
      entids +=1
      allEnts += 1
      startI = ents[j].startIndex; endI = ents[j].endIndex
      key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
      
      linkTag = entMentsTags[ent_id][1]
      ent_id += 1
      entRep_vector = np.sum(lstm_output[i][startI:endI],axis=0)
      if linkTag!='NIL':
        EntMentVector.append(entRep_vector)
        enti_name = ents[j].content.lower()
        print enti_name
        ids2entName.append(ents[j].content)
        ids2linkTag.append(linkTag)
        if i == 9:
          went_ids.append(rel_entids)
          went_name.append(ents[j].content)
      rel_entids += 1
  
  EntMentVector = np.asarray(EntMentVector)
  
  dist_out = cosine_similarity(EntMentVector)
  
  for ids in range(len(went_ids)):
    print '--------------'
    key = went_ids[ids]
    print went_name[ids]
    argsorts = np.argsort(dist_out[key])
    
    for tt in range(1,10):
      rel_tt = tt *(-1)
      print dist_out[key][argsorts[rel_tt]],ids2entName[argsorts[rel_tt]],ids2linkTag[argsorts[rel_tt]]
    print '--------------'
  
    
      
      
parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
parser.add_argument('--features', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
 
  
data_args = parser.parse_args()
data_tag = data_args.data_tag
dir_path = data_args.dir_path
features = data_args.features

corefEnts = cPickle.load(open(dir_path+'process/'+data_tag+'_entMent2repMent.p'))
print data_tag,' coref numbers:',len(corefEnts),features



entMentsTags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
ent_id=0
if data_tag=='train':
  ent_id = 0
if data_tag=='testa':
  ent_id = 23396
if data_tag=='testb':
  ent_id = 29313

print 'start to load datasets....'
stime = time.time()
if data_tag=='train':
  f_input_ent_ments = dir_path + '/features/' + data_tag+'_entms.p100_new'
else:
  f_input_ent_ments = dir_path + '/features/' + data_tag+'_entms.p100'
dataEnts = cPickle.load(open(f_input_ent_ments,'rb'));
ent_Mentions = dataEnts['ent_Mentions']
data = cPickle.load(open(dir_path+'process/'+ data_tag+'.p','r'))
aNosNo2id = data['aNosNo2id']
id2aNosNo = {val:key for key,val in aNosNo2id.items()}


'''
generate average ambigugation!
'''
#printfeatues(ent_id)

'''
generate real candidates
'''
#data_param={'all_candidate_mids':all_candidate_mids,'wiki_candidate_mids':wiki_candidate_mids,'freebase_candidate_mids':freebase_candidate_mids}
data_param = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_cand_mid.p'))  #[300,candsNums]
NERentCands = data_param['all_candidate_mids']
print 'NERentCands:',len(NERentCands)
wiki_candidate_mids = data_param['wiki_candidate_mids']
freebase_candidate_mids = data_param['freebase_candidate_mids']
printfeatuesThreshold(ent_id)


'''
@evaluation
'''
#stime = time.time()
#linkcandsProb = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_ent_linking_candprob.p'+str(candidateEntNum),'rb'))
#ent_linking  = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_ent_linking.p'+str(candidateEntNum),'rb'))
#print 'load dataset cost time:',time.time()-stime
#print 'ent_linking:',len(ent_linking['ent_mention_tag'])
#
#NERrets = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_entityLinkingResult.p'+features))
#
##NERrets = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+features+'_'+data_tag+'_entityLinkingResult.p'+str(candidateEntNum)))
##print dir_path+data_tag+'_entityLinkingResult.p'
#print np.shape(NERrets)
#if candidateEntNum==50:
#  NERentCands = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'++data_tag+'_ent_cand_mid_new.p'))
#else:
#  NERentCands = cPickle.load(open(dir_path+'features/'+str(candidateEntNum)+'/'+data_tag+'_ent_cand_mid_new.p'+str(candidateEntNum)))
#getAccuracy(ent_id)


'''
@get similar entity mentions
'''
#NERrets = cPickle.load(open(dir_path+data_tag+'_entityLinkingResult.p'))
#print dir_path+data_tag+'_entityLinkingResult.p'
#print np.shape(NERrets)
#NERentCands = cPickle.load(open(dir_path+'features/'+data_tag+'_ent_cand_mid_new.p')+str(candidateEntNum))
#lstm_output = cPickle.load(open(dir_path+data_tag+'lstm_output.p'))
#print len(ent_Mentions)
#print np.shape(lstm_output)
#getSimilarEnts(ent_id)

