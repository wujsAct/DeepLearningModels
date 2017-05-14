#from __future__ import print_function
# -*- coding: utf-8 -*-
'''
@editor: wujs
function: chunk
revise: 2017/4/17
'''
import sys
sys.path.append("/home/wjs/demo/entityType/NEMType/evals/")
import sys
import numpy as np
import cPickle
import gzip
import tensorflow as tf
from tqdm import tqdm
import collections
import time
from description_embed_model import WordVec,MyCorpus
from evalChunk import openSentid2aNosNoid,getaNosNo2entMen
import argparse

def openFile(fileName):
  if fileName.endswith('gz'):
    return gzip.open(fileName,'r')
  else:
    return open(fileName)
  
def find_max_length(fileObject):
  sentid = 0
  temp_len = 0
  max_length = 0
  max_sent= []
  sents = []
  for line in fileObject:
    if line in ['\n', '\r\n']:
      sentid += 1
      if temp_len > max_length:
        max_length = temp_len
        max_sent = sents
      temp_len = 0
      sents=[]
    else:
      sents.append(line.split(' ')[0])
      temp_len += 1
  print 'max_length:',max_length
  print max_sent
  print 'total sents:',sentid
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


#our goal is the predict chunk
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

'''
revise time: 2017/1/11, add more extrac features!
'''
def capital(word):
  ret =np.array([0])
  if ord('A') <= ord(word[0]) <= ord('Z'): #inital words are capital
    ret[0]=1
  return ret
def get_figerChunk(dir_path):
  fName = dir_path+'figer_chunk.txt'
  tag=[]
  
  for line in open(fName):
    if line not in ['\n']:
      tag.append(line.strip())
  return tag

'''
tag: sentence_length*[typeId], 
'''
def getFigerEntTags(entList,sid,ent_no):
  ent_mention_mask=[]
  type_indices=[]
  type_val=[]
  for i in range(len(entList)):
    ent = entList[i]
    ent_start= int(ent[0])
    ent_end = int(ent[1])
    typeList = sorted(list(set(ent[2])))  #ascending sort and duplicate type remove!
#    #print len(typeList)
    if ent_start >  250:
      print ent_no
    '''
    sparse pattern
    '''
    ent_mention_mask.append([sid,ent_start,ent_end])
    for t in range(len(typeList)): 
      ind =[ent_no,typeList[t]]  #batch_id,sequence_length_id,class_id
      type_indices.append(ind)
      type_val.append(1)

    ent_no += 1
  
  return ent_no,ent_mention_mask,type_indices,type_val

def get_input_figerTest_chunk(flag,dataset,dir_path,model,word_dim,sentence_length=-1):
  vocab = model.vocab
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  epochs = 1
#  vocab2id = cPickle.load(open('data/vocab2id.p'))
#  print len(vocab2id)
  #figerTypes = 113
  
  if dir_path=='data/figer_test/':
    input_file_obj = open(dir_path+'features/figerData.txt')
    entMents = cPickle.load(open(dir_path+'features/figer_entMents.p','rb'))
  else:
    input_file_obj = open(dir_path+'features/'+dataset+'Data_test.txt')
    entMents = cPickle.load(open(dir_path+'features/'+'test_entMents.p','rb'))
  
  allid=0
  word = []
  #tag = []
  type_indices=[]
  type_val=[]
  sentence = []
  ent_mention_mask=[]
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  ent_no=0
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
#  sentence_final=[]
#  type_final = []
#  ent_mention_mask_final=[]
  for epoch in range(epochs):
    sid = 0
    for line in input_file_obj:
      if line in ['\n', '\r\n']:
        #for _ in range(max_sentence_length - sentence_length):
          #tag.append(np.array([0] * 5))
        #  temp = np.array([0 for _ in range(word_dim)])
        #  word.append(temp)
        #if sentence_length > max_sentence_length:
        #  print 'sentence_length is longer than max_sentence_length...', sentence_length
        if flag=='LSTM':
          word += [np.zeros((311,))]* (max_sentence_length - sentence_length)
        else:
          word += [np.zeros((300,))]* (max_sentence_length - sentence_length)
        entList = entMents[sid]
        ent_no,temp_ent_mention_mask,temp_type_indices,temp_type_val = getFigerEntTags(entList,allid,ent_no)
        sentence.append(word)
        ent_mention_mask += temp_ent_mention_mask
        type_indices += temp_type_indices
        type_val += temp_type_val
        
        
        allid += 1   
        sid += 1
        sentence_length = 0
        word = []
      else:
        assert (len(line.split()) == 3)  #only has Word,pos_tag
        sentence_length += 1
        wd = line.split()[0]
        if wd in vocab:
          temp = model[wd]
        elif wd.lower() in vocab:
          temp = model[wd.lower()]
        else:
          temp = np.zeros((300,))
        if flag=='LSTM':
          temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
          temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
          temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
          #print 'non words...'
          #temp = randomVector
#        wd = line.split()[0]
#        if wd in vocab2id:
#          temp = vocab2id[wd]
#        elif wd.lower() in vocab2id:
#          temp = vocab2id[wd.lower()]
#        else:
#          temp = vocab2id['NIL']
        word.append(temp)
  return ent_mention_mask,sentence,[np.asarray(type_indices, dtype=np.int64),np.asarray(type_val, dtype=np.float32)]

def get_input_figer_chunk(batch_size,dir_path,set_tag,model,word_dim,sentence_length=-1):
  epochs = 1
  #figerTypes = 113
  input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  #figer2id = cPickle.load(open(dir_path+"figer2id.p",'rb'))
  #otherType=len(figer2id)
  #print 'figer types:',len(figer2id)
  
  allid=0
  word = []
  #tag = []
  type_indices=[]
  type_val=[]
  sentence = []
  ent_mention_mask=[]
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  ent_no=0
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
  sentence_final=[]
  type_final = []
  ent_mention_mask_final=[]
  for epoch in range(epochs):
    sid = 0
    for line in input_file_obj:
      if line in ['\n', '\r\n']:
        for _ in range(max_sentence_length - sentence_length):
          #tag.append(np.array([0] * 5))
          temp = np.array([0 for _ in range(word_dim)])
          word.append(temp)
        entList = entMents[sid]
        ent_no,temp_ent_mention_mask,temp_type_indices,temp_type_val = getFigerEntTags(entList,allid%batch_size,ent_no)
        sentence.append(word)
        ent_mention_mask += temp_ent_mention_mask
        type_indices += temp_type_indices
        type_val += temp_type_val
        if (allid+1)%batch_size==0 and allid!=0:
          if len(sentence) == batch_size:
            #yield ent_mention_mask,sentence,[np.asarray(type_indices, dtype=np.int64),np.asarray(type_val, dtype=np.float32)]
            sentence_final.append(np.asarray(sentence))
            ent_mention_mask_final.append(ent_mention_mask)
            type_final.append([np.asarray(type_indices, dtype=np.int64),np.asarray(type_val, dtype=np.float32)])
            
            #sentence_final;ent_mention_mask_final=[];type_final=[]
            sentence=[];type_indices=[];type_val=[];ent_mention_mask=[];ent_no=0
        allid += 1   
        sid += 1
        sentence_length = 0
        word = []
      else:
        assert (len(line.split()) == 3)  #only has Word,pos_tag
        sentence_length += 1
        #temp = model[line.split()[0]]
        temp = temp = np.zeros((100,))
        #temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        #temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim
        word.append(temp)
  return ent_mention_mask_final,sentence_final,type_final

def get_input_figer_chunk_train(flag,dataset,batch_size,dir_path,set_tag,model,word_dim,sentence_length=-1):
  #vocab2id = cPickle.load(open('data/vocab2id.p'))
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  vocab = model.vocab
  epochs = 2
  figerTypes = 113
  
  
  allid=0
  word = []
  #tag = []
  type_indices=[]
  type_val=[]
  sentence = []
  ent_mention_mask=[]
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  ent_no=0
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
  for epoch in range(epochs):
    sid = 0
    #input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
    input_file_obj = open(dir_path+'features/'+dataset+'Data_'+set_tag+'.txt')
    entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
    for line in input_file_obj:
      if line in ['\n', '\r\n']:
        if flag=='LSTM':
          word += [np.zeros((311,))]* (max_sentence_length - sentence_length)
        else:
          word += [np.zeros((300,))]* (max_sentence_length - sentence_length)

        entList = entMents[sid]
        ent_no,temp_ent_mention_mask,temp_type_indices,temp_type_val = getFigerEntTags(entList,allid%batch_size,ent_no)
        sentence.append(word)
        ent_mention_mask += temp_ent_mention_mask
        type_indices += temp_type_indices
        type_val += temp_type_val
        if (allid+1)%batch_size==0 and allid!=0:
          if len(sentence) == batch_size:
            yield ent_mention_mask,np.asarray(sentence),[np.asarray(type_indices, dtype=np.int64),np.asarray(type_val, dtype=np.float32)]
            sentence=[];type_indices=[];type_val=[];ent_mention_mask=[];ent_no=0
        allid += 1   
        sid += 1
        sentence_length = 0
        word = []
      else:
        assert (len(line.split()) == 3)  #only has Word,pos_tag
        sentence_length += 1
        wd = line.split()[0]
        if wd in vocab:
          temp = model[wd]
        elif wd.lower() in vocab:
          temp = model[wd.lower()]
        else:
          #we also utilize a random models
          #temp = randomVector
          #print wd,'non words...'
          temp = np.zeros((300,))
        if flag=='LSTM':
          temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
          temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
          temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        #temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        #temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        #assert len(temp) == word_dim
        word.append(temp)
        
      
if __name__ == '__main__':
  print 'genereate the figer typing embeddings ....'
  
  dataset = "OntoNotes"
  #cPickle.dump(np.random.rand(300),open('data/figer/randomVector.p','wb'))
  get_input_figerTest_chunk('MLP',dataset,'data/'+dataset+'/',model=None,word_dim=100,sentence_length=250)
  