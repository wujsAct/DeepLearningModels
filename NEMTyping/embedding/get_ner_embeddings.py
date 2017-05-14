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
import re
import gensim

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
    #print len(typeList)
    
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

def getConll2003EntTags(tags):
  nerTags = [[0,0,1]]*len(tags)
  strs = ''.join(tags)
  types = ['0','1','2','3']
  for ti in types:
    classType = r''+ti+r'+'   #greedy matching
    pattern = re.compile(classType)
    
    matchList = re.finditer(pattern,strs)  #find all the entity type tags
    for match in matchList:
      start = match.start(); end = match.end()
      #print 'conll2003:',start,end,match.group()
      nerTags[start]=list([1,0,0])
      if start+1 <= end-1:
        for i in range(start+1,end):
          nerTags[i] = list([0,1,0])
  return nerTags
#transfer into B,I,O formats
def get_input_conll2003_test(model,word_dim, input_file_obj, output_embed, output_tag, sentence_length=-1):
  vocabs = model.vocab
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  
  word = []
  tag = []
  sentence = []
  sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  #ner2id = {'I-PER':'0','B-PER':'0','I-LOC':'1','I-ORG':'2','I-MISC':'3','O':'4'}
  #nertag2id={'B-E':0,'I-E':1,'0':2}
  print("max sentence length is %d" % max_sentence_length)
  for epoch in range(3):
    for line in input_file_obj:
      if line in ['\n', '\r\n']:
        #we need to generate B,I,O format tags
        
        for _ in range(max_sentence_length - sentence_length):
          tag.append('4')
          temp = np.array([0 for _ in range(word_dim + 10)])
          word.append(temp)
        sentence.append(word)
        sentence_tag.append(np.array(getConll2003EntTags(tag)))
          
        sentence_length = 0
        word = []
        tag = []
      else:
        line = line.strip()
        assert (len(line.split()) == 4)
        sentence_length += 1
        wd = line.split()[0]
        if wd in vocabs:
            temp = model[wd]
        elif wd.lower() in vocabs:
            temp = model[wd.lower()]
        else:
            #temp = np.zeros((300,))
            temp = randomVector
        
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim+10  #还需要添加上
        word.append(temp)
        t = line.split()[3] # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
        if 'PER' in t:
          tag.append('0')
        elif 'LOC' in t:
          tag.append('1')
        elif 'ORG' in t:
          tag.append('2')
        elif 'MISC' in t:
          tag.append('3')
        elif 'O' in t:
          tag.append('4')
        else:
          print 'tag is wrong...'
          exit(0)
  
  assert (len(sentence) == len(sentence_tag))
  print 'start to save datasets....'
  cPickle.dump(np.asarray(sentence), open(output_embed, 'wb'))
  cPickle.dump(np.asarray(sentence_tag), open(output_tag, 'wb'))

#transfer into B,I,O formats
def get_input_conll2003(model,batch_size,word_dim, input_file_obj,sentence_length=-1):
  vocabs = model.vocab
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  
  word = []
  tag = []
  sentence = []
  sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  #ner2id = {'I-PER':'0','B-PER':'0','I-LOC':'1','I-ORG':'2','I-MISC':'3','O':'4'}
  #nertag2id={'B-E':0,'I-E':1,'0':2}
  print("max sentence length is %d" % max_sentence_length)
  for epoch in range(3):
    for line in input_file_obj:
      if line in ['\n', '\r\n']:
        #we need to generate B,I,O format tags
        
        for _ in range(max_sentence_length - sentence_length):
          tag.append('4')
          temp = np.array([0 for _ in range(word_dim + 10)])
          word.append(temp)
        sentence.append(word)
        sentence_tag.append(np.array(getConll2003EntTags(tag)))
        
        if len(sentence)== batch_size:
          yield sentence,sentence_tag
          sentence=[];sentence_tag=[]
          
        sentence_length = 0
        word = []
        tag = []
      else:
        line = line.strip()
        assert (len(line.split()) == 4)
        sentence_length += 1
        wd = line.split()[0]
        if wd in vocabs:
            temp = model[wd]
        elif wd.lower() in vocabs:
            temp = model[wd.lower()]
        else:
            #temp = np.zeros((300,))
            temp = randomVector
        
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim+10  #还需要添加上
        word.append(temp)
        t = line.split()[3] # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
        if 'PER' in t:
          tag.append('0')
        elif 'LOC' in t:
          tag.append('1')
        elif 'ORG' in t:
          tag.append('2')
        elif 'MISC' in t:
          tag.append('3')
        elif 'O' in t:
          tag.append('4')
        else:
          print 'tag is wrong...'
          exit(0)
  
#  assert (len(sentence) == len(sentence_tag))
#  print 'start to save datasets....'
#  cPickle.dump(np.asarray(sentence), open(output_embed, 'wb'))
#  cPickle.dump(np.asarray(sentence_tag), open(output_tag, 'wb'))

def getFigerTestTag(max_sentence_length):
  input_file_obj = openFile('data/figer_test/gold.segment')
  sentence_length=0
  finalTag =[]
  tag=[]
  for line in input_file_obj:
    if line in ['\n','\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        tag.append([0,0,1])
      sentence_length = 0
      finalTag.append(tag)
      tag=[]
    else:
      sentence_length += 1
      line = line.strip()
      items = line.split('\t')
      assert len(items)==2
                
      if 'O' in items[1]:
        tag.append([0,0,1])
      elif 'B' in items[1]:
        tag.append([1,0,0])
      elif 'I' in items[1]:
        tag.append([0,1,0])
      else:
        print 'tag is wrong...'
        exit(0)
  return finalTag



def get_input_figer(model,word_dim,input_file_obj,output_embed, output_tag, sentence_length=-1):
  vocabs = model.vocab
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  
  word = []

  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence = []
  sentence_tag = getFigerTestTag(max_sentence_length)
  
  sentence_length = 0
  ids = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in input_file_obj:
    if line in ['\n', '\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        temp = np.array([0 for _ in range(word_dim + 10)])
        word.append(temp)
      assert len(sentence_tag[ids]) == len(word)
      sentence.append(word)
      sentence_length = 0
      word = []
      ids += 1
    else:
      line = line.strip()
      assert (len(line.split()) == 3)  #only has Word,pos_tag
      sentence_length += 1
      #temp = model[line.split()[0]]
      wd = line.split()[0]
      if wd in vocabs:
        temp = model[wd]
      elif wd.lower() in vocabs:
        temp = model[wd.lower()]
      else:
        #temp = np.zeros((300,))
        temp = randomVector
          
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
      #assert len(temp) == word_dim+11
      assert len(temp) == word_dim+10
      word.append(temp)
      
  assert (len(sentence) == len(sentence_tag))
  print 'start to save datasets....'
  cPickle.dump(np.asarray(sentence), open(output_embed, 'wb'))
  cPickle.dump(np.asarray(sentence_tag), open(output_tag, 'wb'))
  
def getFigerTag(entList,max_sentence_length):
  nerTags = [[0,0,1]]*max_sentence_length
  for i in range(len(entList)):
    ent = entList[i]
    start= int(ent[0])
    end = int(ent[1])
    nerTags[start]=[1,0,0]
    if start+1 <= end-1:
        for j in range(start+1,end):
          nerTags[j] = list([0,1,0])
  return np.asarray(nerTags)
    

def get_input_figer_chunk_test_ner(model,word_dim, input_file_obj,entMents, output_embed, output_tag, sentence_length=-1):
  
  vocabs = model.vocab
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  #input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  #entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  #print 'figer types:',len(figer2id)
  
  allid=0
  word = []
  tag = []
  sentence = []
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
  sid = 0
  allLines = input_file_obj.readlines()
  for line in allLines:
    if line in ['\n', '\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        #tag.append(np.array([0] * 5))
        temp = np.array([0 for _ in range(word_dim + 6+4)])
        word.append(temp)
      
      entList = entMents[sid]
      nerTags=getFigerTag(entList,max_sentence_length) 
      sentence.append(word)
      tag.append(nerTags)
      
      
      allid += 1   
      sid += 1
      sentence_length = 0
      word = []
    else:
      line = line.strip()
      assert (len(line.split("\t")) == 3)  #only has Word,pos_tag
      sentence_length += 1
      wd = line.split()[0]
      if wd in vocabs:
        temp = model[wd]
      elif wd.lower() in vocabs:
        temp = model[wd.lower()]
      else:
        #temp = np.zeros((300,))
        temp = randomVector
      #temp = np.zeros((100,))
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
      assert len(temp) == word_dim+6+4
      word.append(temp)
        
  assert (len(sentence) == len(tag))
  print 'start to save datasets....'
  cPickle.dump(np.asarray(sentence), open(output_embed, 'wb'))
  cPickle.dump(np.asarray(tag), open(output_tag, 'wb'))

def get_input_figer_chunk_test_ner_train(batch_size,model,word_dim, input_file_obj,entMents, sentence_length=-1):
  epochs=1
  vocabs = model.vocab
  randomVector = cPickle.load(open('data/figer/randomVector.p','rb'))
  #input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  #entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  #print 'figer types:',len(figer2id)
  
  allid=0
  word = []
  tag = []
  sentence = []
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
  for epoch in range(epochs):
    sid = 0
    
    allLines = input_file_obj.readlines()
    for line in allLines:
      if line in ['\n', '\r\n']:
        for _ in range(max_sentence_length - sentence_length):
          #tag.append(np.array([0] * 5))
          temp = np.array([0 for _ in range(word_dim + 6+4)])
          word.append(temp)
        
        entList = entMents[sid]
        nerTags=getFigerTag(entList,max_sentence_length) 
        sentence.append(word)
        tag.append(nerTags)
        if ((allid+1)%batch_size==0 and allid!=0) or allid == 2854-1:
          #if len(sentence) == batch_size:
          if len(sentence)!=0:
            yield np.asarray(sentence),np.asarray(tag)
            sentence=[];tag=[]
        allid += 1   
        sid += 1
        sentence_length = 0
        word = []
      else:
        line = line.strip()
        assert (len(line.split("\t")) == 3)  #only has Word,pos_tag
        sentence_length += 1
        wd = line.split()[0]
        if wd in vocabs:
          temp = model[wd]
        elif wd.lower() in vocabs:
          temp = model[wd.lower()]
        else:
          #temp = np.zeros((300,))
          temp = randomVector
        #temp = np.zeros((100,))
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim+6+4
        word.append(temp)

def get_input_figer_chunk_train_ner(batch_size,dir_path,set_tag,model,word_dim,sentence_length=-1):
  epochs=1
  input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  #print 'figer types:',len(figer2id)
  
  allid=0
  word = []
  tag = []
  sentence = []
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
  for epoch in range(epochs):
    sid = 0
    for line in input_file_obj:
      if line in ['\n', '\r\n']:
        for _ in range(max_sentence_length - sentence_length):
          #tag.append(np.array([0] * 5))
          temp = np.array([0 for _ in range(word_dim +6+4)])
          word.append(temp)
        
        entList = entMents[sid]
        nerTags=getFigerTag(entList,max_sentence_length) 
        sentence.append(word)
        tag.append(nerTags)
        
        if ((allid+1)%batch_size==0 and allid!=0) or allid == 200000-1:
          #if len(sentence) == batch_size:
          if len(sentence)!=0:
            yield np.asarray(sentence),np.asarray(tag)
            sentence=[];tag=[]
        allid += 1   
        sid += 1
        sentence_length = 0
        word = []
      else:
        assert (len(line.split()) == 3)  #only has Word,pos_tag
        sentence_length += 1
        temp = model[line.split()[0]]
        #temp = np.zeros((100,))
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        #temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        #assert len(temp) == word_dim+6+5
        assert len(temp) == word_dim+10
        word.append(temp)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str, help='data file', required=True)
  parser.add_argument('--data_tag', type=str, help='raw datasets', required=True)
  parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
  parser.add_argument('--use_model', type=str, help='model location', required=True)
  parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
  
  start_time = time.time()
  
  args = parser.parse_args()
  print args.data_tag
  if args.dir_path == 'data/figer/' or args.dir_path == 'data/figer_test/':
    input_file_obj = openFile(args.dir_path+args.data_tag+'Data.txt')
  elif args.dir_path =='data/conll2003/':
    input_file_obj = openFile(args.dir_path+'eng.'+args.data_tag)
  
  print 'start to load word2vec models!'
  #trained_model = cPickle.load(open(args.use_model, 'rb'))
  trained_model = gensim.models.Word2Vec.load_word2vec_format('/home/wjs/demo/entityType/informationExtract/data/GoogleNews-vectors-negative300.bin', binary=True)
  print 'load word2vec model cost time:',time.time()-start_time
  #print trained_model.wvec_model.vocab

  
  if args.dir_path == 'data/figer_test/':
    get_input_figer(trained_model, args.model_dim, input_file_obj,
            args.dir_path+'nerFeatures/'+args.data_tag+'_embed.p'+str(args.model_dim),
            args.dir_path+'nerFeatures/'+args.data_tag+'_tag.p'+str(args.model_dim),
            sentence_length=args.sentence_length)
  elif args.dir_path =='data/WebQuestion/':
    input_file_obj = open(args.dir_path+'features/'+args.data_tag+'_Data.txt')
    entMents = cPickle.load(open(args.dir_path+'features/'+args.data_tag+'_entMents.p','rb'))
    output_embed = args.dir_path+'nerFeatures/'+args.data_tag+'_embed.p'+str(args.model_dim)
    output_tag = args.dir_path+'nerFeatures/'+args.data_tag+'_tag.p'+str(args.model_dim)
    #print 'figer types:',len(figer2id)
    get_input_figer_chunk_test_ner(trained_model,args.model_dim, input_file_obj,entMents, output_embed, output_tag, sentence_length=args.sentence_length)
  else:
    get_input_conll2003_test(trained_model, args.model_dim, input_file_obj,
            args.dir_path+'nerFeatures/'+args.data_tag+'_embed.p'+str(args.model_dim),
            args.dir_path+'nerFeatures/'+args.data_tag+'_tag.p'+str(args.model_dim),
            sentence_length=args.sentence_length)
  
  
  
  
  