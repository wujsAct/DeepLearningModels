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
       

def get_input_figer(model,word_dim,input_file_obj,sentence_length=-1):
  word = []
  tag = []
#  sentence = []
#  sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in input_file_obj:
    if line in ['\n', '\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        tag.append(np.array([0] * 5))
        temp = np.array([0 for _ in range(word_dim + 6)])
        word.append(temp)
      assert len(tag) == len(word)
      yield word,tag
      sentence_length = 0
      word = []
      tag = []
    else:
      assert (len(line.split()) == 2)  #only has Word,pos_tag
      sentence_length += 1
      temp = model[line.split()[0]]
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      #temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
      assert len(temp) == word_dim+6
      word.append(temp)
      tag.append(np.array([0, 0, 0, 0, 0])) #we need to predict the chunk tag!

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
def getFigerEntTags(otherType,entList,sentence_length,sid):
  indices = []
  val=[]
  hasTypeSq=collections.defaultdict(list)
  for i in range(len(entList)):
    ent = entList[i]
    ents= int(ent[0])
    ente = int(ent[1])
    typeList = sorted(list(set(ent[2])))  #ascending sort and duplicate type remove!
    '''
    sparse pattern
    '''
    for j in range(ents,ente):
      hasTypeSq[j]=typeList
      #for t in range(len(typeList)): 
        #ind =[sid,j,t]  # batch_id,sequence_length_id,class_id
        #indices.append(ind)
        #val.append(1)
  for i in range(sentence_length):  #add those nontype tags!
    if i in hasTypeSq:
      for t in hasTypeSq[i]:
        ind=list([sid,i,t])
        indices.append(list(ind))
        val.append(1)
    else:
      ind=list([sid,i,otherType])
      indices.append(list(ind))
      val.append(1)
  return indices,val

def get_input_figer_chunk(dir_path,set_tag,model,word_dim,sentence_length=-1):
  batch_size = 256
  epochs = 2
  figerTypes = 114
  input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  figer2id = cPickle.load(open(dir_path+"figer2id.p",'rb'))
  otherType=len(figer2id)
  #print 'figer types:',len(figer2id)
  
  allid=0
  word = []
  #tag = []
  tag_indices=[]
  tag_val=[]
  sentence = []
  tag_shape = np.array([batch_size, sentence_length,figerTypes], dtype=np.int64)
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  #print("max sentence length is %d" % max_sentence_length)
 
  for epoch in range(epochs):
    for line in input_file_obj:
      
      if line in ['\n', '\r\n']:
        for _ in range(max_sentence_length - sentence_length):
          #tag.append(np.array([0] * 5))
          temp = np.array([0 for _ in range(word_dim + 6+5)])
        
          word.append(temp)
          
        entList = entMents[allid]
        temp_tag_indices,temp_tag_val = getFigerEntTags(otherType,entList,sentence_length,allid%batch_size)
        sentence.append(word)
        tag_indices += temp_tag_indices
        tag_val += temp_tag_val
        if (allid+1)%batch_size==0 and allid!=0:
          if len(sentence) == batch_size:
            yield sentence,[np.asarray(tag_indices, dtype=np.int64),np.asarray(tag_val, dtype=np.float32),tag_shape]
            sentence=[]
            tag_indices=[];tag_val=[];
        allid += 1   
        sentence_length = 0
        
        sentence_length = 0
        word = []
        #tag = []
      else:
        assert (len(line.split()) == 3)  #only has Word,pos_tag
        sentence_length += 1
        temp = model[line.split()[0]]
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim+6+5
        word.append(temp)
  #return sentence,[np.asarray(tag_indices, dtype=np.int64),np.asarray(tag_val, dtype=np.float32),tag_shape]

def get_input_figer_chunk_train(dir_path,set_tag,model,word_dim,sentence_length=-1):
  batch_size = 256
  epochs = 30
  figerTypes = 114
  input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  figer2id = cPickle.load(open(dir_path+"figer2id.p",'rb'))
  otherType=len(figer2id)
  print 'figer types:',len(figer2id)
  
  allid=0
  word = []
  #tag = []
  tag_indices=[]
  tag_val=[]
  sentence = []
  tag_shape = np.array([batch_size, sentence_length,figerTypes], dtype=np.int64)
  #sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
 
  for epoch in range(epochs):
    for line in tqdm(input_file_obj):
      if line in ['\n', '\r\n']:
        for _ in range(max_sentence_length - sentence_length):
          #tag.append(np.array([0] * 5))
          temp = np.array([0 for _ in range(word_dim + 6+5)])
        
          word.append(temp)
          
        entList = entMents[allid]
        temp_tag_indices,temp_tag_val = getFigerEntTags(otherType,entList,sentence_length,allid%batch_size)
        sentence.append(word)
        tag_indices += temp_tag_indices
        tag_val += temp_tag_val
        if (allid+1)%batch_size==0 and allid!=0:
          if len(sentence) == batch_size:
            yield sentence,[np.asarray(tag_indices, dtype=np.int64),np.asarray(tag_val, dtype=np.float32),tag_shape]
            sentence=[]
            tag_indices=[];tag_val=[];
        allid += 1   
        sentence_length = 0
        word = []
        #tag = []
      else:
        assert (len(line.split()) == 3)  #only has Word,pos_tag
        sentence_length += 1
        temp = model[line.split()[0]]
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim+6+5
        word.append(temp)
def get_input(model, word_dim, input_file_obj, output_embed, output_tag, sentence_length=-1):
  word = []
  tag = []
  sentence = []
  sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in input_file_obj:
    if line in ['\n', '\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        tag.append(np.array([0] * 5))
        temp = np.array([0 for _ in range(word_dim + 6)])
        word.append(temp)
      sentence.append(word)
      sentence_tag.append(np.array(tag))
      sentence_length = 0
      word = []
      tag = []
    else:
      assert (len(line.split()) == 3)
      sentence_length += 1
      temp = model[line.split()[0]]
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      #temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
      assert len(temp) == word_dim+6   #还需要添加上
      word.append(temp)
      t = line.split()[2] # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
      if t.endswith('B-NP'):
        tag.append(np.array([1, 0, 0, 0, 0]))
      elif t.endswith('I-NP'):
        tag.append(np.array([0, 1, 0, 0, 0]))
      elif t.endswith('B-VP'):
        tag.append(np.array([0, 0, 1, 0, 0]))
      elif t.endswith('I-VP'):
        tag.append(np.array([0, 0, 0, 1, 0]))
      else:
        tag.append(np.array([0, 0, 0, 0, 1]))
  assert (len(sentence) == len(sentence_tag))
  print 'start to save datasets....'
  cPickle.dump(sentence, open(output_embed, 'wb'))
  cPickle.dump(sentence_tag, open(output_tag, 'wb'))
  
  
def get_input_conll2003(model, word_dim, input_file_obj, output_embed, output_tag, sentence_length=-1):
  word = []
  tag = []
  sentence = []
  sentence_tag = []
  if sentence_length == -1:
    max_sentence_length = find_max_length(input_file_obj)
  else:
    max_sentence_length = sentence_length
  sentence_length = 0
  print("max sentence length is %d" % max_sentence_length)
  for line in input_file_obj:
    if line in ['\n', '\r\n']:
      for _ in range(max_sentence_length - sentence_length):
        tag.append(np.array([0] * 5))
        temp = np.array([0 for _ in range(word_dim + 6 + 5)])
        word.append(temp)
      sentence.append(word)
      sentence_tag.append(np.array(tag))
      sentence_length = 0
      word = []
      tag = []
    else:
      assert (len(line.split()) == 4)
      sentence_length += 1
      temp = model[line.split()[0]]
      temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
      temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
      temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
      assert len(temp) == word_dim+6+5   #还需要添加上
      word.append(temp)
      t = line.split()[3]
      #Five classes 0-Person,1-Location,2-Organisation,3-Misc,4-None
      if t.endswith('PER'):
        tag.append(np.array([1, 0, 0, 0, 0]))
      elif t.endswith('LOC'):
        tag.append(np.array([0, 1, 0, 0, 0]))
      elif t.endswith('ORG'):
        tag.append(np.array([0, 0, 1, 0, 0]))
      elif t.endswith('MISC'):
        tag.append(np.array([0, 0, 0, 1, 0]))
      elif t.endswith('O'):
        tag.append(np.array([0, 0, 0, 0, 1]))
      else:
        print("error in input tag {%s}" % t)
        sys.exit(0)
  assert (len(sentence) == len(sentence_tag))
  print 'start to save datasets....'
  cPickle.dump(sentence, open(output_embed, 'wb'))
  cPickle.dump(sentence_tag, open(output_tag, 'wb'))


if __name__ == '__main__':

  word2vecModel = cPickle.load(open('data/wordvec_model_100.p', 'rb'))
  #word2vecModel = None
  output_data = tf.sparse_placeholder(tf.float32, name='outputdata')
  sess = tf.InteractiveSession()
  '''
  @sparse_tensor need to be in order and non duplicate elements!
  '''
  stime = time.time()
  for train_input,train_out in get_input_figer_chunk('data/figer/',"testa",model=word2vecModel,word_dim=100,sentence_length=80):
#    print np.shape(train_input)
#    print len(train_out[0])
#    print len(train_out[1])
#    print train_out[2]
    tt = sess.run(tf.sparse_tensor_to_dense(output_data), feed_dict={output_data:tf.SparseTensorValue(train_out[0],train_out[1],train_out[2])})
    print tt[0]
  #print 'cost time:',time.time()-stime
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_path', type=str, help='data file', required=True)
  parser.add_argument('--data_tag', type=str, help='raw datasets', required=True)
  parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
  parser.add_argument('--use_model', type=str, help='model location', required=True)
  parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
  
  start_time = time.time()
  
  args = parser.parse_args()
  print args.data_tag
  if args.dir_path == 'data/figer/':
    input_file_obj = openFile(args.dir_path+args.data_tag+'Data.txt')
  elif args.dir_path=='data/conll2000/':
    input_file_obj = openFile(args.dir_path+args.data_tag+'.txt.gz')
  elif args.dir_path =='data/conll2003/':
    input_file_obj = openFile(args.dir_path+'eng.'+args.data_tag)
  
  print 'start to load word2vec models!'
  trained_model = cPickle.load(open(args.use_model, 'rb'))
  print 'finish load cost time:',time.time()-start_time
  #print trained_model.wvec_model.vocab

  
  if args.dir_path == 'data/conll2000/':
    get_input(trained_model, args.model_dim, input_file_obj,
            args.dir_path+'features/'+args.data_tag+'_embed.p'+str(args.model_dim),
            args.dir_path+'features/'+args.data_tag+'_tag.p'+str(args.model_dim),
            sentence_length=args.sentence_length)
  elif args.dir_path == 'data/conll2003/':
    get_input_conll2003(trained_model, args.model_dim, input_file_obj,
            args.dir_path+'features/'+args.data_tag+'_embed.p'+str(args.model_dim),
            args.dir_path+'features/'+args.data_tag+'_tag.p'+str(args.model_dim),
            sentence_length=args.sentence_length)
  '''
  
  
  
  