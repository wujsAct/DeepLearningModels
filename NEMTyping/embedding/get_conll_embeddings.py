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
def getFigerEntTags(entList,sid,ent_no):
  ent_mention_mask=[]
  type_indices=[]
  type_val=[]
  for i in range(len(entList)):
    ent = entList[i]
    ent_start= int(ent[0])
    ent_end = int(ent[1])
    typeList = sorted(list(set(ent[2])))  #ascending sort and duplicate type remove!
    '''
    sparse pattern
    '''
    ent_mention_mask.append([sid,ent_start,ent_end])
    for t in range(len(typeList)): 
      ind =[ent_no,t]  # batch_id,sequence_length_id,class_id
      type_indices.append(ind)
      type_val.append(1)

    ent_no += 1
  
  return ent_no,ent_mention_mask,type_indices,type_val




def get_input_figer_chunk(batch_size,dir_path,set_tag,model,word_dim,sentence_length=-1):
  epochs = 1
  figerTypes = 113
  input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  figer2id = cPickle.load(open(dir_path+"figer2id.p",'rb'))
  otherType=len(figer2id)
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
#        for _ in range(max_sentence_length - sentence_length):
#          #tag.append(np.array([0] * 5))
#          temp = np.array([0 for _ in range(word_dim + 6+5)])
#          
#          word.append(temp)
        entList = entMents[sid]
        ent_no,temp_ent_mention_mask,temp_type_indices,temp_type_val = getFigerEntTags(entList,allid%batch_size,ent_no)
        sentence.append(word)
        ent_mention_mask += temp_ent_mention_mask
        type_indices += temp_type_indices
        type_val += temp_type_val
        if (allid+1)%batch_size==0 and allid!=0:
          if len(sentence) == batch_size:
            sentence_final.append(sentence)
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
        temp = model[line.split()[0]]
        temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
        temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
        temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
        assert len(temp) == word_dim+6+5
        word.append(temp)
  return ent_mention_mask_final,sentence_final,type_final

def get_input_figer_chunk_train(batch_size,dir_path,set_tag,model,word_dim,sentence_length=-1):
  
  epochs = 2
  figerTypes = 113
  input_file_obj = open(dir_path+'features/figerData_'+set_tag+'.txt')
  
  entMents = cPickle.load(open(dir_path+'features/'+set_tag+'_entMents.p','rb'))
  figer2id = cPickle.load(open(dir_path+"figer2id.p",'rb'))
  otherType=len(figer2id)
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
#        for _ in range(max_sentence_length - sentence_length):
#          #tag.append(np.array([0] * 5))
#          temp = np.array([0 for _ in range(word_dim + 6+5)])
#          
#          word.append(temp)
        entList = entMents[sid]
        ent_no,temp_ent_mention_mask,temp_type_indices,temp_type_val = getFigerEntTags(entList,allid%batch_size,ent_no)
        sentence.append(word)
        ent_mention_mask += temp_ent_mention_mask
        type_indices += temp_type_indices
        type_val += temp_type_val
        if (allid+1)%batch_size==0 and allid!=0:
          if len(sentence) == batch_size:
            yield ent_mention_mask,sentence,[np.asarray(temp_type_indices, dtype=np.int64),np.asarray(temp_type_val, dtype=np.float32)]
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
        temp = np.zeros((100,))
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
def genEntMentMask(entment_mask_final):
  entNums = len(entment_mask_final)
  entment_masks = np.zeros((entNums,256,80,256))
  for i in range(entNums):
    items = entment_mask_final[i]
    
    ids = items[0];start=items[1];end=items[2]
    for ient in range(start,end):
        entment_masks[i,ids,ient] = np.asarray([1]*256)
  return entment_masks
def padZeros(sentence_final,max_sentence_length=80,dims=111):
  for i in range(len(sentence_final)):
    offset = max_sentence_length-len(sentence_final[i])
    sentence_final[i] += [[0]*dims]*offset
    
  return np.asarray(sentence_final)
if __name__ == '__main__':
  
  i = 0
  for train_entment_mask,train_sentence_final,train_tag_final in get_input_figer_chunk_train(256,'data/figer/',"train",model=None,word_dim=100,sentence_length=80):
    indexs = genEntMentMask(train_entment_mask)
    print np.shape(padZeros(train_sentence_final))
    print len(indexs)
  
  #word2vecModel = cPickle.load(open('data/wordvec_model_100.p', 'rb'))
  #word2vecModel = None
  #output_data = tf.sparse_placeholder(tf.float32, name='outputdata')
  #sess = tf.InteractiveSession()
  '''
  @sparse_tensor need to be in order and non duplicate elements!
  '''
#  stime = time.time()
#  for train_input,train_out in get_input_figer_chunk('data/figer/',"testa",model=word2vecModel,word_dim=100,sentence_length=80):
#    print np.shape(train_input)
#    print len(train_out[0])
#    print len(train_out[1])
#    print train_out[2]
#    tt = sess.run(tf.sparse_tensor_to_dense(output_data), feed_dict={output_data:tf.SparseTensorValue(train_out[0],train_out[1],train_out[2])})
#    print tt[0]
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
  
  
  
  