# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:35:21 2016

@author: DELL
"""
import logging
import time
import pickle
from nltk.corpus import stopwords
from tqdm import *
import random
import numpy
import numpy as np 
import os
import sys
from glob import glob

cachedStopWords = stopwords.words("english")
_ENTITY = "@entity"

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class QADataset():
    def __init__(self,dir_path,dataset_name,vocab_file, **kwargs):
        self.path = dir_path
        self.dataset_name = dataset_name
        self.vocab_file = vocab_file
        self.concat_question_before = False
        self.padding=0
        self.query_lent = 50
        self.context_lent=1000
        self.contextAndquery_lent = 1000
    #lacking of the basic computing skill  
    def create_vocabulary(self):
        if not os.path.exists(self.vocab_file):
            entset = set()
            ignoreWord = cachedStopWords + ['<UNK>', '@placeholder']
            ignoreWordDict = dict(zip(ignoreWord,range(len(ignoreWord))))
            t0 = time.time()
            print "creating vocabulary %s" %(self.vocab_file)
            texts =set()
            
            i=0
            for fname in glob(os.path.join(self.path,self.dataset_name,'questions/training','*.question')):
                with open(fname) as f:
                    try:
                        context=''
                        lines = f.read().split('\n')
                        context += lines[2] + ' '
                        context += lines[4] + ' '
                        i = i + 1
                         #english word is splited by ' '
                        for word in context.lower().split(' '):
                            if '@entity' in word:
                                entset.add(word)
                            elif word not in ignoreWordDict:
                                texts.add(word)
                            else:
                                continue
                        if i % 1000 ==0:
                            print 'no:',i
                    except:
                        print(" [!] Error occured for %s" % fname)
                        #exit(-1)
            entList=['@entity%d' %(i) for i in range(len(entset))]
            vocab = entList+list(texts) + ['<UNK>', '@placeholder']
            dicts = dict(zip(vocab, range(len(vocab))))
            with open(self.vocab_file, 'wb') as handle:
                pickle.dump(dicts, handle)
            print "tokenizer: %.4fs" %(t0-time.time())
            
    def initialize_vocabulary(self):
        entNum = 0
        if os.path.exists(self.vocab_file):
            dicts={}
            with open(self.vocab_file, 'rb') as handle:
                dicts =  pickle.loads(handle.read())
            for keys in dicts:
                if '@entity' in keys:
                    entNum += 1
                else:
                    continue
            print entNum
            return dicts,entNum
        else:
            raise ValueError("vocabulary file %s not found",self.vocab_file)
            
    def to_word_id(self, w, cand_mapping):
        if w in cand_mapping:
            return cand_mapping[w]
        elif w[:7] == '@entity':
            raise ValueError("Unmapped entity token: %s"%w)
        elif w in self.vocab2id:
            return self.vocab2id[w]
        else:
            return self.vocab2id['<UNK>']
                
    def to_word_ids(self, s, cand_mapping):
        return [self.to_word_id(x, cand_mapping) for x in s.split(' ')]
    
    def get_data(self,fname):
        lines = [l.rstrip('\n') for l in open(fname)]
        
        ctx = lines[2]
        q = lines[4]
        a = lines[6]
        cand = [s.split(':')[0] for s in lines[8:]]
        entities = self.n_entities
        
        while len(cand) > entities:
            exit(-1)
            logger.warning("Too many entities (%d) for question: %s, using duplicate entity identifiers"
                %(len(cand), fname))
            entities = entities + entities
        entities = range(entities)
        random.shuffle(entities)
        cand_mapping = {t:k for t,k in zip(cand,entities)}
        
        ctx = self.to_word_ids(ctx,cand_mapping)
        q = self.to_word_ids(q,cand_mapping)
        cand =[self.to_word_id(x, cand_mapping) for x in cand]
        a = self.to_word_id(a,cand_mapping)
        if not a < self.n_entities:
            raise ValueError("Invalid answer token %d"%a)
        if not np.all(np.asarray(cand) < self.n_entities):
            raise ValueError("Invalid candidate in list %s"%repr(cand))
        if not np.all(np.asarray(ctx) < self.vocab_size):
            raise ValueError("Context word id out of bounds: %d"%int(ctx.max()))
        if not np.all(ctx >= 0):
            raise ValueError("Context word id negative: %d"%int(ctx.min()))
        if not np.all(np.asarray(q) < self.vocab_size):
            raise ValueError("Question word id out of bounds: %d"%int(q.max()))
        if not np.all(q >= 0):
            raise ValueError("Question word id negative: %d"%int(q.min()))

        return (ctx, q, a, cand)
        
    def prepare_data(self):
        vocab_fname = self.vocab_file
        
        if not os.path.exists(vocab_fname):
            print(" [*] Create vocab to %s ..." % (vocab_fname))
            self.create_vocabulary()
        else:
            print " [*] Skip creating vocab"
            
    def write_QueryAndContext(self,batch_size):
        train_files = glob(os.path.join(self.path,self.dataset_name,'questions/training' ,'*.question'))
        
        self.vocab2id,self.n_entities = self.initialize_vocabulary()
        self.vocab_size = len(self.vocab2id)
        final_txt = open(os.path.join(self.path,self.dataset_name,'train_final_1000_padding'+str(self.padding)+'.txt'),'w')
        #max_idx = len(train_files)
        total = 1
        for idx,fname in enumerate(train_files):
            total += 1
            (ctx, q, a, cand) = self.get_data(fname)
            if not self.concat_question_before:
                data = ctx + q
            else:
                data = q + ctx
            lent = len(data)
            if lent>1000:
                continue
            else:
                data = map(str,data)+[str(self.vocab2id['<UNK>'])]*(self.context_lent-lent)
                #adding the candidate entity information
                #cand_r = map(str,cand) +[str(self.padding)]*(self.n_entities-len(cand))
                cand_r = []
                for i in range(self.n_entities):
                    if i in cand:
                        cand_r.append(str(i))
                    else:
                        cand_r.append(str(self.vocab2id['<UNK>']))
                dataf = ' '.join(data) +'\t'+str(lent)+'\t'+str(a)+'\t'+' '.join(cand_r)+'\t'+str(len(cand))
                final_txt.write(dataf+'\n')
            if total %10000==0:
                print total
        final_txt.close()
        

    def write_SeperateQueryAndContext(self,batch_size):
        train_files = glob(os.path.join(self.path,self.dataset_name,'questions/training' ,'*.question'))
        
        self.vocab2id,self.n_entities = self.initialize_vocabulary()
        self.vocab_size = len(self.vocab2id)
        CtxQueryFile = open(os.path.join(self.path,self.dataset_name,'train_context_query_'+ str(self.query_lent+self.context_lent)+'_padding'+str(self.padding)+'.txt'),'w')   
        maxquery=0
        #max_idx = len(train_files)
        total = 1
        for idx,fname in enumerate(train_files):
            total += 1
            (ctx, q, a, cand) = self.get_data(fname)
            if not self.concat_question_before:
                data = ctx + q
            else:
                data = q + ctx
            lent = len(data)
            if lent>1000:
                continue
            else:
                if len(q) > maxquery:
                  maxquery = len(q)
                #data = map(str,data)+[str(self.vocab2id['<UNK>'])]*(self.context_lent-lent)
                #adding the candidate entity information
                #cand_r = map(str,cand) +[str(self.padding)]*(self.n_entities-len(cand))
                cand_r = []
                for i in range(self.n_entities):
                    if i in cand:
                        cand_r.append(str(i))
                    else:
                        cand_r.append(str(self.vocab2id['<UNK>']))
                query = map(str,q)+[str(self.vocab2id['<UNK>'])]*(self.query_lent-len(q))
                context = map(str,ctx)+[str(self.vocab2id['<UNK>'])]*(self.context_lent-len(ctx))
                CtxQueryFile.write(' '.join(context)+'\t'+str(len(ctx))+'\t'+' '.join(query)+'\t'+str(len(q))+'\t'+str(a)+'\t'+' '.join(cand_r)+'\t'+str(len(cand))+'\n')
            if total %10000==0:
                print total
        print('max question',maxquery)
        CtxQueryFile.close()
                
    def load_dataset1(self,batch_size):
        final_txt = open(os.path.join(self.path,self.dataset_name,'train_final_1000_padding'+str(self.padding)+'.txt'),'r')
        _,self.n_entities = self.initialize_vocabulary()
        total = 0
        x = np.zeros([batch_size,1000])
        cand_x = np.zeros([batch_size,self.n_entities])
        #y = np.zeros([batch_size, self.n_entities])
        y = []
        input_length = []
        #cand_length=[]
        for line in final_txt:
            line = line.strip()
            #print(line)
            item = line.split('\t')
            docquery = map(int,item[0].split(' '))
            answer = int(item[2])
            cand = map(int,item[3].split(' '))
            if total == batch_size:
                yield x,np.asarray(y,dtype='int32'),np.array(input_length,dtype='int32'),cand_x#,np.array(cand_length,dtype='int32')
                x = np.zeros([batch_size,1000])
                cand_x = np.zeros([batch_size,self.n_entities])
                #y = np.zeros([batch_size, self.n_entities])
                y=[]
                total =0
                input_length=[]
                #cand_length=[]
                
            #for i in range(int(item[1])):
            x[total] = docquery
            #for i in range(int(item[4])):
            cand_x[total] = cand
            #y[total][int(answer)] = 1
            y.append(int(answer))
            input_length.append(int(item[1]))
            #cand_length.append(int(item[4]))
            total +=1
            
    def load_dataset2(self,batch_size):
        CtxQueryFile = open(os.path.join(self.path,self.dataset_name,'train_context_query_'+ str(self.query_lent+self.context_lent)+'_padding'+str(self.padding)+'.txt'),'r')
        _,self.n_entities = self.initialize_vocabulary()
        total = 0
        q = np.zeros([batch_size,self.query_lent])
        ctx = np.zeros([batch_size, self.context_lent])
        ctx_input_length = []
        q_input_length = []
        y = []
        for line in CtxQueryFile:
            line = line.strip()
            item = line.split('\t')
            #print len(item)
            ctx_data = map(int,item[0].split(' '))
            q_data = map(int,item[2].split(' '))
            answer = int(item[4])
            if total == batch_size:
                yield ctx,np.asarray(ctx_input_length),q,np.asarray(q_input_length),np.array(y)
                q = np.zeros([batch_size,self.query_lent])
                ctx = np.zeros([batch_size, self.context_lent])
                ctx_input_length = []
                q_input_length = []
                y=[]
                total = 0
    
            q[total] = q_data
            ctx[total] = ctx_data
            y.append(int(answer))
            ctx_input_length.append(int(item[1]))
            q_input_length.append(int(item[3]))
            total +=1
            
if __name__ == '__main__':
  #264588  total words!
  if len(sys.argv) < 2:
    print(" [*] usage: python data_utils.py DATA_DIR DATASET_NAME VOCAB_SIZE")
  else:
    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    batch_size = 32
    print(data_dir)
    print(dataset_name)
    vocab_fname = os.path.join(data_dir,dataset_name,dataset_name+'.vocab')
    qaData = QADataset(data_dir,dataset_name,vocab_fname)
    qaData.prepare_data()
    #qaData.write_SeperateQueryAndContext(batch_size)
    #qaData.write_QueryAndContext(batch_size)
    t= qaData.load_dataset2(1)
    print t.next()  
        
        
        
    
        
        
