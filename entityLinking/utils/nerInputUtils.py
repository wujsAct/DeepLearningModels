# -*- coding: utf-8 -*-
'''
@editor: wujs
revise: 2017/1/9
'''
import numpy as np
import cPickle as pkl


    
'''
@we need to padding the sentences into the same length!
'''
class nerInputUtils(object):
  '''
  @sentence_final: shape:(batch_size,sequence_length,dims)
  '''
  def padZeros(self,sentence_final,dims=110,max_sentence_length=250):
    for i in range(len(sentence_final)):
      offset = max_sentence_length-len(sentence_final[i])
      sentence_final[i] =np.concatenate((sentence_final[i], [[0]*dims]*offset))
    return np.asarray(sentence_final)

  def __call__(self,dims,flag='train'):
    dir_path ='/home/wjs/demo/entityType/informationExtract/data/aida/features'
    print 'load ner'+flag+"..."
    self.emb = self.padZeros(pkl.load(open(dir_path+'/'+flag+'_embed.p'+dims,'rb')))
    self.tag = self.padZeros(pkl.load(open(dir_path+'/'+flag+'_tag.p'+dims,'rb')),5)
    
  def iterate(self,batch_size,emb,tag):
    num_shape = np.shape(emb)[0]
    print num_shape,batch_size
    if num_shape <=batch_size:
      yield emb,tag
    else:
      ret_emb=[];ret_tag=[]
      nk = 0
      for i in range(num_shape):
        ret_emb.append(emb[i])
        ret_tag.append(tag[i])
        nk += 1
        if (nk % batch_size==0 and nk!=0) or (i==num_shape-1 and np.shape(ret_emb)[0] !=0):
          yield np.asarray(ret_emb,dtype=np.float32),np.asarray(ret_tag,dtype=np.int32)
          ret_emb=[];ret_tag=[];nk=0
    