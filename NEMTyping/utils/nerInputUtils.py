# -*- coding: utf-8 -*-
'''
@editor: wujs
revise: 2017/1/9
'''
import cPickle as pkl
import numpy as np
import gc
import time

def load_cpickle_gc(pklFile):
  output = open(pklFile, 'rb')

  # disable garbage collector
  gc.disable()

  data = pkl.load(output)
  
  gc.enable()
  
  output.close()
  # enable garbage collector again
  return data
    
class nerInputUtils(object): 
  def __init__(self,dims,dir_path,flag='train'):
    start_time = time.time()
    self.dims = dims
    self.flag = flag
    self.dir_path=dir_path
    print 'load ner '+self.flag
    #self.emb = np.asarray(pkl.load(open(self.dir_path+self.flag+'_embed.p'+self.dims,'rb')))
    #self.tag = np.asarray(pkl.load(open(self.dir_path+self.flag+'_tag.p'+self.dims,'rb')))
    self.emb = load_cpickle_gc(self.dir_path+self.flag+'_embed.p'+self.dims)
    self.tag = load_cpickle_gc(self.dir_path+self.flag+'_tag.p'+self.dims)
    print 'load ',self.flag, ' data cost time:',time.time()-start_time
    