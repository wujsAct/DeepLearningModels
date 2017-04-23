# -*- coding: utf-8 -*-
'''
@editor: wujs
revise: 2017/1/9
'''
import cPickle as pkl
import numpy as np
import time

class nerInputUtils(object):
  def __init__(self,dims,dir_path,flag='train'):
    dims = "100"
    
    if flag =='train':
      stime = time.time()
      print 'load ner train...'
#      self.TFfileName = dir_path+'/train_embed.p'+dims+'.tfrecords'
#      self.nerShapeFile = dir_path+'/train_embed.p'+dims+'.shape'
      self.emb = pkl.load(open(dir_path+'train_embed.p'+dims,'rb')) 
      self.tag = pkl.load(open(dir_path+'train_tag.p'+dims,'rb'))
      print 'load train cost time:',time.time()-stime
    elif flag=='testa':
      print 'load ner testa...'
      stime = time.time()
      self.emb = pkl.load(open(dir_path+'testa_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'testa_tag.p'+dims,'rb'))
      print 'load testa cost time:',time.time()-stime
    elif flag=='testb':
      print 'load ner testb'
      stime = time.time()
      self.emb = pkl.load(open(dir_path+'testb_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'testb_tag.p'+dims,'rb'))
      print 'load testb cost time:',time.time()-stime
    else:
      self.emb = pkl.load(open(dir_path+flag+'_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+flag+'_tag.p'+dims,'rb'))
    print np.shape(np.array(self.emb))
    print np.shape(np.array(self.tag))