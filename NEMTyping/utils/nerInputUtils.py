# -*- coding: utf-8 -*-
'''
@editor: wujs
revise: 2017/1/9
'''
import cPickle as pkl
import numpy as np


class nerInputUtils(object): 
  def __init__(self,dims,dir_path,flag='train'):
    self.dims = dims
    self.flag = flag
    self.dir_path=dir_path
    print 'load ner'+self.tag
    self.emb = np.asarray(pkl.load(open(self.dir_path+self.flag+'_embed.p'+self.dims,'rb')))
    self.tag = np.asarray(pkl.load(open(self.dir_path+self.flag+'_tag.p'+self.dims,'rb')))