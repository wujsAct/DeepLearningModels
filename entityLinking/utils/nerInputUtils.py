# -*- coding: utf-8 -*-
'''
@editor: wujs
revise: 2017/1/9
'''

import cPickle as pkl

class nerInputUtils(object):
  def __init__(self,dims,flag='train'):
    dir_path ='/home/wjs/demo/entityType/informationExtract/data/aida/features'
    if flag =='train':
      self.TFfileName = dir_path+'/train_embed.p'+dims+'.tfrecords'
      self.nerShapeFile = dir_path+'/train_embed.p'+dims+'.shape'
    elif flag=='testa':
      self.emb = pkl.load(open(dir_path+'/test_a_embed.p'+dims,'rb'))  #相对来说，数据量小了很多呢！
      self.tag = pkl.load(open(dir_path+'/test_a_tag.p'+dims,'rb'))
    else:
      self.emb = pkl.load(open(dir_path+'/test_b_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/test_b_tag.p'+dims,'rb'))
  
  