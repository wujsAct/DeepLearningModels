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
      print 'load ner train...'
#      self.TFfileName = dir_path+'/train_embed.p'+dims+'.tfrecords'
#      self.nerShapeFile = dir_path+'/train_embed.p'+dims+'.shape'
      self.emb = pkl.load(open(dir_path+'/train_embed.p'+dims,'rb')) 
      self.tag = pkl.load(open(dir_path+'/train_tag.p'+dims,'rb'))
    elif flag=='testa':
      print 'load ner testa...'
      self.emb = pkl.load(open(dir_path+'/test_a_embed.p'+dims,'rb')) 
      self.tag = pkl.load(open(dir_path+'/test_a_tag.p'+dims,'rb'))
    else:
      print 'load ner testb'
      self.emb = pkl.load(open(dir_path+'/test_b_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/test_b_tag.p'+dims,'rb'))
  