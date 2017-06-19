# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:12:59 2017

@author: wujs
@function: load data for 
"""
import collections
from tqdm import tqdm


def getMid2WikiTitle():
  fname = 'data/mid2name.tsv'
  wikititle2fb = collections.defaultdict(list)
  fb2wikititle={}
  with open(fname,'r') as file:
    for line in tqdm(file):
      line = line.strip()
      items = line.split('\t')
      if len(items)==2:
        fbId = items[0]; title = items[1]  
        fb2wikititle[fbId] = title.lower()
        wikititle2fb[title.lower()].append(fbId)
  fb2wikititle['NIL'] = 'NIL'
  
  return wikititle2fb,fb2wikititle
  