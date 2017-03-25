#encoding=utf-8
__author__ = 'hw'
from pymongo import MongoClient
import math

class NGDUtils(object):
  def __init__(self):
    client = MongoClient('mongodb://192.168.3.196:27017',connect=False)
    self.db = client.wiki #database name
    self.collect = self.db['pagelinks']  #collections
    
  def getLinkedEnts(self,title):
    coents=set()
    for item in self.collect.find({'pl_title':title}).limit(10000):
      linkId = item['pl_from']
      coents.add(linkId)
    #print coents
    return coents
  
  def get_page_links_2(self,list1,list2):
    sr_scores = self.semantic_relatedness(list1, list2)
    return sr_scores
  def get_page_links(self,entity_str1, entity_str2):
    list1 = self.getLinkedEnts(entity_str1)
    list2 = self.getLinkedEnts(entity_str2)
    sr_scores = self.semantic_relatedness(list1, list2)
    return sr_scores


  def semantic_relatedness(self,list1, list2, c=9568051):
    a = max(len(list1), len(list2))
    b = len(set(list1) & set(list2))
    d = min(len(list1), len(list2))
    #c = 41657481 #all wikipedia pages
    if a!=0 and b!=0 and d!=0:
      res = 1- ((math.log(a*1.0)-math.log(b*1.0)) / (math.log(c*1.0)-math.log(d*1.0)))
    else:
      res = 0.0
    return res

if __name__ == '__main__':
  ngd = NGDUtils()
  sr = ngd.getLinkedEnts("Surrey Lions")
  print len(sr)
#  sr = ngd.get_page_links('France','France in the Middle Ages')
#  print sr
#  sr = ngd.get_page_links('England','England national football team')
#  
#  
#  print 'England','England national football team',sr
#  
#  sr = ngd.get_page_links('England','England cricket team')
#  print sr
#  
#  sr = ngd.get_page_links('England','England national rugby union team')
#  print sr