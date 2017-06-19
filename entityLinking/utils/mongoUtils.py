# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 10:01:01 2016

@author: wujs
"""
import codecs
from pymongo import MongoClient

def get_mediator_relation():
  return mediator

class mongoUtils(object):
  def __init__(self):
    client = MongoClient('mongodb://192.168.3.196:27017')
    #print client
    self.db = client.freebase_full # database name
    self.freebase = self.db['freebase']  # collections
    fileName='/home/wjs/demo/entityType/informationExtract/data/mediator-relations'
    self.mediator= {}
    meids = 0
    with codecs.open(fileName) as file:
      for line in file:
        line = line.strip()
        rel = u'<http://rdf.freebase.com/ns/'+line+u'>'
        self.mediator[rel] = meids
        meids += 1
  def getHasRel(self,midi,midj):
    midi = u'<http://rdf.freebase.com/ns/'+midi.replace('/m/','m.')+'>'
    midj = u'<http://rdf.freebase.com/ns/'+midj.replace('/m/','m.')+'>'
    ret = self.freebase.find({'head':midi,'tail':midj}).limit(1000)
    '''
    @reduce the searching times!
    '''
    if ret.count()>0:
      return True
    else:
      ret1 = self.freebase.find({'head':midj,'tail':midi}).limit(1000)
      if ret1.count()>0:
        return True
      else:
        return False
    
  def get_tail_from_enwikiTitle(self,title):
    title1 = '\"'+title+'\"'
    #print title1
    mids = set()
    wikiRel = '<http://rdf.freebase.com/key/wikipedia.en>'
    for item in self.freebase.find({'rel':wikiRel,'tail':title1}):
      mid = item['head'].replace('<http://rdf.freebase.com/ns/m.','/m/')[0:-1]
      mids.add(mid)
    if len(mids)==0:
      rel2 = "<http://rdf.freebase.com/ns/type.object.key>"
      title2 = '\"/wikipedia/en/'+title+'\"'
      #print title2
      for item in self.freebase.find({'rel':rel2,'tail':title2}):
        mid = item['head'].replace('<http://rdf.freebase.com/ns/m.','/m/')[0:-1]
        mids.add(mid)
    return mids
      
      
  def get_coOccurent_ents(self,ent):
    coents_dict={}
    enttag=u'<http://rdf.freebase.com/ns/m.'
    
    #print ent.split('/')[-1]#,self.freebase.count({'head':ent}).limit(10000)#,self.freebase.count({'tail':ent})
    #print self.freebase.count({'head':ent})
    '''
    @function:从前往后寻找共现的实体，出现cvt点的话，往后找一跳的结果！
    '''
    for item in self.freebase.find({'head':ent}).limit(10000):
      try:
        head = item['head']; rel= item['rel']; tail=item['tail']
        if rel not in self.mediator and enttag in tail:
          coents_dict[tail]=1
#        if rel in self.mediator:
#          #print 'rel:',rel
#          for item2 in self.freebase.find({'head':tail}):
#            head1 = item2['head']; rel1= item2['rel']; tail1=item2['tail'] 
#            if enttag in tail1 and rel1 not in self.mediator:
#              coents.add(tail1)
      except:
        print 'mongo api wrong'
        pass
    return coents_dict
