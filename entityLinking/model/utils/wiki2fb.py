# -*- coding: utf-8 -*-
import cPickle
import codecs
import collections

def getWikidata2Freebase(fname="/home/wjs/demo/entityType/informationExtract/data/fb2w.nt"): #给出的真是答案呢！
  w2fb = {}
  with codecs.open(fname,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split('\t')
      if len(items)==3:
        fbId = items[0].replace('<http://rdf.freebase.com/ns/m.','/m/').replace(u'>',u''); 
        wId = items[2].split(u'/')[-1].replace(u'> .',u'').strip()
        w2fb[wId] = fbId
  return w2fb
  
def getWikititle2Freebase(fname="/home/wjs/demo/entityType/informationExtract/data/mid2name.tsv"):  
  wikititle2fb = collections.defaultdict(list)
  with codecs.open(fname,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split('\t')
      if len(items)==2:
        fbId = items[0]; title = items[1].lower()  #所有地方均采用小写字母啦！
        wikititle2fb[title].append(fbId)  #没法倒着来，因为一个mid可能对应着很多不用的alias！
  return wikititle2fb
  
#wid2fbid =  getWikidata2Freebase()
wtitle2fbid  = getWikititle2Freebase()
#cPickle.dump(wid2fbid,open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','wb'))
cPickle.dump(wtitle2fbid,open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','wb'))