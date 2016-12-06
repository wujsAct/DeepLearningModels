# -*- coding: utf-8 -*-
import cPickle
import codecs

def getWikidata2Freebase(fname="/home/wjs/demo/entityType/informationExtract/data/fb2w.nt"):
  w2fb = {}
  with codecs.open(fname,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split('\t')
      if len(items)==3:
        fbId = items[0]; wId = items[2].split(u'/')[-1].replace(u'> .',u'').strip()
        w2fb[wId] = fbId
  return w2fb
  
def getWikititle2Freebase(fname="/home/wjs/demo/entityType/informationExtract/data/mid2name.tsv"):
  wikititle2fb = {}
  with codecs.open(fname,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split('\t')
      if len(items)==2:
        fbId = items[0]; title = items[1]
        wikititle2fb[title] = fbId
  return wikititle2fb
  
wid2fbid =  getWikidata2Freebase()
wtitle2fbid  = getWikititle2Freebase()
cPickle.dump(wid2fbid,open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','wb'))
cPickle.dump(wtitle2fbid,open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','wb'))


