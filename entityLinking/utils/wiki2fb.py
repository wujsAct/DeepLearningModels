# -*- coding: utf-8 -*-
import cPickle
import codecs
import collections
from tqdm import tqdm

def getWikidata2Freebase(fname="/home/wjs/demo/entityType/informationExtract/data/fb2w.nt"):
  w2fb = {}
  with open(fname,'r') as file:
    for line in file:
      line = line.strip()
      items = line.split('\t')
      if len(items)==3:
        fbId = items[0].replace('<http://rdf.freebase.com/ns/m.','/m/').replace('>',''); 
        wId = items[2].split('/')[-1].replace('> .','').strip()
        w2fb[wId] = fbId
  return w2fb
  
def getWikititle2Freebase(fname="/home/wjs/demo/entityType/informationExtract/data/mid2name.tsv"):  
  wikititle2fb = collections.defaultdict(list)
  with open(fname,'r') as file:
    for line in tqdm(file):
      line = line.strip()
      items = line.split('\t')
      if len(items) >=2:
        fbId = items[0]; title = ' '.join(items[1:]).lower() #lower case 
        if fbId not in wikititle2fb[title]:
          wikititle2fb[title].append(fbId)  #one title may has more than ids...
  return wikititle2fb
  
#wid2fbid =  getWikidata2Freebase()
wtitle2fbid  = getWikititle2Freebase()
#cPickle.dump(wid2fbid,open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','wb'))
cPickle.dump(wtitle2fbid,open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','wb'))