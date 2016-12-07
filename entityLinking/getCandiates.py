# -*- coding: utf-8 -*-

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import multiprocessing
import urllib2
import codecs
import cPickle
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
from urllibUtils import urllibUtils

def getCanEnts(searchent):
  urllibutil = urllibUtils()
  candidate_ent=[];co_occurence_ent=[];
  try:
    metonymyflag,candidate_ent,co_occurence_ent = urllibutil.get_candidate_entities(searchent,num='5')   #words matching!
    candentSet1=set();candentSet2=set();candentSet=set()
    if metonymyflag:   #sometimes may have troubles!
      candentSet1 = urllibutil.getDirectFromWikiPage(searchent)
    candentSet2 = urllibutil.parseEntCandFromWikiSearch(searchent)  #words matching!
    candentSet = candentSet1 | candentSet2
    if searchent in candentSet:
      candentSet.remove(searchent)
    for candenti in candentSet:
      metonymyflag,candidate_enti,co_occurence_enti = urllibutil.get_candidate_entities(candenti,num='1')
      candidate_ent += candidate_enti
      co_occurence_ent += co_occurence_enti
  except urllib2.URLError,e:
    print e.reason
    pass
  return candidate_ent,co_occurence_ent

def funcs(ids,id2entstr,lent):
  entstr = id2entstr[ids]
  searchent = entstr.title()
  
  candidate_ent,co_occurence_ent = getCanEnts(searchent)  #solve Metonymy problem!
  
  if len(candidate_ent)==0:
      print 'have no candidate_ent'
      for i in ('\''):
        entstr = entstr.replace(i,u' ')
      entstr = u' '.join(entstr.split(u' ')[1:])
      #print 'entstr:',entstr
      candidate_ent,co_occurence_ent = getCanEnts(entstr.title())
     
  print 'ids:',ids,' totalids:',lent,' original:',searchent,' entstr:',entstr,'\t',len(candidate_ent),len(co_occurence_ent)
  return [ids,candidate_ent,co_occurence_ent]

  
if __name__=='__main__':
  if len(sys.argv) !=4:
    print 'usage: python pyfile dir_path inputfile outputfile'
    exit(1)
  #进程池最好设置成CPU核心数量(cpu core), 不然可能出产生一奇怪的错误！
  #grep 'core id' /proc/cpuinfo | sort -u|wc -l
  pool = multiprocessing.Pool(processes=8)
  dir_path = sys.argv[1]
  f_input = dir_path  + sys.argv[2]
  f_output = dir_path + sys.argv[3]
  # f_input context: para_dict={'aNosNo2id':aNosNo2id,'id2aNosNo':id2aNosNo,'sents':sents,'tags':tags,'ents':ents,'depTrees':depTrees}
  para_dict = cPickle.load(open(f_input,'r'))
  ents = para_dict['ents']
  entstr2id = {}
  id2entstr = {}
  
  entsSet=set()
  entId=0
  for entitem in ents:
    entlist =  entitem[0]
    for enti in entlist:
      entstr =  enti.content
      if entstr not in entstr2id:
        entstr2id[entstr] = entId
        id2entstr[entId] = entstr
        entId = entId + 1
  print id2entstr[22]
  lent = len(id2entstr)
#  ids = entstr2id[u"British"]
#  funcs(ids,id2entstr)
#  ids = entstr2id[u"German"]
#  funcs(ids,id2entstr)
  

  candiate_ent=[None]*lent
  candiate_coCurrEnts=[None]*lent
  result = []
  for ids in xrange(lent):
    #print '----------------------'
    #print ids,entstr
    result.append(pool.apply_async(funcs, (ids,id2entstr,lent)))
  pool.close()
  pool.join()
  
  for ret in result:
    retget = ret.get()
    ids = retget[0];candidate_ent_i=retget[1];co_occurence_ent_i=retget[2]
    candiate_ent[ids] = candidate_ent_i
    candiate_coCurrEnts[ids] = co_occurence_ent_i
  
  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
  cPickle.dump(para_dict,open(f_output,'wb'))
  