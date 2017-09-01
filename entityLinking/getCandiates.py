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

def getCandEntsByWiki(searchent):
  urllibutil = urllibUtils()
  candidate_ent = []
 # candentSet1=[];candentSet2=[]
#  '''
#  @top 1
#  '''
#  try:
#    candidate_ent = urllibutil.get_candidateWID_entities(searchent,num='1')
#  except urllib2.URLError,e:
#    pass
#  
#  '''
#  @top 3
#  '''
#  try:
#    candentSet1 = urllibutil.getDirectFromWikiPage(searchent)
#  except:
#    pass
#    
  '''
  @we need to add all the disambiugation pages(50)
  '''
  try:
    candidate_ent += urllibutil.opensearchApi(searchent)
  except:
    pass
#  '''
#  @top 10 
#  '''
#  try:
#    candentSet2 = urllibutil.parseEntCandFromWikiSearch(searchent)  #words matching!
#  except:
#    pass
#  for ent in candentSet1:
#    if ent not in candidate_ent:
#      candidate_ent.append(ent)
#  
#  for ent in candentSet2:
#    if ent not in candidate_ent:
#      candidate_ent.append(ent)
  return candidate_ent

def funcs(ids,id2entstr,lent):
  entstr = id2entstr[ids]
  searchent = entstr.title()
  candidate_ent = []
  #candidate_ent,co_occurence_ent = getCanEnts(searchent)  #solve Metonymy problem!
  candidate_ent = getCandEntsByWiki(searchent)
  
  if len(candidate_ent)==0:
      print 'have no candidate_ent'
      for i in ('\''):
        entstr = entstr.replace(i,u' ')
      entstr = u' '.join(entstr.split(u' ')[1:])
      #print 'entstr:',entstr
      candidate_ent = getCandEntsByWiki(entstr.title())
     
  print 'ids:',ids,' totalids:',lent,' original:',searchent,' entstr:',entstr,'\t',len(candidate_ent)
  return [ids,candidate_ent]

#print candidate_ent
candidate_ent = getCandEntsByWiki("New York")
print len(candidate_ent)
for key in candidate_ent:
  print key
#candidate_ent = getCandEntsByWiki("Schindler 'S List")
#print 'candidate_ent:',candidate_ent
'''
if __name__=='__main__':
  if len(sys.argv) !=4:
    print 'usage: python pyfile dir_path inputfile outputfile'
    exit(1)
  #grep 'core id' /proc/cpuinfo | sort -u|wc -l
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
#  ids = entstr2id[u"Englishman"]
#  funcs(ids,id2entstt)
##  ids = entstr2id[u"German"]
##  funcs(ids,id2entstr)
  
  
  candiate_ent=[None]*lent
  candiate_coCurrEnts=[None]*lent
  result = []
  
  for ptr in xrange(0,lent,30):
    pool = multiprocessing.Pool(processes=6)
    for ids in xrange(ptr,min(ptr+30,lent)):
      #print '----------------------'
      #print ids,entstr
      result.append(pool.apply_async(funcs, (ids,id2entstr,lent)))
    pool.close()
    pool.join()
    
    for ret in result:
      retget = ret.get()
      ids = retget[0];candidate_ent_i=retget[1]#;co_occurence_ent_i=retget[2]
      candiate_ent[ids] = candidate_ent_i
      #candiate_coCurrEnts[ids] = co_occurence_ent_i
  
  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent}
  cPickle.dump(para_dict,open(f_output,'wb'))
'''