# -*- coding: utf-8 -*-
"""
Spyder Editor
@author:wujs
@time: 2017/2/7
"""
import codecs
import pycurl
import StringIO
import urllib
import json
from urllib import unquote

dir_path = 'testdata/ace/' 
'''read sents2aNosNo'''
sentid = 0
sentId2aNosNo = {}
with codecs.open(dir_path+'sentid2aNosNoid.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    sentId2aNosNo[sentid] = line
    sentid += 1


'''read entmention 2 aNosNoid'''
entsFile = dir_path+'entMen2aNosNoid.txt'
hasMid = 0
entMentsTags={}
entMents2surfaceName={}
with codecs.open(entsFile,'r','utf-8') as file:
  for line in file:
    line = line.strip()
    items = line.split('\t')
    entMent = items[0]; linkingEnt = items[1]; aNosNo = items[2]; start = items[3]; end = items[4]
    key = aNosNo + '\t' + start+'\t'+end
    hasMid += 1
    entMentsTags[key]=linkingEnt
    entMents2surfaceName[key] = entMent
print 'entMentsTags nums:',len(entMentsTags)


def getAidaNEL(sent,lineNo):
  post_data_dic = {"text":sent}
  crl = pycurl.Curl()
  crl.fp = StringIO.StringIO()
  # Option -d/--data <data>   HTTP POST data
  crl.setopt(crl.POSTFIELDS,  urllib.urlencode(post_data_dic))
  
  crl.setopt(pycurl.URL,'http://localhost:8080/aida/service/disambiguate')
  #crl.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json'])
  
  crl.setopt(crl.WRITEFUNCTION, crl.fp.write)
  crl.perform()
  ret =  json.loads(crl.fp.getvalue())
  ment_list = ret['mentions']
  metadata_dict = ret['entityMetadata']
  retNum=0
  for item_dict in ment_list:
    offset = item_dict['offset']
    length = item_dict['length']
    if 'bestEntity' in item_dict:
      entities = unquote(metadata_dict[item_dict['bestEntity']['kbIdentifier']]['url'])
      start_index =  len(sent[0:offset-1].split(u' '))
      end_index = start_index+len(sent[offset:offset+length-1].split(u' '))
      key = sentId2aNosNo[lineNo]+'\t'+str(start_index)+'\t'+str(end_index)
      if key in entMentsTags:
        print entities, entities.split(u'http://en.wikipedia.org/wiki/')[1],entMentsTags[key]
      if key in entMentsTags and entMentsTags[key] == entities.split(u'http://en.wikipedia.org/wiki/')[1]:
        retNum += 1
      #print offset,length,item_dict['name'],entities
      #print sent.split(u' ')[start_index:end_index]
      #print '----------------'
    else:
      start_index =  len(sent[0:offset-1].split(u' '))
      end_index = start_index+len(sent[offset:offset+length-1].split(u' '))
      key = sentId2aNosNo[lineNo]+'\t'+str(start_index)+'\t'+str(end_index)
      if key in entMentsTags and entMentsTags[key] =='NIL':
        retNum+=1
  return retNum
        
      #print offset,length,item_dict['name'],'NULL'
  
  
sent = ''
lineNo = 0
rightPredict=0
with codecs.open('testdata/ace/aceData.txt','r','utf-8') as file:
  for line in file:
    line = line.strip()
    word = line.split(u'\t')[0]
    
    if word!=u'':
      sent = sent + word+u' '
    else:
      sent = sent + word
      print lineNo
      rets = getAidaNEL(sent,lineNo)      
      lineNo+=1
      rightPredict += rets
      sent = ''  

print rightPredict,'\t',len(entMentsTags),'\t', rightPredict*1.0/len(entMentsTags)