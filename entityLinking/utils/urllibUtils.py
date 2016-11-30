# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:23:45 2016
we can add all the urllib method into this class
@author: wujs
"""
from urllib import urlencode
import urllib2
import json
from bs4 import BeautifulSoup


class urllibUtils():
  def __init__(self):
    User_Agent = 'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:43.0) Gecko/20100101 Firefox/43.0'
    header = {}
    header['User-Agent'] = User_Agent
    self.header = header
  
  def getRequest(self,url):
    req = None
    try:
      req = urllib2.Request(url,headers=self.header)
    except:
      pass
      #print 'wrong url..'   
    return req

  def getAhref(self,tags):
    co_occurence_ent=set()
    for tag in tags:
      for ai in tag.find_all('a',href=True):
        if '/wiki/' in ai['href']:
          co_occurence_ent.add(ai['href']+'\t'+ai.text)
    return co_occurence_ent
    
  def get_candidate_entities(self,searchent):
    data = {'action': 'wbsearchentities', 'search':searchent, 'language':'en','limit':'10','format': 'json'}
    data = urlencode(data)
    #search all related wiki entity
    url = 'https://www.wikidata.org/w/api.php?'+ data
    req = self.getRequest(url)
    res = json.loads(urllib2.urlopen(req,timeout=200).read())
    #print res
    candidate_ent = []
    co_occurence_ent = []
    if u'search' in res:
      for item in res[u'search']:
        description = None
        co_occurence_ent_item1 = set();co_occurence_ent_item2=set()
        co_occurence_ent_item = set()
        #print '---------------'
        ids = item[u'id']
        label = item.get('label')
        if label !=None:
          title = label
        else:
          title = searchent
        #print 'title:',title
        ent_item = {}
        ent_item['ids'] = ids
        ent_item['title'] = title
        #if entity without description, we need to delete it. Not so popular!
        
        if 'description' in item:
          description = item[u'description']
          if 'Wikipedia disambiguation page' != description:
            #diamabiguation page need to parse again
            url = 'https://www.wikidata.org/wiki/'+ids
            try:
              req = self.getRequest(url)
              properties = urllib2.urlopen(req,timeout=200).read()
              #print 'properties'
              soup = BeautifulSoup(properties,"lxml")
              
              #print 'right here'
              tags = soup.find_all('div',class_='wikibase-snakview-value wikibase-snakview-variation-valuesnak')
              co_occurence_ent_item1 = self.getAhref(tags)
            except:
              pass
              #print 'can not find wikidate page'
        else:
          url = 'https://en.wikipedia.org/wiki/'+title
          try:
            req = self.getRequest(url)
            pages = urllib2.urlopen(req,timeout=200).read()
            soup = BeautifulSoup(pages, "lxml")
            tags = soup.find_all('p')
            #print tags[0].text
            if len(tags)>=1:
              description = tags[0].text.split('.')[0]
              #print 'description:',description
              co_occurence_ent_item2 = self.getAhref(tags)
          except:
            pass
            #print 'can not find entity wikipedia pages'
        co_occurence_ent_item = co_occurence_ent_item1 | co_occurence_ent_item2
        if (description !=None) & (len(co_occurence_ent_item)!=0):
          ent_item['description'] = description
          co_occurence_ent.append(co_occurence_ent_item)
          candidate_ent.append(ent_item)
    return candidate_ent,co_occurence_ent
    
  def parseEntCandFromWikiSearch(self,searchent):
    print 'step into the parseEntCandFromWikiSearch'
    data = {'search':searchent,'limit':'10','offset':'0','profile':'default', 'title':'Special:Search','fulltext': '1'}
    data = urlencode(data)
    #search all related wiki entity
    url = 'https://en.wikipedia.org/w/index.php?'+ data
    #print url
    req = self.getRequest(url)
    pages = urllib2.urlopen(req,timeout=200).read()
    soup = BeautifulSoup(pages,"lxml")
    tags = soup.find_all('div',class_='mw-search-result-heading')
    #print 'tags',tags
    candidate_ent=[];co_occurence_ent=[]
    #print len(tags)
    if len(tags)>=1:
      for tag in tags:
        itemss = tag.find_all('a',href=True)
        for a_item in itemss:
          if '/wiki/' in a_item['href']:
            ntitle = a_item.get('title')
            #print 'ntitile',ntitle
            if (ntitle!=None) and ('disambiguation' not in ntitle):
              #if (searchent.lower() in ntitle.lower()) or (ntitle.lower() in searchent.lower()):
              #print 'equal:','ntitile',ntitle
              candidate_ent_i,co_occurence_ent_i = self.get_candidate_entities(ntitle)
              candidate_ent = candidate_ent + candidate_ent_i
              co_occurence_ent = co_occurence_ent + co_occurence_ent_i
              if len(candidate_ent)>=10:
                break;
        if len(candidate_ent)>=10:
                break;
    else:
      print 'can not find in wikipedia search!'
    return candidate_ent,co_occurence_ent
