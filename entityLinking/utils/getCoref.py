# -*- coding: utf-8 -*-
'''
@editor: wujs
@time: 2017/2/14; revise: 6/13 add kbp datasets
function: get coreference for entity mentions
'''

import codecs
import cPickle
import collections
from tqdm import tqdm
from collections import Counter
from mongoUtils import mongoUtils
import argparse
import loadDataUtils


parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc or kbp)', required=True)
parser.add_argument('--dataset', type=str, help='train or eval(but " " for ace and msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)

data_args = parser.parse_args()
data_tag = data_args.data_tag
dir_path = data_args.dir_path
dataset = data_args.dataset

mongo = mongoUtils()
#title = 'Lake Burton (Georgia)'
#title1 = title.replace(' ','_')
#title1 = title1.replace('(',ord(u'('))
#print mongo.get_tail_from_enwikiTitle()
#exit(0)

'''read mid2name'''
wikititle2fb,fb2wikititle = loadDataUtils.getMid2WikiTitle()
if dataset!="":
  dataset = dataset +"/"

kbpEntid2Wiki = dict()
if data_tag == 'tackbp':
  with open('data/kbp/kbpentid2wiki.txt') as file:
    for line in file:
      line = line.strip()
      items = line.split("\t")
      entid = items[0]; wikiname = items[2].lower()
      kbpEntid2Wiki[entid] = wikiname

kbpQid2wiki = dict()


if data_tag=='tackbp':
  lineNo = 0
  with open(dir_path+dataset+'tac_kbp_2014_english_EDL_'+dataset+'_KB_links.tab') as file:
    for line in file:
      if lineNo!=0:
        line = line.strip()
        items = line.split('\t')
        kbpQid = items[0]; kbpEntid = items[1]
        if kbpEntid in kbpEntid2Wiki:
          kbpQid2wiki[kbpQid] = kbpEntid2Wiki[kbpEntid]  
        else:
          kbpQid2wiki[kbpQid] = kbpEntid
print 'len kbpQid2wiki:',kbpQid2wiki
      
 
  
entsFile = dir_path+dataset+'entMen2aNosNoid.txt'
hasMid = 0
entMentsTags={}
entMents2surfaceName={}
entMent2line = {}
notInFreebase = 0
with open(entsFile) as file:
  for line in file:
    line = line.strip()
    items = line.split('\t')
    entMent = items[0]; linkingEnt = items[1]; aNosNo = items[2]; start = items[3]; end = items[4]
    if data_tag=='kbp':
      linkingEnt = kbpQid2wiki[linkingEnt]
      
    if "Walters" in entMent:
      print line
    if "Walters" in entMent:
      print line
    key = aNosNo + '\t' + start+'\t'+end
    
    
    #print line
    if linkingEnt.startswith('NIL'):
    #if linkingEnt == 'NIL':
      entMentsTags[key]='NIL'
      entMent2line[key] = line
      entMents2surfaceName[key] = entMent
    else:

      if linkingEnt.lower() in wikititle2fb:
#        #print wikititle2fb[linkingEnt]
        hasMid +=1 

        entMentsTags[key] =wikititle2fb[linkingEnt.lower()]
        entMent2line[key] = line
        entMents2surfaceName[key] = entMent
      else:
        new_linkingEnt = linkingEnt.replace(' ','_')
        mids = mongo.get_tail_from_enwikiTitle(new_linkingEnt)
        print mids
        if len(mids)>=1:
          entMentsTags[key] = mids.pop()
          entMent2line[key] = line
          entMents2surfaceName[key] = entMent
        else:
          print 'not in freebase:',aNosNo,linkingEnt
          entMentsTags[key] ='NIL'
          entMent2line[key] = line
          entMents2surfaceName[key] = entMent
          notInFreebase += 1
print 'entMentsTags nums:',len(entMentsTags)
print 'not in freebase:',notInFreebase
print 'has mid:',hasMid

cPickle.dump(entMentsTags,open(dir_path+'entMentsTags.p','w'))
'''
with codecs.open(dir_path+"corefRet.txt",'r','utf-8') as file:
  for line in file:
    line =line.strip()
    items = line.split("\t\t")
    repMent = items[0]; iMent = items[1]
    iMentKey = '\t'.join(iMent.split('\t')[0:3]); iMentSpan = iMent.split('\t')[3]
    repMentKey = '\t'.join(repMent.split('\t')[0:3]); repMentSpan = repMent.split('\t')[3]
    
    #@relative clause, submetion without clause
    #@location: we utilize location(a longer, non-conjunctive mention is prefered if possible)
    #@organization acronym dictionary
    
    entMent2repMent[iMentKey] = repMentKey

print "coreference length:",len(entMent2repMent)     
'''

#just split 's and , relative clause
entMent2repMent = {}
Line2WordDict = {}
Line2entRep={}  #a little complex
with open(dir_path+"corefRet.txt") as file:
  for line in file:
    line =line.strip()
    items = line.split("\t\t")
    entInDict={}
    wordList = []
    for enti in items:
      aNosNo, start, end, mention = enti.split('\t')
      key = aNosNo +'\t'+start+'\t'+end 
      if key in entMentsTags:
        for word in mention.split(' '):
          wordList.append(word)
    wordDict = Counter(wordList)
    wordDict= sorted(wordDict.iteritems(), key=lambda d:d[1], reverse = True)
    
    for enti in items:
      aNosNo, start, end, mention = enti.split('\t')
      key = aNosNo +'\t'+start+'\t'+end
      
      #representive entity is the longest entities and exist in the extracted entity mentions!
      
      flag = False
      if key in entMentsTags:
        for iment in mention.split(' '):
          if iment == wordDict[0][0]:
            flag = True
        if flag:
          entInDict[enti] = len(mention.split(" "))
      else:
        for keyi in entMentsTags:  #relative cluase deleted!
          aNosNok, startk, endk = keyi.split('\t')
          mentionk = entMents2surfaceName[keyi]
          if aNosNok == aNosNo and int(start) >= int(startk) and int(end) <= int(endk):
            entInDict[keyi+'\t'+mentionk]=len(mentionk.split(' '))
                  
    
    entInDict = sorted(entInDict.iteritems(), key=lambda d:d[1], reverse = True)
    if len(entInDict)>0:
      if '. Nick' in entInDict[0][0]:
        print 'wrong step1...'
        exit(0)
      Line2entRep[line] = entInDict[0][0]
      Line2WordDict[line] = wordDict
      #print entInDict[0],':',line
#print 'the firest iteration:',Line2entRep.values()

needMerge=[]
lineDelete=set()
with open(dir_path+"corefRet.txt") as file:
  for line in file:
    line =line.strip()
    items = line.split("\t\t")
    if line in Line2entRep:
      entRep = Line2entRep[line]
      repaNosNo, repstart, repend, repmention = entRep.split('\t')
      
      for enti in items:
        if enti != entRep:
          aNosNo, start, end, mention = enti.split('\t')
          aNo = aNosNo.split('_')[0]
          key = aNosNo +'\t'+start+'\t'+end 
          if key not in entMentsTags:
            #ent2RepEnt[enti] = entRep
            for il in range(int(start),int(end)):
              for jl in range(1,int(end)-int(start)-il+1):
                keyNgram = aNosNo +'\t'+str(il)+'\t'+str(il+jl)
                
                if keyNgram in entMentsTags:
                  #search entRep
                  mergeListi=[]
                  for linei in Line2WordDict:
                    flag=False
                    wordDict = Line2WordDict[linei]
                    entRep1 = Line2entRep[linei]
                    entRep1MentLent = len(entRep1.split('\t')[-1].split(' '))
                    aNosNorep = entRep1.split('\t')[0].split('_')[0]
                    if aNo == aNosNorep:
                      for iment in entMents2surfaceName[keyNgram].split(' '):
                        if iment == wordDict[0][0]:
                          flag = True
                      
                      if flag:
                        if len(entMents2surfaceName[keyNgram].split(' '))>entRep1MentLent:
                          #Line2entRep.pop(linei)
                          lineDelete.add(linei)
                          linei = linei + '\t\t' + keyNgram+'\t'+entMents2surfaceName[keyNgram] +'\t\t'+ entRep1
                          if '. Nick' in entMents2surfaceName[keyNgram]:
                            print 'wrong step2...'
                            exit(0)
                          Line2entRep[linei] = keyNgram+'\t'+entMents2surfaceName[keyNgram]
                          
                        else:
                          lineDelete.add(linei)
                          #Line2entRep.pop(linei)
                          linei = linei + '\t\t' + keyNgram+'\t'+entMents2surfaceName[keyNgram] +'\t\t'+ entRep1
                          if '. Nick' in entRep1:
                            print 'wrong step3...'
                            exit(0)                                                          
                          Line2entRep[linei] = entRep1
                          
                          
                        mergeListi.append(linei)
                        #print 'submention in this datasets:',entRep1,':',keyNgram,entMents2surfaceName[keyNgram]
                  needMerge.append(mergeListi)
#print 'second process..',Line2entRep.values()                  
for key in needMerge:
  if len(mergeListi) >1:
    maxentLents = 0
    repent = ''
    finalline = ''
    for linei in needMerge:
      if linei not in lineDelete:
        lents = len(Line2entRep[linei].split('\t')[-1].split(' '))
        if lents > maxentLents:
          maxentLents = lents
          repent = Line2entRep[linei]
        finalline = finalline + linei +'\t\t'
        Line2entRep.pop(linei)
        lineDelete.append(linei)
    finalline= finalline.strip()
    Line2entRep[finalline] = repent

entMent2repMent={}

for line in Line2entRep:
  if line not in lineDelete:
    line = line.strip()
    entRep = Line2entRep[line]
    entMention = entRep.split('\t')[-1]
    
    for item in line.split('\t\t'):
      if item != entRep:
        aNosNo,start,end,mention = item.split('\t')
        itemkey = '\t'.join(item.split('\t')[0:3])
        if itemkey in entMentsTags:
          #itemment= item.split('\t')[3]
          itemment=entMents2surfaceName[itemkey]
          if itemment == entMention:
            continue
          if itemment in entMention or entMention in itemment:
            entMent2repMent[item] = entRep
            #print entRep,':',item
            
print 'reference entity nums:',len(entMent2repMent)            

print 'before:',len(entMent2repMent)
aNoEntstr2repMent={}

for item in entMent2repMent:
  aNosNo,start,end,mention = item.split('\t')
  aNo = aNosNo.split("_")[0]
  key = aNo+'\t'+mention.lower()
  aNoEntstr2repMent[key] = entMent2repMent[item]

ewai = 0
for key in entMentsTags:
  aNo = key.split('\t')[0].split('_')[0]
  mention = entMents2surfaceName[key]
  keyrep = aNo+'\t'+mention.lower()
  if keyrep in aNoEntstr2repMent:
    if key+"\t"+mention not in entMent2repMent:
      ewai += 1
      #print key,'\t', mention, aNoEntstr2repMent[keyrep]
      keyii = key+'\t'+mention
      entMent2repMent[keyii] = aNoEntstr2repMent[keyrep]
      
print len(entMent2repMent)

cPickle.dump(entMent2repMent,open(dir_path+'entMent2repMent.p','wb'))

#for key in entMent2repMent:
#  print key,entMent2repMent[key]
  

'''
@there are also some entity has no reference, such as Diaze,Saban ...
'''

entMent2repMent = cPickle.load(open(dir_path+'entMent2repMent.p','rb'))
print 'entMent2repMent lent:',len(entMent2repMent)
#for key in entMent2repMent:
#  print key,entMent2repMent[key]                     
newEntsFile = open(dir_path+'new_entMen2aNosNoid.txt','w')
for key in entMentsTags:
  line = entMent2line[key]
  aNo = key.split('\t')[0].split('_')[0]
  mention = entMents2surfaceName[key]
  item = key+'\t'+mention
  if item in entMent2repMent:# and entMent2repMent.get(key) in entMentsTags:  #we do not need to do entity linking for this kind of entity!
    repEntsMents = entMent2repMent[item].split('\t')[-1]
    #print repEntsMents
    newEntsFile.write(line+'\t'+repEntsMents+'\n')
  else:
    newEntsFile.write(line+'\t'+mention+'\n')
  

    