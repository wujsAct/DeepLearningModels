# -*- coding: utf-8 -*-
'''
@editor: wujs
@time: 2017/3/1
function: get coreference for entity mentions
'''
import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import codecs
import cPickle
import collections
from tqdm import tqdm
from collections import Counter
from PhraseRecord import EntRecord
import argparse

'''read mid2name'''

#fname = 'data/mid2name.tsv'
#
#wikititle2fb = collections.defaultdict(list)
#fb2wikititle={}
#with codecs.open(fname,'r','utf-8') as file:
#  for line in tqdm(file):
#    line = line.strip()
#    items = line.split('\t')
#    if len(items)==2:
#      fbId = items[0]; title = items[1]  
#      fb2wikititle[fbId] = title.lower()
#      wikititle2fb[title.lower()].append(fbId)
#fb2wikititle['NIL'] = 'NIL'
parser = argparse.ArgumentParser()
parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc)', required=True)
parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)
data_args = parser.parse_args()
  
data_tag = data_args.data_tag
dir_path = data_args.dir_path

#dir_path = 'data/aida/'
#data_tag = 'testa'
f_input_ent_ments = dir_path+'features/'+data_tag+"_entms.p100"
dataEnts = cPickle.load(open(f_input_ent_ments,'rb'))
ent_Mentions = dataEnts['ent_Mentions']

data = cPickle.load(open(dir_path+'process/'+ data_tag+'.p','r'))
  
aNosNo2id = data['aNosNo2id']
id2aNosNo2id = {val:key for key,val in aNosNo2id.items()}

print 'aNosNo2id',len(aNosNo2id)
print 'ent_Mentions',len(ent_Mentions)
ent_ment_link_tags = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p','rb'))

#
#print ent_ment_link_tags


if data_tag=='train':
  ent_id = 0
if data_tag=='testa':
  ent_id = 23396
if data_tag=='testb':
  ent_id = 29313
print 'finish load all datas'
k = -1
entMentsTags={}
entMents2surfaceName={}
#entMent2line = {}
hasMid =0
allEnts = -1
for i in tqdm(range(len(ent_Mentions))):
    aNosNo = id2aNosNo2id[i]
    
    ents = ent_Mentions[i]
    
    #print i,'\tentM:',len(ents)
    for j in range(len(ents)):
      allEnts += 1
      
      
      totalCand = 0
      enti = ents[j]
      
      enti_name = enti.content.lower()
      
      start = enti.startIndex; end = enti.endIndex
      key = aNosNo + '\t' + str(start)+'\t'+str(end)
      
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]
     
      if tag == 'NIL':
        ent_id += 1
        continue
      ent_id += 1 
      k += 1
      linkingEnt = tag
      if linkingEnt == 'NIL':
        hasMid += 1
        entMentsTags[key]='NIL'
        #entMent2line[key] = line
      else:
        hasMid +=1 
        entMentsTags[key] = linkingEnt
        #entMent2line[key] = line
        entMents2surfaceName[key] = enti_name


Line2WordDict = {}
Line2entRep={}  #a little complex
with codecs.open(dir_path+'process/'+data_tag+"corefRet.txt",'r','utf-8') as file:
  for line in tqdm(file):
    line =line.strip()
    items = line.split(u"\t\t")
    entInDict={}
    wordList = []
    for enti in items:
      aNosNo, start, end, mention = enti.split(u'\t')
      key = aNosNo +'\t'+start+'\t'+end 
      if key in entMentsTags:
        for word in mention.split(u' '):
          wordList.append(word.lower())
    wordDict = Counter(wordList)
    wordDict= sorted(wordDict.iteritems(), key=lambda d:d[1], reverse = True)
    
    for enti in items:
      aNosNo, start, end, mention = enti.split('\t')
      key = aNosNo +'\t'+start+'\t'+end
      
      #representive entity is the longest entities and exist in the extracted entity mentions!
      
      flag = False
      if key in entMentsTags:
        for iment in mention.split(' '):
          if iment.lower() == wordDict[0][0]:
            flag = True
        if flag:
          entInDict[enti] = len(mention.split(" "))
      else:
        for keyi in entMentsTags:  #relative cluase deleted!
          aNosNok, startk, endk = keyi.split(u'\t')
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

print len(Line2WordDict)
for key in Line2WordDict:
  print key,'\t', Line2WordDict[key]

needMerge=[]
lineDelete=set()
with codecs.open(dir_path+'process/'+data_tag+"corefRet.txt",'r','utf-8') as file:
  for line in file:
    line =line.strip()
    items = line.split(u"\t\t")
    if line in Line2entRep:
      entRep = Line2entRep[line]
      repaNosNo, repstart, repend, repmention = entRep.split('\t')
      
      for enti in items:
        if enti != entRep:
          aNosNo, start, end, mention = enti.split(u'\t')
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
                        if len(wordDict)>0:
                          if iment.lower() == wordDict[0][0]:
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
          if itemment.lower() == entMention.lower():
            continue
          if itemment.lower() in entMention.lower() or entMention.lower() in itemment.lower():
            entMent2repMent[item] = entRep
            #print entRep,':',item
print 'reference entity nums:',len(entMent2repMent)            
#there are still a lot of entity can not be linked into

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
      print key,'\t', mention, aNoEntstr2repMent[keyrep]
      keyii = key+'\t'+mention
      entMent2repMent[keyii] = aNoEntstr2repMent[keyrep]
print len(entMent2repMent)

  
cPickle.dump(entMent2repMent,open(dir_path+'process/'+data_tag+'_entMent2repMent.p','wb'))  