# -*- coding: utf-8 -*-
'''
@editor: wujs
@time: 2017/2/14
function: get coreference for entity mentions
'''
import codecs
import collections
from tqdm import tqdm
from collections import Counter

'''read mid2name'''

fname = 'data/mid2name.tsv'
wikititle2fb = collections.defaultdict(list)
fb2wikititle={}
with codecs.open(fname,'r','utf-8') as file:
  for line in tqdm(file):
    line = line.strip()
    items = line.split('\t')
    if len(items)==2:
      fbId = items[0]; title = items[1]  
      fb2wikititle[fbId] = title
      wikititle2fb[title].append(fbId)
fb2wikititle['NIL'] = 'NIL'

dir_path = 'data/msnbc/'
'''read entmention 2 aNosNoid'''
entsFile = dir_path+'entMen2aNosNoid.txt'
hasMid = 0
entMentsTags={}
entMents2surfaceName={}
entMent2line = {}
with codecs.open(entsFile,'r','utf-8') as file:
  for line in file:
    line = line.strip()
    items = line.split('\t')
    entMent = items[0]; linkingEnt = items[1]; aNosNo = items[2]; start = items[3]; end = items[4]
    if "Walters" in entMent:
      print line
    key = aNosNo + '\t' + start+'\t'+end
    
    #print line
    if linkingEnt == 'NIL':
      hasMid += 1
      entMentsTags[key]='NIL'
      entMent2line[key] = line

    if linkingEnt in wikititle2fb:
      #print wikititle2fb[linkingEnt]
      hasMid +=1 
      entMentsTags[key] =wikititle2fb[linkingEnt]
      entMent2line[key] = line
      entMents2surfaceName[key] = entMent
    else:
      entMentsTags[key] ='NIL'
      entMent2line[key] = line
      entMents2surfaceName[key] = entMent
print 'entMentsTags nums:',len(entMentsTags)
print 'has mid:',hasMid
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

with codecs.open(dir_path+"corefRet.txt",'r','utf-8') as file:
  for line in file:
    line =line.strip()
    items = line.split(u"\t\t")
    entInDict={}
    wordList = []
    for enti in items:
      aNosNo, start, end, mention = enti.split(u'\t')
      key = aNosNo +'\t'+start+'\t'+end 
      if key in entMentsTags:
        for word in mention.split(u' '):
          wordList.append(word)
    wordDict = Counter(wordList)
    wordDict= sorted(wordDict.iteritems(), key=lambda d:d[1], reverse = True)
    
    for enti in items:
      aNosNo, start, end, mention = enti.split('\t')
      key = aNosNo +'\t'+start+'\t'+end
      '''
      representive entity is the longest entities and exist in the extracted entity mentions!
      '''
      flag = False
      if key in entMentsTags:
        for iment in mention.split(' '):
          if iment == wordDict[0][0]:
            flag = True
        if flag:
          entInDict[enti] = len(mention.split(" "))
      else:
        '''
        @开始尝试n-gram哈
        '''
        if len(wordDict) >0:
          for il in range(int(start),int(end)):
            for jl in range(1,int(end)-int(start)-il+1):
              keyNgram = aNosNo +'\t'+str(il)+'\t'+str(il+jl)
              if keyNgram in entMentsTags:
#                for iment in entMents2surfaceName[keyNgram].split(' '):
#                  if iment == wordDict[0][0]:
#                    flag = True
#                if flag:
                  print 'submention in this datasets:',keyNgram,entMents2surfaceName[keyNgram]
                  
    
    entInDict = sorted(entInDict.iteritems(), key=lambda d:d[1], reverse = True)
    if len(entInDict)>0:
      print entInDict[0],':',line
      


'''
newEntsFile = codecs.open(dir_path+'new_entMen2aNosNoid.txt','w','utf-8')
for key in tqdm(entMentsTags):
  line = entMent2line[key]
  if key in entMent2repMent and entMent2repMent.get(key) in entMentsTags:  #we do not need to do entity linking for this kind of entity!
    newEntsFile.write(line+'\t'+entMent2line.get(entMent2repMent[key]).split('\t')[0]+'\n')
  else:
    newEntsFile.write(line+'\t'+line.split('\t')[0]+'\n')
'''
    