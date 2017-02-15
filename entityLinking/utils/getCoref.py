# -*- coding: utf-8 -*-
'''
@editor: wujs
@time: 2017/2/14
function: get coreference for entity mentions
'''
import codecs
import collections
from tqdm import tqdm


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
entMent2repMent = {} 
with codecs.open(dir_path+"corefRet.txt",'r','utf-8') as file:
  for line in file:
    line =line.strip()
    items = line.split("\t\t")
    repMent = items[0]; iMent = items[1]
    iMentKey = '\t'.join(iMent.split('\t')[0:3]); iMentSpan = iMent.split('\t')[3]
    repMentKey = '\t'.join(repMent.split('\t')[0:3]); repMentSpan = repMent.split('\t')[3]
    '''
    @relative clause, submetion without clause
    @location: we utilize location(a longer, non-conjunctive mention is prefered if possible)
    @organization acronym dictionary
    '''
    entMent2repMent[iMentKey] = repMentKey

print "coreference length:",len(entMent2repMent)     

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
print 'entMentsTags nums:',len(entMentsTags)
print 'has mid:',hasMid

newEntsFile = codecs.open(dir_path+'new_entMen2aNosNoid.txt','w','utf-8')
for key in tqdm(entMentsTags):
  line = entMent2line[key]
  if key in entMent2repMent and entMent2repMent.get(key) in entMentsTags:  #we do not need to do entity linking for this kind of entity!
    newEntsFile.write(line+'\t'+entMent2line.get(entMent2repMent[key]).split('\t')[0]+'\n')
  else:
    newEntsFile.write(line+'\t'+line.split('\t')[0]+'\n')
    