#generate figer data entMents
import numpy as np
import cPickle
import re
'''
generate entMents [start,end,typeList]
'''
def generateEntMent(tag,types):
  entMents=[]
  tagStr =''.join(tag)
  print tagStr
  
  classType = r'01*'   #Greedy  matching
  pattern = re.compile(classType)

  matchList = re.finditer(pattern,tagStr)  #very efficient layers!
  for match in matchList:
    start= match.start();end = match.end()
    print start,end
    if len(types[start])!=0:
      entMents.append(list([str(start),str(end),types[start]]))
  print 'entMents:',entMents
  return entMents
  
figer2id = cPickle.load(open('data/figer/figer2id.p','rb'))
print figer2id
input_file_obj = open('data/figer_test/exp.label')

tag=[]
types=[]
entMents=[]
for line in input_file_obj:
  if line in ['\n', '\r\n','\t\n']:
    print '---------------------------'
    print tag,types
    if len(types)!=0:
      print types
      entMents.append(generateEntMent(tag,types))
    tag=[];types=[]
  else:
    line = line.strip()
    items = line.split('\t')
    #print items
    tempType = []
    if items[1]=='O':
      tag.append('2')
      #tempType.append(114)
    elif 'B-/' in items[1]:
      tag.append('0')
      typestr = items[1][2:]
      typeitem = typestr.split(',')
      for typei in typeitem:
        tempType.append(figer2id[typei])
      
    elif 'I-/' in items[1]:
      tag.append('1')
      typeitem = typestr.split(',')
      for typei in typeitem:
        tempType.append(figer2id[typei])
    else:
      print line
      print 'tag is wrong...'
      exit(0)
    types.append(tempType)
    
cPickle.dump(entMents,open('data/figer_test/features/'+'figer_entMents.p','wb'))