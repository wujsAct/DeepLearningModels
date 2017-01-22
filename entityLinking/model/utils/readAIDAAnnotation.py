# -*- coding: utf-8 -*-
'''
@time: 2016/12/28
@editor: wujs
@description: 读取AIDA annotation的结果
'''
'''
@这个两个东西实质是一个东西咯！
'''
import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')

import codecs
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
from spacy.en import English
import cPickle
import codecs
import cPickle 

fileName = '/home/wjs/demo/entityType/informationExtract/data/aida/AIDA-YAGO2-annotations.tsv'
annotation_datas = []
with codecs.open(fileName,'r','utf-8') as file:
  for line in file:
    line = line.strip()
    items = line.split(u'\t')
    if len(items)>=2:
      annotation_datas.append(items)
print 'annotation data lent:', len(annotation_datas)


fileName = '/home/wjs/demo/entityType/informationExtract/data/aida/AIDA-YAGO2-dataset.tsv'
adia_datas = []
entstr_lower2mid = {}
mid2entstr_lower={}
with codecs.open(fileName,'r','utf-8') as file:
  for line in file:
    line = line.strip()
    item = line.split(u'\t')
    if len(item) >=4:
      if item[2].startswith(item[0]) and item[1]=='B':
        adia_datas.append(item)
        #print item
    if len(item)==7:
      entstr_lower2mid[item[2].lower()] = item[6]
      mid2entstr_lower[item[6]] = item[3].lower()

total_right =0
i=0;j=0
'''
@time:2016/12/30 store entity tag in lists, every item is [entmen,mid]
'''
wrong_ent = 0
entmen_tag=[]
while(i<len(annotation_datas)):
  if len(annotation_datas[i])==2:
    entmen_tag.append([adia_datas[j][2].lower(),'NIL'])
    total_right += 1
    #print 'NIL'
    #print 'i:',annotation_datas[i]
    #print 'j:',adia_datas[j]
    i += 1 
    j += 1
  else:
    if len(annotation_datas[i])>=4:
      if len(adia_datas[j])<=4:
        entmen_tag.append([adia_datas[j][2].lower(),'NIL'])   #annotation不存在的实体啦, 但是还是抽取出来了啦！
        #print 'wrong j:',j
        #print annotation_datas[i]
        #print adia_datas[j]
        j+=1
        wrong_ent += 1
        #exit()
      else:
        if annotation_datas[i][2] == adia_datas[j][4]:
          #print 'right'
          total_right += 1
          item = adia_datas[j]
          if len(item)==7:
            if item[2].lower()==u'pds':
              print item, annotation_datas[i]
            entmen_tag.append([item[2].lower(),item[6]])
            entstr_lower2mid[item[2].lower()] = item[6]
            mid2entstr_lower[item[6]] = item[3].lower()
          else:
            entmen_tag.append([item[2].lower(),'NIL'])  #没法mapping到知识库中去啦！
          i+=1
          j+=1
        else:
          #print annotation_datas[i]  #start with 失效的结果！所以不重复加入进去啦！
          #print adia_datas[j]
          j+=1
          print 'j:',j
print j
print 'non entity mentions:', wrong_ent
print 'total entities:', len(entmen_tag)
#print entstr_lower2mid
#param_dict ={'entstr_lower2mid':entstr_lower2mid,'mid2entstr_lower':mid2entstr_lower}
cPickle.dump(entmen_tag,open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p','wb'))#
print 'annotation data lent:', len(annotation_datas)
print 'aida data lent:', len(adia_datas)
print 'total_right:',total_right

'''
@完全按照顺序去获取tag即可啦！
'''

entment_tag = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p','rb'))
datas = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/features/testa_entms.p100','rb'))
entments = datas['ent_Mentions']
#ent_id = 0
#ent_id = 23396 #eng.testa
#ent_id = 29313 #eng.testb
#total entment is 34929
for ent in entments:
  for enti in ent:
    #print 'enti:',enti
    enti_name = enti.getContent().lower()
    if enti_name.replace(u' ','') == entment_tag[ent_id][0].replace(u' ','') or enti_name.replace(u' ','')==u'czech' or enti_name.replace(u' ','')==u'netherlands':
      pass
      #print enti_name,entment_tag[ent_id]
      #print 'right'
    else:
      print 'wrong'
      print ent_id,enti_name,entment_tag[ent_id],entment_tag[ent_id+1]
      #ent_id += 1
      exit()
    ent_id += 1
print 'ent_id:',ent_id


