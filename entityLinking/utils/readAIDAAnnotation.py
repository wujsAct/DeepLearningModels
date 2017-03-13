# -*- coding: utf-8 -*-
'''
@time: 2016/12/28
@editor: wujs
@description: ¶ÁÈ¡AIDA annotationµÄ½á¹û
'''
'''
@Õâ¸öÁ½¸ö¶«Î÷ÊµÖÊÊÇÒ»¸ö¶«Î÷¿©£¡
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


def getAnnotation():
  fileName = '/home/wjs/demo/entityType/informationExtract/data/aida/AIDA-YAGO2-annotations.tsv'
  annotation_datas = []
  with codecs.open(fileName,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split(u'\t')
      if len(items)>=2:
        annotation_datas.append(items)
    print 'annotation data lent:', len(annotation_datas)
  
  print len(annotation_datas)
  
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
          entmen_tag.append([adia_datas[j][2].lower(),'NIL'])
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
            item = list(adia_datas[j])
            if len(item)==7:
              if item[2].lower()==u'pds':
                print item, annotation_datas[i]
              entmen_tag.append([item[2].lower(),item[6]])
              entstr_lower2mid[item[2].lower()] = item[6]
              mid2entstr_lower[item[6]] = item[3].lower()
            else:
              entmen_tag.append([item[2].lower(),'NIL'])  #Ã»·¨mappingµ½ÖªÊ¶¿âÖÐÈ¥À²£¡
            i+=1
            j+=1
          else:
            #print annotation_datas[i]  #start with Ê§Ð§µÄ½á¹û£¡ËùÒÔ²»ÖØ¸´¼ÓÈë½øÈ¥À²£¡
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
  return annotation_datas

def reviseDatas():
  entment_tag = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p','rb'))
  entment_tag[31291]=['new york commodities desk','NIL']
  entment_tag.remove(entment_tag[34657])
  entment_tag.remove(entment_tag[30140])
  cPickle.dump(entment_tag,open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','wb'))
  
  datas = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/features/train_entms.p100','rb'))
  entments = datas['ent_Mentions']
  #print entments[10144][0].content,entments[10144][0].startIndex, entments[10144][0].endIndex
  
  #@train ents problem
  newstr = entments[10144][0].content.split(' ')[0]
  print newstr
  newstr1 = ' '.join(entments[10144][0].content.split(' ')[1:3])
  print newstr1
  newent = EntRecord(entments[10144][0].startIndex,entments[10144][0].startIndex+1)
  newent.content = newstr
  newent1 = EntRecord(entments[10144][0].startIndex+1,entments[10144][0].endIndex)
  newent1.content = newstr1
  entments[10144]=[]
  entments[10144].append(newent)
  entments[10144].append(newent1)
  entsNum=0
  for entlist in entments:
    for enti in entlist:
      entsNum += 1
  print len(entments), entsNum
  datas['ent_Mentions'] = list(entments)
  cPickle.dump(datas,open('/home/wjs/demo/entityType/informationExtract/data/aida/features/train_entms.p100_new','wb'))
  
  datas = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/process/train.p','rb'))
  entments = datas['ents']
  entments[10144]=[]
  entments[10144]=[[newent,newent1],['I-MISC','I-MISC']]
  cPickle.dump(datas,open('/home/wjs/demo/entityType/informationExtract/data/aida/process/train.p_new','wb'))
  entsNum=0
  for entlist in entments:
    #print entlist
    for enti in entlist[0]:
      entsNum += 1
  print len(entments), entsNum

#  '''
#  @revise candidata entities
#  '''
#  candEnts = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/process/train_candEnts.p','rb'))
#  entstr2id = candEnts['entstr2id']
#  
#  if newstr in entstr2id:
#    print newstr,entstr2id[newstr]
#    
#  if newstr1 in entstr2id:
#    print newstr1,entstr2id[newstr1]
annotation_datas = getAnnotation()


'''only run once'''
reviseDatas()

'''run ...'''
entment_tag = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/aida-annotation.p_new','rb'))
datas = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/aida/features/testa_entms.p100','rb'))
entments = datas['ent_Mentions']
#for entlist in entments:
#  for enti in entlist:
#    print enti.content, enti.startIndex, enti.endIndex
#ent_id = 0
ent_id = 23396 #eng.testa
#ent_id = 29313 #eng.testb
#total entment is 34929
LineNo = -1
nums = 0
for ent in entments:
  LineNo +=1
  EntNo = -1
  for enti in ent:
    EntNo += 1
    #print 'enti:',enti
    enti_name = enti.content.lower()
    if enti_name == 'england':
      print enti.content,entment_tag[ent_id],annotation_datas[ent_id]
      nums += 1
    #print  entment_tag[ent_id][0]
    if enti_name.replace(u' ','') == entment_tag[ent_id][0].replace(u' ',''):
      
      #if enti_name.replace(u' ','')==u'czech':
      #print ent_id,enti.content,len(enti.content),len(enti.content.strip()),enti.startIndex, enti.endIndex
      pass
    else:
      print 'wrong:',ent_id,enti_name,entment_tag[ent_id],entment_tag[ent_id+1],entment_tag[ent_id+2]
      print LineNo,EntNo
      exit(0)
#      if enti_name == 'detroit':
#        print 'wrong:',ent_id,enti_name,entment_tag[ent_id],entment_tag[ent_id+1],entment_tag[ent_id+2]
#        ent_id += 1
#      elif enti_name == 'chicago':
#        print 'wrong:',ent_id,enti_name,entment_tag[ent_id],entment_tag[ent_id+1],entment_tag[ent_id+2]
#        ent_id +=1
#      else:
#        
#        print 'wrong:',ent_id,enti_name,entment_tag[ent_id],entment_tag[ent_id+1],entment_tag[ent_id+2]
#        ent_id += 1
    ent_id += 1 
print nums  
print 'ent_id:',ent_id


