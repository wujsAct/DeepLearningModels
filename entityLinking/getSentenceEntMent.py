# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
@editor: wujs
@function: to generate the final candidate
'''

import sys
import os
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
from getCandiates import funcs

  
if __name__=='__main__':
  if len(sys.argv) !=7:
    print 'usage: python pyfile dir_path inputfile candidates_entfile outputfile1 outputfile2 outputfile3'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  + sys.argv[2]
  f_input_canents = dir_path  + sys.argv[3]
  f_output1 = dir_path + sys.argv[4]
  f_output2 = dir_path + sys.argv[5]
  f_sentid2aNosNoid = codecs.open(dir_path + sys.argv[6]+'_sentid2aNosNoid.txt','w','utf-8')
  f_sents = codecs.open(dir_path + sys.argv[6]+'_sentout.txt','w','utf-8')
  
  data = cPickle.load(open(f_input_canents,'r'))
  entstr2id = data['entstr2id'];
  id2entstr = {value:key for key,value in entstr2id.items()}
  candiate_ent = data['candiate_ent'];candiate_coCurrEnts = data['candiate_coCurrEnts']
  
  
  #f_input context is: para_dict={'aNosNo2id':aNosNo2id,'id2aNosNo':id2aNosNo,'sents':sents,'tags':tags,'ents':ents,'depTrees':depTrees}
  data = cPickle.load(open(f_input,'r'))
  
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['all_sentence_list']; ents=data['ents'];sents_real=data['sents'];tags=data['tags'];mentags=data['mentags']
  print len(aNosNo2id)
  print len(sents)
  print len(sents_real)
  #print aNosNo2id
  id2aNosNo2id = {val:key for key,val in aNosNo2id.items()}
  for i in range(len(id2aNosNo2id)):
    f_sentid2aNosNoid.write(id2aNosNo2id[i]+'\n')
    sent = sents[i]
    for word in sent.split(' '):
      f_sents.write(word+'\n')
    f_sents.write('\n')
    
  f_sentid2aNosNoid.close()
  f_sents.close()
  
  
  
    
  #这个word2vec，直接利用文本本身的信息作为训练集！
#  f_output_w = codecs.open(f_output1,'w','utf-8')
#  for i in xrange(len(sents)):
#    f_output_w.write(sents[i]+'\n')
#  f_output_w.close() 
#  print 'finish generate all sentences...'
  '''
  w2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wid2fbid.p','rb'))
  wikititle2fb = cPickle.load(open('/home/wjs/demo/entityType/informationExtract/data/wtitle2fbid.p','rb'))  
  candEnt_descrip_dict={}
  for ents in candiate_ent:
    for enti in ents:
      ids = enti['ids']
      titles = enti['title']
      description = enti['description']
      
      if ids in w2fb:
        if w2fb[ids] not in candEnt_descrip_dict:
          new_descrip = titles+u' '+description.replace(titles,u'')
          print new_descrip
          f_output_w.write(new_descrip+'\n')
          candEnt_descrip_dict[w2fb[ids]] = new_descrip
      else:
        if titles in wikititle2fb:
          if wikititle2fb[titles] not in candEnt_descrip_dict:
            new_descrip = titles+u' '+description.replace(titles,u'')
            f_output_w.write(new_descrip+'\n')
            candEnt_descrip_dict[wikititle2fb[titles]] = new_descrip
            print new_descrip
  cPickle.dump(candEnt_descrip_dict,open(f_output2,'wb'))          
  '''
  
